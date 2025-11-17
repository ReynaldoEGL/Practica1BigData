from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import boto3
import os
import math
import traceback
from decimal import Decimal, InvalidOperation

try:
    import pandas as pd
except Exception as e:
    pd = None
    print("pandas not available:", e)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    print("matplotlib not available:", e)
try:
    import numpy as np
except Exception as e:
    np = None
    print("numpy not available:", e)
try:
    import seaborn as sns
except Exception as e:
    sns = None
    print("seaborn not available:", e)


DATABASE = "salesdb"                         # Glue database name
TABLE_CATALOG = "salesraw"           # Glue table name (crawler output)
S3_PATH = "s3://big-data-practica-1/raw/"    # fallback S3 path/prefix
BUCKET = "big-data-practica-1"               # bucket para plots/tmp
REGION = "us-east-1"
SUMMARY_TABLE_NAME = "sales_summary"        
TOP5_TABLE_NAME = "sales_top5_by_region"     
TEMP_DIR = f"s3://{BUCKET}/tmp/"
MAX_PANDAS_ROWS = 20000
PLOTS_PREFIX = "plots/"
TEST_PUT = False  

# Init Glue/Spark
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

print("STARTING SCRIPT - agrupación POR (category, region)")
print("Config:", {"DATABASE": DATABASE, "TABLE_CATALOG": TABLE_CATALOG, "S3_PATH": S3_PATH, "BUCKET": BUCKET, "REGION": REGION})

# boto3 resources
dynamodb = boto3.resource('dynamodb', region_name=REGION)
s3_client = boto3.client('s3', region_name=REGION)

# ---------------- utilidades ----------------
def safe_to_pandas(spark_df, max_rows=MAX_PANDAS_ROWS):
    if pd is None:
        raise RuntimeError("pandas no disponible en el entorno")
    try:
        cnt = spark_df.count()
        if cnt > max_rows:
            print(f"safe_to_pandas: DataFrame tiene {cnt} filas; limitando a {max_rows}")
            return spark_df.limit(max_rows).toPandas()
        return spark_df.toPandas()
    except Exception as e:
        print("safe_to_pandas error:", e)
        try:
            return spark_df.limit(min(10000, max_rows)).toPandas()
        except Exception as e2:
            print("safe_to_pandas fallback failed:", e2)
            raise

def upload_if_exists(local_path, bucket, key_prefix=PLOTS_PREFIX):
    if os.path.exists(local_path):
        key = key_prefix + os.path.basename(local_path)
        print(f"Uploading {local_path} -> s3://{bucket}/{key}")
        s3_client.upload_file(local_path, bucket, key)
        return f"s3://{bucket}/{key}"
    else:
        print("Archivo no encontrado:", local_path)
        return None

def to_dynamo_value(v):
    """
    Convierte valores a tipos aceptados por boto3/DynamoDB:
    - Decimal para floats, int para ints, str para strings, bool para bool,
    - None para nulls (omitidos en put_item).
    Maneja numpy/pandas types y NaN/Inf.
    """
    if v is None:
        return None

    # pandas NA
    if pd is not None:
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass

    # numpy numeric
    if np is not None:
        try:
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    return None
                return Decimal(str(fv))
        except Exception:
            pass

    # Python builtins
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        try:
            return Decimal(str(v))
        except (InvalidOperation, ValueError):
            return Decimal(int(v))
    if isinstance(v, Decimal):
        return v

    # datetime -> ISO
    try:
        import datetime as _dt
        if isinstance(v, (_dt.datetime, _dt.date)):
            return v.isoformat()
    except Exception:
        pass

    # fallback -> str
    try:
        return str(v)
    except Exception:
        return None

# ---------------- lectura: catalogo -> fallback s3 ----------------
df = None
try:
    print("Leyendo desde Glue Data Catalog:", DATABASE, TABLE_CATALOG)
    dyf = glueContext.create_dynamic_frame.from_catalog(database=DATABASE, table_name=TABLE_CATALOG)
    df = dyf.toDF()
    print("Lectura desde catalog OK. Schema:")
    df.printSchema()
except Exception as e:
    print("Lectura desde catalog falló:", e)
    print("FALLBACK: leyendo CSV desde S3:", S3_PATH)
    try:
        dyf = glueContext.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={"paths": [S3_PATH]},
            format="csv",
            format_options={"withHeader": True, "sep": ","}
        )
        df = dyf.toDF()
        print("Lectura desde S3 via DynamicFrame OK. Schema:")
        df.printSchema()
    except Exception as e2:
        print("Fallback DynamicFrame falló:", e2)
        try:
            df = spark.read.option("header","true").option("inferSchema","true").csv(S3_PATH)
            print("Lectura desde S3 via spark.read.csv OK. Schema:")
            df.printSchema()
        except Exception as e3:
            print("Todas las lecturas fallaron:", e3)
            traceback.print_exc()
            raise RuntimeError("No se pudo leer datos de Glue catalog ni S3. Abortando.")

# Debug: conteo y muestra input
try:
    cnt_all = df.count()
    print("DEBUG: filas totales en df (input):", cnt_all)
    df.show(10, truncate=False)
except Exception as e:
    print("DEBUG error conteo/mostrar df:", e)

# ---------------- normalización de columnas ----------------
print("Normalizando columnas. Columnas originales:", df.columns)
cols = set(df.columns)

# total_amount normalización
if "total_amount" not in cols:
    if "sale_amount" in cols:
        df = df.withColumn("total_amount", F.col("sale_amount").cast("double"))
        print("Mapped 'sale_amount' -> 'total_amount'")
    elif "sale_total" in cols:
        df = df.withColumn("total_amount", F.col("sale_total").cast("double"))
        print("Mapped 'sale_total' -> 'total_amount'")
    elif "amount" in cols:
        df = df.withColumn("total_amount", F.col("amount").cast("double"))
        print("Mapped 'amount' -> 'total_amount'")
    else:
        df = df.withColumn("total_amount", F.lit(0.0))
        print("No column monetaria; creado 'total_amount'=0.0")

# quantity
if "quantity" not in cols:
    if "quantity_sold" in cols:
        df = df.withColumn("quantity", F.col("quantity_sold").cast("long"))
        print("Mapped 'quantity_sold' -> 'quantity'")
    elif "qty" in cols:
        df = df.withColumn("quantity", F.col("qty").cast("long"))
        print("Mapped 'qty' -> 'quantity'")
    else:
        df = df.withColumn("quantity", F.lit(1))
        print("No quantity; creado 'quantity'=1 (fallback)")

# profit_margin
if "profit_margin" not in cols:
    if "profit" in cols:
        df = df.withColumn("profit_margin", (F.col("profit").cast("double") / F.when(F.col("total_amount") == 0, F.lit(1.0)).otherwise(F.col("total_amount").cast("double"))) * 100)
        print("Calculated 'profit_margin' from 'profit' and 'total_amount'")
    elif "margin" in cols:
        df = df.withColumn("profit_margin", F.col("margin").cast("double"))
        print("Mapped 'margin' -> 'profit_margin'")
    else:
        df = df.withColumn("profit_margin", F.lit(0.0))
        print("No margin/profit; creado 'profit_margin'=0.0")

# region
if "region" not in cols:
    if "sale_region" in cols:
        df = df.withColumn("region", F.col("sale_region").cast("string"))
        print("Mapped 'sale_region' -> 'region'")
    else:
        df = df.withColumn("region", F.lit("UNKNOWN"))
        print("No region found; creado 'region'='UNKNOWN'")

# product_id / product_name
if "product_id" not in cols:
    if "sale_id" in cols:
        df = df.withColumn("product_id", F.col("sale_id").cast("string"))
        print("Mapped 'sale_id' -> 'product_id'")
    else:
        df = df.withColumn("product_id", F.lit("UNKNOWN_PRODUCT"))
        print("No product_id; creado 'product_id'='UNKNOWN_PRODUCT'")

if "product_name" not in cols:
    if "item_name" in cols:
        df = df.withColumn("product_name", F.col("item_name").cast("string"))
        print("Mapped 'item_name' -> 'product_name'")
    else:
        df = df.withColumn("product_name", F.lit("UNKNOWN_NAME"))
        print("No product_name; creado 'product_name'='UNKNOWN_NAME'")

# asegurar tipos
df = df.withColumn("total_amount", F.col("total_amount").cast("double"))
df = df.withColumn("quantity", F.col("quantity").cast("long"))
df = df.withColumn("profit_margin", F.col("profit_margin").cast("double"))
df = df.withColumn("product_id", F.col("product_id").cast("string"))
df = df.withColumn("product_name", F.col("product_name").cast("string"))
df = df.withColumn("region", F.col("region").cast("string"))
print("Columns after normalization:", df.columns)
try:
    df.select("product_id","product_name","region","total_amount","quantity","profit_margin").show(5, truncate=False)
except Exception as e:
    print("Preview failed:", e)

# ---------------- TRANSFORMACIONES ---------
print("Computando summary por (category, region)...")
summary_by_category_region = df.groupBy("category", "region").agg(
    F.sum("total_amount").alias("total_sales"),
    F.avg("profit_margin").alias("avg_profit_margin"),
    F.sum("quantity").alias("total_quantity")
)

print("Computando summary por region (solo region totals)...")
summary_by_region = df.groupBy("region").agg(
    F.sum("quantity").alias("total_quantity"),
    F.sum("total_amount").alias("total_sales")
)

# Top 5 por region (productos)
top5 = None
if {"region", "product_id", "product_name", "quantity"}.issubset(set(df.columns)):
    print("Computando top5 por region (products based on units sold)...")
    top_by_region = df.groupBy("region", "product_id", "product_name").agg(
        F.sum(F.col("quantity")).alias("units_sold"),
        F.sum(F.col("total_amount")).alias("total_revenue")
    )
    w = Window.partitionBy("region").orderBy(F.desc("units_sold"))
    top5 = top_by_region.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= 5)
else:
    print("Omitiendo top5: faltan columnas product-level.")

# ---------------- debug counts BEFORE write ----------------
try:
    cnt_summary = summary_by_category_region.count() if summary_by_category_region is not None else 0
    print("DEBUG: summary_by_category_region count =", cnt_summary)
    if cnt_summary > 0:
        summary_by_category_region.show(10, truncate=False)
except Exception as e:
    print("DEBUG: error mostrando summary_by_category_region:", e)

try:
    cnt_region = summary_by_region.count() if summary_by_region is not None else 0
    print("DEBUG: summary_by_region count =", cnt_region)
    if cnt_region > 0:
        summary_by_region.show(10, truncate=False)
except Exception as e:
    print("DEBUG: error mostrando summary_by_region:", e)

try:
    cnt_top5 = top5.count() if top5 is not None else 0
    print("DEBUG: top5 count =", cnt_top5)
    if cnt_top5 > 0:
        top5.show(10, truncate=False)
except Exception as e:
    print("DEBUG: error mostrando top5:", e)

# ---------------- Escritura a DynamoDB (Decimal-safe) ----------------
def write_summary_category_region(spark_df, table_name):
    """
    Escribe filas (category, region, total_sales, avg_profit_margin, total_quantity) en DynamoDB.
    Usa to_dynamo_value para convertir floats a Decimal.
    """
    print(f"Escribiendo summary por category+region en tabla: {table_name}")
    table = dynamodb.Table(table_name)
    if pd is not None:
        try:
            pdf = safe_to_pandas(spark_df)
            pdf = pdf.where(pd.notnull(pdf), None)
            with table.batch_writer() as batch:
                for _, row in pdf.iterrows():
                    item = {}
                    # claves requeridas
                    cat_val = row.get('category') if 'category' in pdf.columns else None
                    region_val = row.get('region') if 'region' in pdf.columns else None
                    item['category'] = to_dynamo_value(cat_val) or "UNKNOWN_CATEGORY"
                    item['region'] = to_dynamo_value(region_val) or "UNKNOWN_REGION"
                    # agregados
                    if 'total_sales' in pdf.columns and row.get('total_sales') is not None:
                        v = to_dynamo_value(row.get('total_sales'))
                        if v is not None:
                            item['total_sales'] = v
                    if 'avg_profit_margin' in pdf.columns and row.get('avg_profit_margin') is not None:
                        v = to_dynamo_value(row.get('avg_profit_margin'))
                        if v is not None:
                            item['avg_profit_margin'] = v
                    if 'total_quantity' in pdf.columns and row.get('total_quantity') is not None:
                        v = to_dynamo_value(row.get('total_quantity'))
                        if v is not None:
                            item['total_quantity'] = v
                    batch.put_item(Item={k: v for k, v in item.items() if v is not None})
            print("Summary category+region escrito en DynamoDB.")
            return
        except Exception as e:
            print("Write summary via pandas failed:", e)
            traceback.print_exc()
            print("Falling back to spark.collect()")
    # fallback (spark.collect)
    rows = spark_df.collect()
    with table.batch_writer() as batch:
        for r in rows:
            d = r.asDict()
            item = {}
            item['category'] = to_dynamo_value(d.get('category')) or "UNKNOWN_CATEGORY"
            item['region'] = to_dynamo_value(d.get('region')) or "UNKNOWN_REGION"
            if d.get('total_sales') is not None:
                v = to_dynamo_value(d.get('total_sales'))
                if v is not None:
                    item['total_sales'] = v
            if d.get('avg_profit_margin') is not None:
                v = to_dynamo_value(d.get('avg_profit_margin'))
                if v is not None:
                    item['avg_profit_margin'] = v
            if d.get('total_quantity') is not None:
                v = to_dynamo_value(d.get('total_quantity'))
                if v is not None:
                    item['total_quantity'] = v
            batch.put_item(Item={k: v for k, v in item.items() if v is not None})
    print("Summary category+region escrito (fallback).")

def write_top5_to_dynamodb(top5_spark_df, table_name):
    if top5_spark_df is None:
        print("No hay top5 para escribir.")
        return
    print(f"Escribiendo top5 en tabla: {table_name}")
    table = dynamodb.Table(table_name)
    if pd is not None:
        try:
            pdf = safe_to_pandas(top5_spark_df)
            pdf = pdf.where(pd.notnull(pdf), None)
            with table.batch_writer() as batch:
                for _, row in pdf.iterrows():
                    item = {}
                    item['region'] = to_dynamo_value(row.get("region")) or "UNKNOWN_REGION"
                    item['product_id'] = to_dynamo_value(row.get("product_id")) or "UNKNOWN_PRODUCT"
                    if 'product_name' in pdf.columns and row.get('product_name') is not None:
                        v = to_dynamo_value(row.get('product_name'))
                        if v is not None:
                            item['product_name'] = v
                    if 'units_sold' in pdf.columns and row.get('units_sold') is not None:
                        v = to_dynamo_value(row.get('units_sold'))
                        if v is not None:
                            item['units_sold'] = v
                    if 'total_revenue' in pdf.columns and row.get('total_revenue') is not None:
                        v = to_dynamo_value(row.get('total_revenue'))
                        if v is not None:
                            item['total_revenue'] = v
                    if 'rank' in pdf.columns and row.get('rank') is not None:
                        v = to_dynamo_value(row.get('rank'))
                        if v is not None:
                            item['rank'] = v
                    batch.put_item(Item={k: v for k, v in item.items() if v is not None})
            print("Top5 escrito en DynamoDB.")
            return
        except Exception as e:
            print("Write top5 via pandas failed:", e)
            traceback.print_exc()
            print("Falling back to spark.collect()")
    rows = top5_spark_df.collect()
    with table.batch_writer() as batch:
        for r in rows:
            d = r.asDict()
            item = {}
            item['region'] = to_dynamo_value(d.get('region')) or "UNKNOWN_REGION"
            item['product_id'] = to_dynamo_value(d.get('product_id')) or "UNKNOWN_PRODUCT"
            if d.get('product_name') is not None:
                v = to_dynamo_value(d.get('product_name'))
                if v is not None:
                    item['product_name'] = v
            if d.get('units_sold') is not None:
                v = to_dynamo_value(d.get('units_sold'))
                if v is not None:
                    item['units_sold'] = v
            if d.get('total_revenue') is not None:
                v = to_dynamo_value(d.get('total_revenue'))
                if v is not None:
                    item['total_revenue'] = v
            if d.get('rank') is not None:
                v = to_dynamo_value(d.get('rank'))
                if v is not None:
                    item['rank'] = v
            batch.put_item(Item={k: v for k, v in item.items() if v is not None})
    print("Top5 escrito (fallback).")

# ---------------- Ejecutar escrituras ----------------
try:
    write_summary_category_region(summary_by_category_region, SUMMARY_TABLE_NAME)
except Exception as e:
    print("Error writing summary category+region to DynamoDB:", e)
    traceback.print_exc()

try:
    write_top5_to_dynamodb(top5, TOP5_TABLE_NAME)
except Exception as e:
    print("Error writing top5 to DynamoDB:", e)
    traceback.print_exc()

# TEST PUT desde Glue (solo si TEST_PUT=True)
if TEST_PUT:
    try:
        test_table = dynamodb.Table(SUMMARY_TABLE_NAME)
        resp = test_table.put_item(
            Item={
                "category": "GLUE_TEST",
                "region": "TEST",
                "total_sales": Decimal("1"),
                "avg_profit_margin": Decimal("0"),
                "total_quantity": 1
            }
        )
        print("DEBUG: put_item test response:", resp)
    except Exception as e:
        print("DEBUG: put_item test FAILED:", e)

if plt is not None and pd is not None:
    try:
        # 1) Total Sales by Category (agregado por category sumando regiones)
        try:
            agg_cat = summary_by_category_region.groupBy("category").agg(F.sum("total_sales").alias("total_sales"))
            pdf_cat = safe_to_pandas(agg_cat)
            if not pdf_cat.empty:
                pdf_cat_sorted = pdf_cat.sort_values("total_sales", ascending=False)
                plt.figure(figsize=(10,6))
                plt.bar(pdf_cat_sorted['category'].astype(str), pdf_cat_sorted['total_sales'].astype(float))
                plt.title("Total Sales by Category")
                plt.xlabel("Category")
                plt.ylabel("Total Sales ($)")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                out1 = "/tmp/total_sales_by_category.png"
                plt.savefig(out1)
                plt.close()
                upload_if_exists(out1, BUCKET)
        except Exception as e:
            print("Plot total sales failed:", e, traceback.format_exc())

        # 2) Average Profit Margin by Category (promedio ponderado por region no trivial -> promedio simple sobre agregados)
        try:
            # usamos avg_profit_margin aggregated per (category,region) -> luego promedio simple por category
            pdf_margin = safe_to_pandas(summary_by_category_region.select("category","avg_profit_margin"))
            if not pdf_margin.empty:
                pdf_margin_grouped = pdf_margin.groupby("category")["avg_profit_margin"].mean().reset_index().sort_values("avg_profit_margin", ascending=False)
                plt.figure(figsize=(10,6))
                plt.bar(pdf_margin_grouped['category'].astype(str), pdf_margin_grouped['avg_profit_margin'].astype(float))
                plt.title("Average Profit Margin by Category")
                plt.xlabel("Category")
                plt.ylabel("Profit Margin")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                out2 = "/tmp/avg_profit_margin_by_category.png"
                plt.savefig(out2)
                plt.close()
                upload_if_exists(out2, BUCKET)
        except Exception as e:
            print("Plot avg margin failed:", e, traceback.format_exc())

        # 3) Total quantity sold by region
        try:
            if summary_by_region is not None:
                pdf_region = safe_to_pandas(summary_by_region)
                if not pdf_region.empty and 'total_quantity' in pdf_region.columns:
                    pdf_region_sorted = pdf_region.sort_values("total_quantity", ascending=True)
                    plt.figure(figsize=(8,6))
                    plt.barh(pdf_region_sorted['region'].astype(str), pdf_region_sorted['total_quantity'].astype(float))
                    plt.title("Total Quantity Sold by Region")
                    plt.xlabel("Quantity")
                    plt.ylabel("Region")
                    plt.tight_layout()
                    out3 = "/tmp/total_quantity_by_region.png"
                    plt.savefig(out3)
                    plt.close()
                    upload_if_exists(out3, BUCKET)
        except Exception as e:
            print("Plot region failed:", e, traceback.format_exc())

        # 4) Heatmap Region vs Category (total_sales)
        try:
            pivot_spark = df.groupBy("region", "category").agg(F.sum("total_amount").alias("total_sales"))
            pdf_pivot = safe_to_pandas(pivot_spark)
            if not pdf_pivot.empty:
                pivot_table = pdf_pivot.pivot_table(index="region", columns="category", values="total_sales", aggfunc="sum", fill_value=0)
                if sns is not None:
                    plt.figure(figsize=(10,8))
                    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu")
                    plt.title("Total Sales Heatmap (Region vs Category)")
                    out4 = "/tmp/heatmap_region_category.png"
                    plt.tight_layout()
                    plt.savefig(out4)
                    plt.close()
                    upload_if_exists(out4, BUCKET)
                else:
                    # fallback básico sin seaborn
                    if np is not None and not pivot_table.empty:
                        mat = pivot_table.values
                        fig, ax = plt.subplots(figsize=(10,8))
                        im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap='YlGnBu')
                        ax.set_xticks(range(len(pivot_table.columns)))
                        ax.set_xticklabels([str(x) for x in pivot_table.columns], rotation=45, ha='right')
                        ax.set_yticks(range(len(pivot_table.index)))
                        ax.set_yticklabels([str(x) for x in pivot_table.index])
                        for (i, j), val in np.ndenumerate(mat):
                            ax.text(j, i, f"{int(val)}", ha='center', va='center', color='black', fontsize=8)
                        fig.colorbar(im)
                        plt.title("Total Sales Heatmap (Region vs Category)")
                        out4 = "/tmp/heatmap_region_category.png"
                        plt.tight_layout()
                        plt.savefig(out4)
                        plt.close()
                        upload_if_exists(out4, BUCKET)
        except Exception as e:
            print("Plot heatmap failed:", e, traceback.format_exc())

    except Exception as e:
        print("Unexpected plotting exception:", e, traceback.format_exc())
else:
    print("Skipping plotting: pandas o matplotlib no disponibles en el entorno.")

print("SCRIPT FINISHED")

