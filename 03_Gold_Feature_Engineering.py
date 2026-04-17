# Databricks notebook source
df = spark.table("fraud_silver")

# COMMAND ----------

df.groupBy("Class").count().display()


# COMMAND ----------

fraud_count = df.filter("Class = 1").count()
legit_count = df.filter("Class = 0").count()

ratio = legit_count / fraud_count


# COMMAND ----------

from pyspark.sql.functions import col, when

df = df.withColumn(
    "weight",
    when(col("Class") == 1, ratio).otherwise(1)
)


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

feature_cols = [c for c in df.columns if c not in ["Class", "weight"]]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df_final = assembler.transform(df)


# COMMAND ----------

df_final.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("fraud_gold")
