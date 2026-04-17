# Databricks notebook source
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/Volumes/workspace/default/fraud_volume/creditcard.csv")

df.display()


# COMMAND ----------

df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("fraud_bronze")


# COMMAND ----------

spark.sql("SHOW TABLES IN default").show()
