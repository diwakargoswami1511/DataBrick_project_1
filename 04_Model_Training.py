# Databricks notebook source
df = spark.table("fraud_gold")

train, test = df.randomSplit([0.8, 0.2], seed=42)


# COMMAND ----------

import mlflow
mlflow.set_experiment("/Fraud_Detection_Project")


# COMMAND ----------

import os

os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/default/fraud_volume/mlflow_tmp"


# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

with mlflow.start_run():

    gbt = GBTClassifier(
        labelCol="Class",
        featuresCol="features",
        maxIter=50
    )

    model = gbt.fit(train)
    predictions = model.transform(test)

    mlflow.spark.log_model(model, "gbt")


# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

roc = BinaryClassificationEvaluator(labelCol="Class")
f1 = MulticlassClassificationEvaluator(labelCol="Class", metricName="f1")
precision = MulticlassClassificationEvaluator(labelCol="Class", metricName="weightedPrecision")
recall = MulticlassClassificationEvaluator(labelCol="Class", metricName="weightedRecall")

print("ROC AUC:", roc.evaluate(predictions))
print("F1:", f1.evaluate(predictions))
print("Precision:", precision.evaluate(predictions))
print("Recall:", recall.evaluate(predictions))
