# Databricks notebook source
df = spark.table("fraud_bronze")
df.display()


# COMMAND ----------

from pyspark.sql.functions import col, sum

df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show()


# COMMAND ----------

df = df.withColumn("Class", col("Class").cast("int"))


# COMMAND ----------

df.groupBy("Class").mean("Amount").show()


# COMMAND ----------

display(df.select("Amount"))


# COMMAND ----------

quantiles = df.approxQuantile("Amount", [0.25, 0.75], 0.01)

Q1 = quantiles[0]
Q3 = quantiles[1]
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


# COMMAND ----------

from pyspark.sql.functions import col

normal_df = df.filter((col("Amount") >= lower_bound) & (col("Amount") <= upper_bound))
outlier_df = df.filter((col("Amount") < lower_bound) | (col("Amount") > upper_bound))


# COMMAND ----------

print("Total Records:", df.count())
print("Normal Records:", normal_df.count())
print("Outlier Records:", outlier_df.count())


# COMMAND ----------

sample_df = df.sample(fraction=0.1, seed=42).toPandas()
normal_sample = normal_df.sample(fraction=0.1, seed=42).toPandas()
outlier_sample = outlier_df.toPandas()


# COMMAND ----------

# MAGIC %md
# MAGIC to avoid memory issues.

# COMMAND ----------

import matplotlib.pyplot as plt

class_counts = sample_df["Class"].value_counts()

plt.figure()
plt.bar(class_counts.index, class_counts.values)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution (Fraud vs Legitimate)")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Dataset is highly imbalanced.

# COMMAND ----------

plt.figure()
plt.hist(sample_df["Amount"], bins=50)
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.title("Amount Distribution (Before Outlier Removal)")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Most transactions are small.
# MAGIC
# MAGIC Few very high-value transactions exist (outliers).
# MAGIC
# MAGIC Linear models may struggle without transformation.

# COMMAND ----------

plt.figure()
plt.hist(normal_sample["Amount"], bins=50)
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.title("Amount Distribution (After Outlier Removal)")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Distribution becomes more compact.

# COMMAND ----------

plt.figure()
sample_df.boxplot(column="Amount", by="Class")
plt.title("Amount Distribution by Class")
plt.suptitle("")
plt.xlabel("Class (0=Legit, 1=Fraud)")
plt.ylabel("Transaction Amount")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Fraud transactions often show higher median or higher variance.
# MAGIC
# MAGIC Fraud cases have wider spread.
# MAGIC
# MAGIC Extreme fraud values exist.

# COMMAND ----------

import numpy as np

sample_df["LogAmount"] = np.log1p(sample_df["Amount"])

plt.figure()
plt.hist(sample_df["LogAmount"], bins=50)
plt.xlabel("Log(Transaction Amount)")
plt.ylabel("Frequency")
plt.title("Log-Transformed Amount Distribution")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Distribution becomes closer to normal.
# MAGIC
# MAGIC Skew reduces significantly.
# MAGIC
# MAGIC Data becomes more model-friendly.
# MAGIC
# MAGIC Improves:
# MAGIC
# MAGIC Logistic regression performance
# MAGIC
# MAGIC Model stability

# COMMAND ----------

fraud = sample_df[sample_df["Class"] == 1]["LogAmount"]
legit = sample_df[sample_df["Class"] == 0]["LogAmount"]

plt.figure()
plt.hist(legit, bins=50, alpha=0.5)
plt.hist(fraud, bins=50, alpha=0.5)
plt.xlabel("Log(Transaction Amount)")
plt.ylabel("Frequency")
plt.title("Fraud vs Legitimate (Log Amount)")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Need combination of PCA features + Amount.

# COMMAND ----------

df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("fraud_silver")
