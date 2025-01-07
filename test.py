from pyspark.sql import SparkSession
import os
import shutil

spark = SparkSession.builder.appName("Test").getOrCreate()
data = [("Alice", 34), ("Bob", 45)]
df = spark.createDataFrame(data, ["Name", "Age"])
output_path = "output/save_file.csv"

df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

files = os.listdir(output_path)
    # Find the part file and rename it
for file in files:
    if file.startswith("part-"):
        os.rename(os.path.join(output_path, file), "output/my_data.csv")

# Optionally remove the original directory
shutil.rmtree(output_path)