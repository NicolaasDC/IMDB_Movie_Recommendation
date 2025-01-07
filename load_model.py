from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder \
    .appName("LoadALSModel") \
    .getOrCreate()

# Path where the model was saved
model_path = "model/ALS_model"

# Load the ALS model
loaded_model = ALSModel.load(model_path)

print("ALS Model successfully loaded!")

# Test data
test_data = [(0, 1), (1, 2), (0, 2)]
test_df = spark.createDataFrame(test_data, ["userId", "itemId"])

# Rename 'itemId' to 'movieId' to match the model's training data schema
test_df = test_df.withColumnRenamed("itemId", "movieId")

# Generate predictions
predictions = loaded_model.transform(test_df)
predictions.show()

# Stop SparkSession
spark.stop()
