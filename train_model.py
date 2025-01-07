from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Movie Recommendation System") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()

# Read the movies CSV into a PySpark DataFrame
df = spark.read.csv("MovieLens_data/movies.csv", header=True, inferSchema=True)

# Print the schema of the DataFrame
df.printSchema()

# Read the ratings CSV into a PySpark DataFrame
df_rating = spark.read.csv("MovieLens_data/ratings.csv", header=True, inferSchema=True)

df_rating.printSchema()

# Keep only relevant columns
df_rating = df_rating.select("userId", "movieId", "rating")

# Split data into training and test sets
(train_data, test_data) = df_rating.randomSplit([0.8, 0.2])

# Build ALS Model
als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop"  # Avoid NaN predictions for unseen users/movies
)

# Train the ALS model
model = als.fit(train_data)

# Make predictions on the test set
predictions = model.transform(test_data)

# Evaluate the model
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse:.4f}")

# Generate recommendations for all users
user_recs = model.recommendForAllUsers(5)  # Top 5 movie recommendations per user
user_recs.show(truncate=False)

# Generate recommendations for all movies
movie_recs = model.recommendForAllItems(5)  # Top 5 user recommendations per movie
movie_recs.show(truncate=False)

# Save the trained ALS model
try:
    model.write().save( "./model/ALS_model")
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

# Stop the Spark session
spark.stop()