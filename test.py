from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Movie Recommendation for New User") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()

# Read the ratings CSV into a PySpark DataFrame
df_rating = spark.read.csv("MovieLens_data/ratings.csv", header=True, inferSchema=True)

# Drop the timestamp column from df_rating to match the new user's data
df_rating = df_rating.drop("timestamp")

# Define the new user ratings
new_user_ratings = [
    (999, 1, 4.0),  # (userId, movieId, rating)
    (999, 2, 5.0),
    (999, 3, 3.5),
    (999, 4, 4.5),
    (999, 5, 2.0),
    (999, 6, 4.0),
    (999, 7, 3.0),
    (999, 8, 4.5),
    (999, 9, 4.0),
    (999, 10, 5.0)
]

# Create a DataFrame for the new user's ratings
new_user_df = spark.createDataFrame(new_user_ratings, ["userId", "movieId", "rating"])

# Append the new user's data to the existing ratings DataFrame
df_rating = df_rating.union(new_user_df)

# Split the data into training and test sets
(train_data, test_data) = df_rating.randomSplit([0.8, 0.2])

# Build ALS model
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

# Generate predictions for the new user
new_user_unrated_movies = df_rating.filter(df_rating.userId == 999)  # Filter for the new user
unrated_movies = df_rating.select("movieId").distinct().filter(~df_rating.movieId.isin([row[1] for row in new_user_ratings]))

# Create a DataFrame with the new user's unrated movies
new_user_unrated_movies = unrated_movies.withColumn("userId", F.lit(999))

# Predict ratings for the unrated movies
predictions = model.transform(new_user_unrated_movies)

# Sort by predicted rating in descending order
top_recommendations = predictions.orderBy("prediction", ascending=False)

# Read the movies CSV into a PySpark DataFrame
df_movies = spark.read.csv("MovieLens_data/movies.csv", header=True, inferSchema=True)

# Join the recommendations with the movies DataFrame to get the movie titles
top_recommendations_with_titles = top_recommendations.join(df_movies, on="movieId", how="inner") \
    .select("movieId", "title", "prediction")
    
top_recommendations_with_titles = top_recommendations_with_titles.orderBy("prediction", ascending=False)

# Display the top 5 movie recommendations for the new user with movie titles
top_recommendations_with_titles.show(5, truncate=False)

# Stop the Spark session
spark.stop()