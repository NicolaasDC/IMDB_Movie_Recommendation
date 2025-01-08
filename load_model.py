from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

# Create SparkSession
spark = SparkSession.builder \
    .appName("LoadALSModel") \
    .master("local[*]") \
    .getOrCreate()

# Path where the model was saved
model_path = "model/ALS_model"

# Load the ALS model
loaded_model = ALSModel.load(model_path)

print("ALS Model successfully loaded!")

# Read the ratings CSV into a PySpark DataFrame
df_rating = spark.read.csv("MovieLens_data/ratings.csv", header=True, inferSchema=True)

# Select only movieId column
movie_ids = df_rating.select("movieId").distinct()

# Prompt the user to input their favorite movies and ratings
new_user_id = 999  # Replace with the new user ID
favorite_movies = []

print("Please enter your 10 favorite movies (movieId and rating) one by one:")

"""for i in range(1, 4):
    movie_id = int(input(f"Enter movieId for favorite movie #{i}: "))
    rating = float(input(f"Enter your rating (1-5) for movieId {movie_id}: "))
    favorite_movies.append((new_user_id, movie_id, rating))"""

favorite_movies = [(999, 1, 5.0), (999, 2, 4.0), (999, 3, 3.0), (999, 4, 2.0), (999, 5, 1.0), (999, 6, 5.0), (999, 7, 4.0), (999, 8, 3.0), (999, 9, 2.0), (999, 10, 1.0)]

# Create a DataFrame for the new user's ratings
new_user_ratings_df = spark.createDataFrame(favorite_movies, ["userId", "movieId", "rating"])

# Show the ratings for the new user
print("New user's ratings:")
new_user_ratings_df.show()

# Step 1: Create the DataFrame for unrated movies (all movies)
new_user_unrated_movies = movie_ids.withColumn("userId", F.lit(new_user_id)) \
    .withColumn("rating", F.lit(0.0))

# Step 2: Filter out movies that the new user has already rated
new_user_unrated_movies = new_user_unrated_movies.join(new_user_ratings_df, ["userId", "movieId"], "left_anti")

# Step 3: Generate predictions for unrated movies
predictions = loaded_model.transform(new_user_unrated_movies)

# Step 4: Sort by predicted rating in descending order
top_recommendations = predictions.orderBy("prediction", ascending=False)

# Display the top N recommendations
print("Top 5 recommendations for the user based on your favorite movies:")
top_recommendations.show(5, truncate=False)

# Stop SparkSession
spark.stop()
