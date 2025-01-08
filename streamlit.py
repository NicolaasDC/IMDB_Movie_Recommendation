import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Movie Recommendation for New User") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()

# Function to load data
@st.cache
def load_data():
    # Read the ratings CSV into a PySpark DataFrame
    df_rating = spark.read.csv("MovieLens_data/ratings.csv", header=True, inferSchema=True)
    # Read the movies CSV into a PySpark DataFrame
    df_movies = spark.read.csv("MovieLens_data/movies.csv", header=True, inferSchema=True)
    return df_rating, df_movies

# Function to generate recommendations for the new user
def generate_recommendations(user_ratings):
    # Read the data
    df_rating, df_movies = load_data()

    # Drop the timestamp column from df_rating to match the new user's data
    df_rating = df_rating.drop("timestamp")

    # Create a DataFrame for the new user's ratings
    new_user_df = spark.createDataFrame(user_ratings, ["userId", "movieId", "rating"])

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

    # Create a DataFrame of unrated movies for the new user
    unrated_movies = df_movies.select("movieId").distinct().filter(~df_movies.movieId.isin([row[1] for row in user_ratings]))
    new_user_unrated_movies = unrated_movies.withColumn("userId", F.lit(999))

    # Predict ratings for the unrated movies
    predictions = model.transform(new_user_unrated_movies)

    # Sort by predicted rating in descending order
    top_recommendations = predictions.orderBy("prediction", ascending=False)

    # Join the recommendations with the movies DataFrame to get the movie titles
    top_recommendations_with_titles = top_recommendations.join(df_movies, on="movieId", how="inner") \
        .select("movieId", "title", "prediction")

    top_recommendations_with_titles = top_recommendations_with_titles.orderBy("prediction", ascending=False)

    return top_recommendations_with_titles

# Streamlit UI for user input
st.title("Movie Recommendation System")
st.write("Please rate the following 10 movies:")

# Movie options
movies = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
movie_titles = ["Toy Story", "Jumanji", "Grumpier Old Men", "Waiting to Exhale", "Father of the Bride Part II", "Heat", "Sabrina", "Tom and Huck", "Sudden Death", "GoldenEye"]

# Collect ratings from the user
user_ratings = []
for i in range(10):
    rating = st.slider(f"Rate the movie: {movie_titles[i]}", 1, 5, 3)
    user_ratings.append((999, movies[i], rating))

# When the user submits their ratings
if st.button("Get Recommendations"):
    # Get the top 5 recommendations for the new user
    top_recommendations = generate_recommendations(user_ratings)

    # Display the recommendations
    st.write("Top 5 recommendations for you:")
    st.write(top_recommendations.limit(5).toPandas())

# Stop the Spark session when the app is closed
spark.stop()
