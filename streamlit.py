import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F

# Initialize Spark session only once
if 'spark' not in st.session_state:
    st.session_state.spark = SparkSession.builder \
        .appName("Movie Recommendation for New User") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .getOrCreate()

# Function to load data
@st.cache_resource
def load_data():
    df_rating = st.session_state.spark.read.csv("MovieLens_data/ratings.csv", header=True, inferSchema=True)
    df_movies = st.session_state.spark.read.csv("MovieLens_data/movies.csv", header=True, inferSchema=True)
    return df_rating, df_movies


# Function to generate recommendations for the new user
def generate_recommendations(user_ratings):
    # Define the schema for the new user's ratings
    schema = StructType([
        StructField("userId", IntegerType(), False),
        StructField("movieId", IntegerType(), False),
        StructField("rating", FloatType(), False)
    ])

    # Cast the ratings to ensure they are of the correct type
    user_ratings_casted = [(int(user), int(movie), float(rating)) for user, movie, rating in user_ratings]

    # Create a DataFrame for the new user's ratings using the schema
    new_user_df = st.session_state.spark.createDataFrame(user_ratings_casted, schema)

    # Read the data
    df_rating, df_movies = load_data()

    # Drop the timestamp column from df_rating to match the new user's data
    df_rating = df_rating.drop("timestamp")

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
        .select("title")

    top_recommendations_with_titles = top_recommendations_with_titles.orderBy("prediction", ascending=False)

    return top_recommendations_with_titles


# Streamlit UI for user input
st.title("Movie Recommendation System")
st.write("Please select and rate the following 10 movies:")

# Load the movie titles from the CSV
df_rating, df_movies = load_data()
movie_titles_df = df_movies.select("movieId", "title").toPandas()

# Let the user choose 10 movies
selected_movies = st.multiselect("Choose 10 movies", movie_titles_df["title"].tolist(), default=movie_titles_df["title"][:10].tolist(), max_selections=10)

# Collect ratings from the user for the selected movies
user_ratings = []
for movie_title in selected_movies:
    movie_id = movie_titles_df[movie_titles_df["title"] == movie_title]["movieId"].values[0]
    rating = st.slider(f"Rate the movie: {movie_title}", 1, 5, 3)
    user_ratings.append((999, movie_id, rating))

# When the user submits their ratings
if st.button("Get Recommendations"):
    # Get the top 5 recommendations for the new user
    top_recommendations = generate_recommendations(user_ratings)

    # Display only the movie titles (without movieId and prediction)
    st.write("Top 5 recommendations for you:")
    st.write(top_recommendations.limit(5).toPandas())

# Stop the Spark session when the app is closed
# spark.stop()  # Don't stop the Spark session here to prevent closing it during reruns
