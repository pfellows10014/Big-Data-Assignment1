import re
import math
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# --- 1. GLOBAL SPARK SESSION INITIATION ---
# Handled globally outside the functions
spark = SparkSession.builder.appName("MovieSearchEngine").getOrCreate()
sc = spark.sparkContext

def clean_text_udf():
    """Returns a UDF for punctuation removal and lowercasing."""
    # Define the cleaning function
    def clean_text(text):
        if text is None:
            return ""
        # Remove all non-alphanumeric characters (except spaces)
        return re.sub(r'[^a-z0-9\s]', '', str(text).lower())
    
    # Register and return the UDF
    return udf(clean_text, StringType())

def load_and_preprocess_data(file_path):
    """
    Loads raw data, cleans punctuation, tokenizes, removes stopwords, 
    and returns the final tokenized RDD: (movie_id, [tokens]).
    """
    # A. Load Data into DataFrame (uses global 'spark' variable)
    plots_df = spark.read.csv(
        file_path,
        sep='\t',            
        header=False         
    ).select(
        col("_c0").alias("movie_id"),        
        col("_c1").alias("plot_summary_raw") 
    ).filter(col("plot_summary_raw").isNotNull()) 

    # B. Apply Punctuation Cleaning UDF
    plots_cleaned_df = plots_df.withColumn(
        "plot_summary_clean", 
        clean_text_udf()(col("plot_summary_raw"))
    ).select("movie_id", "plot_summary_clean")

    # C. Tokenization and Stopword Removal (PySpark ML features)
    tokenizer = Tokenizer(inputCol="plot_summary_clean", outputCol="raw_words")
    words_df = tokenizer.transform(plots_cleaned_df)

    remover = StopWordsRemover(inputCol="raw_words", outputCol="filtered_words")
    cleaned_df = remover.transform(words_df).select("movie_id", "filtered_words")

    # D. Convert to RDD and Cache
    tokenized_rdd = cleaned_df.rdd.map(tuple)
    tokenized_rdd.cache()
    
    return tokenized_rdd

def calculate_tf(tokenized_rdd):
    """
    Calculates Term Frequency (TF) using MapReduce.
    Input RDD: (movie_id, [token1, token2, ...])
    Output RDD: ((term, movie_id), TF_count)
    """
    # (Explode Values): (movie_id, [t1, t2]) -> (movie_id, t1), (movie_id, t2)
    explored_rdd = tokenized_rdd.flatMapValues(lambda x: x)
    
    # (Create Composite Key): (movie_id, term) -> ((term, movie_id), 1)
    tf_map_rdd = explored_rdd.map(
        lambda x: ((x[1], x[0]), 1) 
    )

    # Sum the counts for the composite key ((term, movie_id))
    tf_rdd = tf_map_rdd.reduceByKey(lambda x, y: x + y)

    tf_rdd.cache()

    return tf_rdd

def calculate_df(tf_rdd, N):
    """
    Calculates Document Frequency (DF) from the TD RDD.
    Input RDD: ((term, movie_id), TF_count)
    Output RDD: (term, DF_count)
    """

    print("-> Executing RDD MapReduce for Document Frequency (DF)...")

    # (Map to Term Only): ((term, movie_id), TF_count) -> (term, 1)
    df_map_rdd = tf_rdd.map(
        lambda x: (x[0][0], 1)  # x[0] is (term, movie_id)
    )

    # Sum the counts for each term to get DF
    df_rdd = df_map_rdd.reduceByKey(lambda x, y: x + y)

    df_rdd.cache()

    return df_rdd

def calculate_tf_idf(tf_rdd, df_rdd, N):
    """"
    Calculates TF-IDF vectors for each document.
    Input RDDs: 
        tf_rdd: ((term, movie_id), TF_count)
        df_rdd: (term, DF_count)
    Output RDD: (movie_id, {term: TF-IDF_value, ...})"""

    print("-> Executing RDD MapReduce for TF-IDF Vectors...")
    
    # 1. Calculate Document Lengths for Normalization
    # x[0] is (term, movie_id), x[1] is TF_count
    doc_lengths = tf_rdd.map(
        lambda x: (x[0][1], x[1])
        .reduceByKey(lambda x, y: x + y)) # Sum TF counts per document

    # 2. Calculate IDF: IDF = log(N / DF)
    # (x[0] is term, x[1] is DF_count)
    idf_rdd = df_rdd.map(
        lambda x: (x[0], math.log(N / x[1]))
    )

    # 3. Normalize TF: TF = TF_count / doc_length
    # Prepare TF RDD for join: (movie_id, (term, TF_count))
    tf_prep_rdd = tf_rdd.map(lambda x: (x[0][1], (x[0][0], x[1])))

    # 4 Join with doc_lengths: (movie_id, ((term, TF_count), doc_length))
    # Calculate normalized TF
    # x[1][0][0] is term, x[0] is movie_id, x[1][0][1] is TF_count, x[1][1] is doc_length
    tf_normalized_rdd = tf_prep_rdd.join(doc_lengths).map(
        lambda x: ((x[1][0][0], x[0]), x[1][0][1] / x[1][1])
    )

    # 5. Join with IDF to get TF-IDF: ((term, movie_id), (normalized_TF, IDF))
    tfidf_join_rdd = tf_normalized_rdd.join(idf_rdd)

    # 6. Calculate TF-IDF: TF-IDF = normalized_TF * IDF
    # x[0][1] is movie_id, x[0][0] is term, x[1][0] is normalized_TF, x[1][1] is IDF
    tfidf_rdd = tfidf_join_rdd.map(
        lambda x: (x[0][1], (x[0][0], x[1][0] * x[1][1]))  # (movie_id, (term, TF-IDF))
    )

    # 7. Group by movie_id to get final TF-IDF vectors
    tfidf_vector_rdd = tfidf_rdd.groupByKey().mapValues(dict)

    tfidf_vector_rdd.cache()

    return tfidf_vector_rdd

if __name__ == "__main__":
    PLOT_FILE = "MovieSummaries/plot_summaries.txt"
    
    print("--- Starting Data Preprocessing Pipeline ---")
    
    # 1. Execute the main data pipeline function
    tokenized_rdd = load_and_preprocess_data(PLOT_FILE)

    # Get total document count (N) for later use
    N = tokenized_rdd.count()
    print(f"Total documents processed (N): {N}")
    
    # 2. Calculate Term Frequency (TF)
    print("\n--- Calculating Term Frequency (TF) ---")
    tf_rdd = calculate_tf(tokenized_rdd)

    # 3. Calculate DF and final TF-IDF Vectors
    print("\n--- Calculating DF and TF-IDF Vectors ---")
    df_bcast, tfidf_vector_bcast = calculate_df_and_tfidf(tf_rdd, N)
    print("DF and TF-IDF vectors successfully calculated and broadcasted.")
    
    # 6. Stop Spark Session (must be the last PySpark action)
    spark.stop()