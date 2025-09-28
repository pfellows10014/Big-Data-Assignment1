import re
import math
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# --- 1. GLOBAL SPARK SESSION INITIATION ---
spark = SparkSession.builder.appName("MovieSearchEngine").getOrCreate()
sc = spark.sparkContext


# --- 2. PREPROCESSING FUNCTIONS ---

def clean_text_udf():
    """Returns a UDF for punctuation removal and lowercasing."""
    def clean_text(text):
        if text is None:
            return ""
        # Remove all non-alphanumeric characters (except spaces)
        return re.sub(r'[^a-z0-9\s]', '', str(text).lower())
    
    return udf(clean_text, StringType())


def load_movie_names(metadata_path):
    """Loads the movie metadata and returns a broadcast map of {wiki_id: movie_name}."""
    # movie.metadata.tsv format: 1. Wikipedia movie ID (index 0), 2. Freebase ID, 3. Movie name (index 2)
    metadata_rdd = sc.textFile(metadata_path)\
        .map(lambda line: line.split('\t'))\
        .filter(lambda parts: len(parts) > 2)
    
    movie_names_map = metadata_rdd.map(lambda parts: (parts[0], parts[2])).collectAsMap()
    return sc.broadcast(movie_names_map)


def load_and_preprocess_data(file_path):
    """
    Loads raw data, cleans punctuation, tokenizes, removes stopwords, 
    and returns the final tokenized RDD: (movie_id, [tokens]).
    """
    # A. Load Data into DataFrame (assuming 2-column, tab-separated format)
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
    # The UDF handled the punctuation fix, so Tokenizer just splits on space.
    tokenizer = Tokenizer(inputCol="plot_summary_clean", outputCol="raw_words")
    words_df = tokenizer.transform(plots_cleaned_df)

    remover = StopWordsRemover(inputCol="raw_words", outputCol="filtered_words")
    cleaned_df = remover.transform(words_df).select("movie_id", "filtered_words")

    # D. Convert to RDD and Cache
    tokenized_rdd = cleaned_df.rdd.map(tuple)
    tokenized_rdd.cache()
    
    return tokenized_rdd


# --- 3. TF-IDF CALCULATION FUNCTIONS (Manual MapReduce) ---

def calculate_tf(tokenized_rdd):
    """
    Calculates Term Frequency (TF) using MapReduce.
    Output RDD: ((term, movie_id), TF_count)
    """
    # Explode Values: (movie_id, [t1, t2]) -> (movie_id, t1), (movie_id, t2)
    explored_rdd = tokenized_rdd.flatMapValues(lambda x: x)
    
    # Create Composite Key: (movie_id, term) -> ((term, movie_id), 1)
    # Note: Key is ((term, movie_id)) for efficient DF calculation later
    tf_map_rdd = explored_rdd.map(
        lambda x: ((x[1], x[0]), 1) 
    )

    # Sum the counts for the composite key ((term, movie_id))
    tf_rdd = tf_map_rdd.reduceByKey(lambda x, y: x + y)
    tf_rdd.cache()
    return tf_rdd


def calculate_df_and_tfidf(tf_rdd, N):
    """
    Calculates Document Frequency (DF) and the final TF-IDF scores using RDD join logic.
    Returns: df_bcast ({term: DF}), tfidf_vector_bcast ({doc: {term: tfidf}})
    """
    # 1. Calculate DF (Document Frequency)
    doc_term_rdd = tf_rdd.map(lambda x: (x[0][0], x[0][1])).distinct()
    df_rdd = doc_term_rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)
    df_map = df_rdd.collectAsMap()
    df_bcast = sc.broadcast(df_map)

    # 2. Calculate IDF (Inverse Document Frequency)
    # Uses the smoothed IDF formula: log((N + 1) / (DF + 1))
    idf_scores_rdd = df_rdd.map(
        lambda x: (x[0], math.log((N + 1) / (x[1] + 1)))
    )
    
    # 3. Prepare TF RDD for Join: (term, (doc_id, tf_score))
    tf_scores_joinable = tf_rdd.map(
        lambda x: (x[0][0], (x[0][1], x[1])) 
    )
    
    # 4. Join TF and IDF scores on the 'term' key
    tfidf_rdd_joined = tf_scores_joinable.join(idf_scores_rdd)
    
    # 5. Compute Final TF-IDF Score: ((doc_id, term), tfidf_score)
    tfidf_rdd_scored = tfidf_rdd_joined.map(
        lambda x: ((x[1][0][0], x[0]), x[1][0][1] * x[1][1])
    )
    
    # 6. Convert to Document Vector: (doc_id, {term: tfidf_score})
    tfidf_vector_rdd = tfidf_rdd_scored.map(
        lambda x: (x[0][0], (x[0][1], x[1])) 
    ).groupByKey().mapValues(dict) 
    
    tfidf_vector_map = tfidf_vector_rdd.collectAsMap()
    tfidf_vector_bcast = sc.broadcast(tfidf_vector_map)
    
    return df_bcast, tfidf_vector_bcast


# --- 4. SEARCH QUERY FUNCTIONS ---

def preprocess_query(query):
    """Applies cleaning and tokenization to the query using similar logic."""
    text = str(query).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    try:
        stopwords = set(StopWordsRemover.loadDefaultStopWords("english"))
    except:
        # Fallback list for environments that struggle to load the default
        stopwords = set(["a", "an", "the", "and", "or", "in", "on", "of", "it", "is", "for"])

    words = text.split()
    return [word for word in words if word and word not in stopwords]


def compute_cosine_similarity(query_terms, tfidf_vector_bcast, df_bcast, N):
    """
    Calculates cosine similarity between the query vector and all document vectors.
    Returns: list of top 10 (similarity_score, movie_id)
    """
    tfidf_vector_map = tfidf_vector_bcast.value
    df_map = df_bcast.value
    
    # 1. Compute Query TF-IDF Vector and Magnitude
    query_tf = {}
    for term in query_terms:
        query_tf[term] = query_tf.get(term, 0) + 1
        
    query_vector = {}
    for term, tf in query_tf.items():
        # Use the smoothed IDF formula
        df = df_map.get(term, 0) 
        idf = math.log((N + 1) / (df + 1)) 
        query_vector[term] = tf * idf
    
    query_mag = math.sqrt(sum(v**2 for v in query_vector.values()))

    if query_mag == 0:
        return []

    results = []
    # 2. Iterate over all document vectors and calculate similarity
    for doc_id, doc_vector in tfidf_vector_map.items():
        dot_product = sum(
            query_vector.get(term, 0) * doc_vector.get(term, 0)
            for term in query_terms
        )
        
        doc_mag = math.sqrt(sum(v**2 for v in doc_vector.values()))
        
        if doc_mag == 0:
            similarity = 0.0
        else:
            # Cosine Similarity Formula
            similarity = dot_product / (query_mag * doc_mag)
            
        if similarity > 0:
            results.append((similarity, doc_id)) 
            
    # 3. Sort and return top 10
    return sorted(results, key=lambda x: x[0], reverse=True)[:10]


def process_queries(search_file_path, tf_rdd, df_bcast, tfidf_vector_bcast, N, movie_names_bcast):
    """Reads queries, processes them, and prints the top 10 results."""
    movie_names = movie_names_bcast.value

    try:
        with open(search_file_path, "r") as f:
            # Filter out comments (lines starting with #) and empty lines
            queries = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    except FileNotFoundError:
        print(f"\nERROR: Search file '{search_file_path}' not found. Please create it.")
        return

    for query in queries:
        processed_terms = preprocess_query(query)
        
        print("\n" + "=" * 80)
        print(f"SEARCH QUERY: '{query}'")
        
        if not processed_terms:
            print("No searchable terms found after preprocessing.")
            continue

        if len(processed_terms) == 1:
            # (a) Single Term Query
            search_term = processed_terms[0]
            df_map = df_bcast.value
            
            def get_tfidf_for_term(tf_entry):
                ((term, doc_id), tf_count) = tf_entry
                if term == search_term:
                    df = df_map.get(term, 0)
                    idf = math.log((N + 1) / (df + 1))
                    tfidf_value = tf_count * idf
                    return (tfidf_value, doc_id)
                return (0.0, None)
            
            single_term_results = tf_rdd.map(get_tfidf_for_term)\
                                        .filter(lambda x: x[1] is not None)
            
            top_10 = single_term_results.top(10)
            
            print(f"Mode: Single Term Search for '{search_term.upper()}' (Highest TF-IDF)")
            print("-" * 80)
            for tfidf, doc_id in top_10:
                name = movie_names.get(doc_id, "Unknown Movie Name")
                print(f"  {name} (Score: {tfidf:.4f})")
            
        elif len(processed_terms) > 1:
            # (b) Multiple Term Query (Cosine Similarity)
            top_10 = compute_cosine_similarity(processed_terms, tfidf_vector_bcast, df_bcast, N)
            
            print(f"Mode: Multi-Term Search (Cosine Similarity)")
            print("-" * 80)
            for similarity, doc_id in top_10:
                name = movie_names.get(doc_id, "Unknown Movie Name")
                print(f"  {name} (Similarity: {similarity:.4f})")


# --- 5. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # Ensure these files are available in your working directory
    PLOT_FILE = "MovieSummaries/plot_summaries.txt"
    METADATA_FILE = "MovieSummaries/movie.metadata.tsv"
    SEARCH_FILE = "search_terms.txt" 

    print("--- Starting Movie Search Engine Setup ---")
    
    # 1. Load and Preprocess Data
    tokenized_rdd = load_and_preprocess_data(PLOT_FILE)
    N = tokenized_rdd.count()
    print(f"Total documents processed (N): {N}")
    
    # 2. Load Movie Names
    print("\n--- Loading Movie Metadata ---")
    movie_names_bcast = load_movie_names(METADATA_FILE)

    # 3. Manual MapReduce TF-IDF Calculation
    print("\n--- Calculating TF and TF-IDF Vectors ---")
    tf_rdd = calculate_tf(tokenized_rdd)
    df_bcast, tfidf_vector_bcast = calculate_df_and_tfidf(tf_rdd, N)
    print("TF-IDF indexing complete. Vectors available for search.")
    
    # 4. Execute Search Queries
    process_queries(SEARCH_FILE, tf_rdd, df_bcast, tfidf_vector_bcast, N, movie_names_bcast)
    
    print("\n--- Program Finished ---")
    # 5. Stop Spark Session
    spark.stop()