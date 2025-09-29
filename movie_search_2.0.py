import re
import math
from pyspark import RDD
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# --- 1. GLOBAL SPARK SESSION INITIATION ---
# Handled globally outside the functions
spark = SparkSession.builder.appName("MovieSearchEngine").getOrCreate()
sc = spark.sparkContext

def clean_text_udf():
    """
    Returns a UDF for robust text cleaning (lowercasing, punctuation removal, 
    and ensuring minimal whitespace).
    """
    # Define the cleaning function
    def clean_text(text):
        if text is None:
            return ""
        
        text = str(text).lower()
        
        # 1. Replace non-alphanumeric/non-space characters with a single space
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # 2. Collapse multiple spaces into a single space
        text = re.sub(r'\s+', ' ', text)
        
        # 3. Strip leading/trailing spaces
        return text.strip()
    
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

def load_and_preprocess_queries(file_path):
    """
    Loads search terms, cleans, tokenizes, and returns the tokenized RDD: (query_id, [tokens]). 
    Uses RDD operations for efficient sequential ID assignment and filtering comments/empty lines.
    """
    try:
        # A. Load Data into RDD and assign sequential index
        # Read the file line-by-line, filter out comments/empty lines, and assign sequential index (0, 1, 2, ...)
        raw_rdd = sc.textFile(file_path).filter(
            lambda line: line.strip() and not line.strip().startswith('#')
        ).zipWithIndex().map(
            lambda x: (x[1], x[0].strip()) # (index_int, query_raw)
        ).cache()

        if raw_rdd.isEmpty():
            print(f"Warning: Query file {file_path} is empty or contains no valid queries.")
            return sc.emptyRDD()

        print(f"Loaded queries from {file_path}.")
            
        # B. Convert RDD to DataFrame for cleaning and tokenization
        queries_df = raw_rdd.toDF(["query_id", "query_raw"])

        # C. Apply cleaning and tokenization steps
        queries_cleaned_df = queries_df.withColumn(
            "query_clean", 
            clean_text_udf()(col("query_raw"))
        ).select("query_id", "query_clean")

        tokenizer = Tokenizer(inputCol="query_clean", outputCol="raw_words")
        words_df = tokenizer.transform(queries_cleaned_df)

        remover = StopWordsRemover(inputCol="raw_words", outputCol="filtered_words")
        cleaned_df = remover.transform(words_df).select("query_id", "filtered_words")
            
        # D. Convert back to RDD: (query_id, [tokens])
        tokenized_rdd = cleaned_df.rdd.map(lambda row: (row.query_id, row.filtered_words))
        tokenized_rdd.cache()
            
        return tokenized_rdd
        
    except Exception as e:
        print(f"Error loading queries from {file_path}: {e}. Returning empty RDD.")
        return sc.emptyRDD()

def calculate_tf(tokenized_rdd: RDD):
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

def calculate_df(tf_rdd: RDD, N: int):
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

def calculate_tf_idf(tf_rdd: RDD, df_rdd: RDD, N: int):
    """"
    Calculates TF-IDF vectors for each document.
    Input RDDs: 
        tf_rdd: ((term, movie_id), TF_count)
        df_rdd: (term, DF_count)
    Output RDD: (movie_id, {term: TF-IDF_value, ...})"""

    print("-> Executing RDD MapReduce for TF-IDF Vectors...")

    print("--- Calculating Document Lengths for Normalization ---")
    print(tf_rdd.take(5)) 
    print("-----------------------------------")
    print(df_rdd.take(5))
    print("-----------------------------------")
    print(f"Total documents (N): {N}")
    print("-----------------------------------")
    
    # 1. Calculate Document Lengths for Normalization
    # x[0] is (term, movie_id), x[1] is TF_count
    doc_lengths = tf_rdd.map(
        lambda x: (x[0][1], x[1]))
    
    print("--- Sample Document Lengths ---")
    print(doc_lengths.take(5))  # Sample output
    print("-------------------------------")
    
    doc_lengths_rdd = doc_lengths.reduceByKey(lambda x, y: x + y).cache()

    print("--- Sample Reduced Document Lengths ---")
    print(doc_lengths_rdd.take(5))  # Sample output
    print("-------------------------------")

    # 2. Calculate IDF: IDF = log(N / DF)
    # (x[0] is term, x[1] is DF_count)
    idf_rdd = df_rdd.map(
        lambda x: (x[0], math.log(N / x[1]))
    ).cache()

    print("--- Sample IDF Values ---")
    print(idf_rdd.take(5))  # Sample output
    print("-------------------------------")

    # 3. Normalize TF: TF = TF_count / doc_length
    # Prepare TF RDD for join: (movie_id, (term, TF_count))
    tf_prep_rdd = tf_rdd.map(lambda x: (x[0][1], (x[0][0], x[1]))).cache()

    print("--- Sample Prepared TF RDD ---")
    print(tf_prep_rdd.take(5))  # Sample output
    print("-------------------------------")

    # 4 Join with doc_lengths: (movie_id, ((term, TF_count), doc_length))
    # Calculate normalized TF
    # x[1][0][0] is term, x[0] is movie_id, x[1][0][1] is TF_count, x[1][1] is doc_length
    tf_normalized_rdd = tf_prep_rdd.join(doc_lengths_rdd).map(
        lambda x: (
            x[1][0][0],  # Key: term (x[1][0][0] = term)
            (x[0], x[1][0][1] / x[1][1]) # Value: (movie_id, normalized_TF = TF_count / doc_length)
        )
    ).cache()

    print("--- Sample Normalized TF Values ---")
    print(tf_normalized_rdd.take(5))  # Sample output
    print("-------------------------------")

    # 5. Join with IDF to get TF-IDF: ((term, movie_id), (normalized_TF, IDF))
    # Calculate TF-IDF: TF-IDF = normalized_TF * IDF
    # x[0][1] is movie_id, x[0][0] is term, x[1][0] is normalized_TF, x[1][1] is IDF
    tfidf_rdd = tf_normalized_rdd.join(idf_rdd).map(
        lambda x: (x[1][0][0], (x[0], x[1][0][1] * x[1][1]))  # (movie_id, (term, TF-IDF))
    ).cache()

    print("--- Sample TF-IDF Values Before Grouping ---")
    print(tfidf_rdd.take(5))  # Sample output
    print("-------------------------------")

    # 6. Group by movie_id to get final TF-IDF vectors
    tfidf_vector_rdd = tfidf_rdd.groupByKey().mapValues(dict)

    print("--- Sample TF-IDF Vectors Before Caching ---")
    print(tfidf_vector_rdd.take(5))  # Sample output
    print("-------------------------------")

    tfidf_vector_rdd.cache()

    print("--- Sample TF-IDF Vectors ---")
    print(tfidf_vector_rdd.take(5))  # Sample output
    print("-------------------------------")

    return tfidf_vector_rdd

def execute_tfidf_pipeline(tf_rdd: RDD, N: int):
    """
    Orchestrates the calculation of DF and TF-IDF for the document corpus.
    
    Returns:
        tuple: (doc_tfidf_vector_rdd, doc_df_rdd)
    """
    
    # 1. Calculate Document Frequency (DF)
    doc_df_rdd = calculate_df(tf_rdd, N)

    # 2. Calculate Final TF-IDF Vectors RDD
    doc_tfidf_vector_rdd = calculate_tf_idf(tf_rdd, doc_df_rdd, N)

    return doc_tfidf_vector_rdd, doc_df_rdd

def single_term_search(term: str, doc_tfidf_rdd: RDD):
    """
    Handles single-term queries (Req 4a). Filters documents and ranks by 
    the TF-IDF score of the specified term.
    
    Input:
        term (str): The single query term.
        doc_tfidf_rdd (RDD): (movie_id, {term1: score1, term2: score2, ...})
        
    Output:
        list of tuples: [(movie_id, tfidf_score), ...] (Top 10 ranked)
    """
    print(f"\n--- Searching for single term: '{term}' ---")
    
    # 1. Filter RDD to keep only documents containing the term and extract the score
    # x[1] is the dictionary of term: score
    ranking_rdd = doc_tfidf_rdd.filter(
        lambda x: term in x[1]
    ).map(
        lambda x: (x[0], x[1][term]) # (movie_id, tfidf_score)
    )

    # 2. Rank and collect top 10 results
    top_10_results = ranking_rdd.top(10, key=lambda x: x[1])

    return top_10_results

def search_engine_pipeline(query_rdd: RDD, doc_tfidf_rdd: RDD, doc_df_rdd: RDD, N_docs: int):
    """
    Processes the tokenized queries and executes the appropriate search method (4a or 4b).
    """
    
    # Convert query RDD to a list of (query_id, [tokens]) for easy iteration
    queries = query_rdd.collect()

    print("\n--- Starting Search Pipeline ---")
    
    for query_id, tokens in queries:
        query_text = " ".join(tokens)
        
        if len(tokens) == 1:
            # Requirement 4a: Single-Term Search
            term = tokens[0]
            if term == '': # Skip if the single term is empty
                continue 
            
            results = single_term_search(term, doc_tfidf_rdd)
            
            print(f"Query {query_id} ('{term}'): Top 10 Documents by TF-IDF Score:")
            if results:
                for rank, (movie_id, score) in enumerate(results):
                    print(f"  {rank+1}. Doc ID: {movie_id}, Score: {score:.4f}")
            else:
                print("  No documents found containing this term.")

        elif len(tokens) > 1:
            # Requirement 4b: Multi-Term Search (Placeholder for Cosine Similarity)
            print(f"\n--- Query {query_id} ('{query_text}'): Multi-Term Search (TBD) ---")
            # The logic for Cosine Similarity will be implemented here later.
            pass
        else:
            print(f"Query {query_id}: Ignored (no terms found).")


if __name__ == "__main__":
    
    PLOT_FILE = "MovieSummaries/plot_summaries.txt"
    QUERY_FILE = "search_terms.txt"

    print("--- 1. Starting Document TF-IDF Pipeline ---")
    
    tokenized_doc_rdd = None
    N_docs = 0

    try:
        # Load and preprocess document data
        tokenized_doc_rdd = load_and_preprocess_data(PLOT_FILE)
        N_docs = tokenized_doc_rdd.count()
        print(f"Total documents processed (N): {N_docs}")
        
    except Exception as e:
        print(f"\n!!! FATAL ERROR LOADING {PLOT_FILE}: {e}")
        # Abort if documents cannot be loaded, as the search engine cannot function.
        N_docs = 0 
        
    if N_docs > 0:
        # 2. Calculate Term Frequency (TF) for documents
        doc_tf_rdd = calculate_tf(tokenized_doc_rdd)

        # 3. Calculate DF and final TF-IDF Vectors for the entire corpus
        doc_tfidf_vector_rdd, doc_df_rdd = execute_tfidf_pipeline(doc_tf_rdd, N_docs)
        print("Document Corpus TF-IDF calculation complete.")

        # 4. Load and preprocess search queries
        query_rdd = load_and_preprocess_queries(QUERY_FILE)
        
        # 5. Execute the Search Pipeline (including Req 4a)
        search_engine_pipeline(query_rdd, doc_tfidf_vector_rdd, doc_df_rdd, N_docs)
        
    else:
        print("\nPipeline aborted: No documents available for processing.")

    # 6. Stop Spark Session
    spark.stop()