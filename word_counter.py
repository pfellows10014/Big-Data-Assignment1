import subprocess
import sys
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from pyspark.sql import SparkSession

# --- Configuration ---
GUTENBERG_URL = "https://www.gutenberg.org/ebooks/2701.txt.utf-8"  # Moby Dick
LOCAL_FILENAME = "moby_dick_raw.txt"

# ----------------------------------------------------------------------
# 1. NLTK Setup and Path Configuration (Crucial for Local PySpark)
# ----------------------------------------------------------------------

def setup_nltk_environment():
    """
    Downloads required NLTK data and attempts to configure the path.
    This runs on the Spark Driver.
    """
    print("Setting up NLTK data...")
    required_data = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
    
    # Optional: Explicitly add a data path if you know where the VENV stores NLTK data
    # This is often needed for local PySpark/NLTK interactions.
    # To find this path, run 'python -m nltk.downloader -d my_nltk_data all' 
    # and then use the path to 'my_nltk_data'
    # Example (adjust path as needed):
    # local_nltk_data_path = os.path.join(os.getcwd(), '.venv', 'nltk_data')
    # if os.path.isdir(local_nltk_data_path) and local_nltk_data_path not in nltk.data.path:
    #     nltk.data.path.append(local_nltk_data_path)
    #     print(f"Added local NLTK data path: {local_nltk_data_path}")

    for data_item in required_data:
        try:
            # Check for the resource, which raises LookupError if missing
            if data_item == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif data_item == 'averaged_perceptron_tagger':
                 nltk.data.find('taggers/averaged_perceptron_tagger')
            else:
                 nltk.data.find(f'corpora/{data_item}')

            # print(f"  {data_item} found.")
        except LookupError: 
            print(f"  Resource {data_item} not found. Downloading...")
            nltk.download(data_item, quiet=True)
            
    print("NLTK setup complete.")


# ----------------------------------------------------------------------
# 2. File Download Utility
# ----------------------------------------------------------------------

def download_file(url, filename):
    """Download file using wget."""
    try:
        # Check=True will raise an error if wget fails
        subprocess.run(['wget', '-O', filename, url], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Downloaded {filename} successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        return False


# ----------------------------------------------------------------------
# 3. Named Entity Extraction UDF Logic
# ----------------------------------------------------------------------

def extract_named_entities(text):
    """
    Extracts named entities from text using NLTK and returns a flat list of entity strings.
    This logic runs on the Spark Executors.
    """
    if not text:
        return []
        
    # CRITICAL: Re-import NLTK functions inside the UDF scope for executors
    from nltk import word_tokenize, pos_tag, ne_chunk
    from nltk.tree import Tree
    
    entities = []
    
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        tree = ne_chunk(pos_tags)

        # Simplified and corrected entity extraction loop
        for subtree in tree.subtrees():
            if isinstance(subtree, Tree) and subtree.label():
                # Join the tokens to form the full entity name
                entity_text = " ".join([leaf[0] for leaf in subtree.leaves()])
                
                # Optionally filter by entity type, if needed, but we keep all for counting
                # entity_type = subtree.label()
                # if entity_type in ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION']:
                entities.append(entity_text)
                
        # Filter out short or non-meaningful entities after extraction
        return [ne for ne in entities if len(ne.split()) > 1]
    
    except Exception as e:
        # In a real environment, you might log this error instead of printing
        # print(f"Error in NER for chunk: {e}")
        return []


# ----------------------------------------------------------------------
# 4. Main Execution Block
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # --- Setup ---
    spark = SparkSession.builder \
        .appName("GutenbergNLTK_NER") \
        .getOrCreate()
        
    sc = spark.sparkContext
    
    setup_nltk_environment()
    
    if not download_file(GUTENBERG_URL, LOCAL_FILENAME):
        sys.exit(1)

    # --- Data Processing ---
    
    # 1. Read the text file (each line is an element in the RDD)
    input_rdd = sc.textFile(LOCAL_FILENAME)
    
    # 2. Combine all lines into a single string on the driver for easy chunking
    full_text_str = " ".join(input_rdd.collect())
    
    # 3. Create chunks for parallel processing (Moby Dick is large)
    # 10,000 characters per chunk
    CHUNK_SIZE = 10000
    chunks = [full_text_str[i:i + CHUNK_SIZE] for i in range(0, len(full_text_str), CHUNK_SIZE)]

    # 4. Parallelize the chunks across the cluster
    chunks_rdd = sc.parallelize(chunks, numSlices=sc.defaultParallelism * 2)
    
    # 5. Apply the NER logic and flatten the results (Executor Logic)
    named_entities_rdd = chunks_rdd.flatMap(extract_named_entities)

    # 6. Count Frequencies and Sort
    named_entity_count = named_entities_rdd \
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .sortBy(lambda x: x[1], ascending=False)

    # --- Output ---
    print("\n" + "="*50)
    print("TOP 10 NAMED ENTITY COUNTS")
    print("="*50)
    
    # Collect the results to the driver and print
    top_entities = named_entity_count.take(10)
    
    for entity, count in top_entities:
        print(f"{entity:<30} | Count: {count}")
        
    print("="*50 + "\n")

    spark.stop()