from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType

# Initialize Spark Session
spark = SparkSession.builder.appName("MovieSearchEngine").getOrCreate()

sc = spark.sparkContext

plots_df = spark.read.csv(
    "MovieSummaries/plot_summaries.txt",
    sep='\t',            
    header=False         
).select(
    # Select the columns by their auto-assigned index and rename them
    col("_c0").alias("movie_id"),        
    col("_c1").alias("plot_summary_raw") 
)

plots_df.printSchema()

tokenizer = Tokenizer(inputCol="plot_summary_raw", outputCol="raw_words")
words_df = tokenizer.transform(plots_df)

remover = StopWordsRemover(inputCol="raw_words", outputCol="filtered_words")
cleaned_df = remover.transform(words_df).select("movie_id", "filtered_words")

# ... Remaining steps (show output, convert back to RDD) ...
tokenized_rdd = cleaned_df.rdd.map(tuple)
tokenized_rdd.cache()

print("----- Sample Tokenized Output -----")
print(tokenized_rdd.take(5))
print("-----------------------------------")

# Stop Spark session
sc.stop()