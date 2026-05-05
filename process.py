"""
CISC 886 - Cloud Computing Project
Stack Overflow Posts - PySpark Preprocessing Pipeline
EMR Cluster | us-east-1

Pipeline Steps:
  1. Load raw Parquet data from S3
  2. Separate Questions and Answers
  3. Join Questions with their Accepted Answers
  4. Clean HTML/Markdown from Body text
  5. Filter low-quality posts
  6. Build Alpaca-style prompt format
  7. EDA: length stats, score distribution, top tags
  8. Train / Validation / Test split (80 / 10 / 10)
  9. Save processed splits back to S3
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import re

# ──────────────────────────────────────────────────────────
# 0. CONFIGURATION
#    Update these paths if your S3 structure changes
# ──────────────────────────────────────────────────────────
INPUT_PATH  = "s3://25wgws-chatbot-data/raw/stackoverflow-parquet/"
OUTPUT_PATH = "s3://25wgws-chatbot-data/processed/stackoverflow/"
EDA_PATH    = "s3://25wgws-chatbot-data/eda/stackoverflow/"

# ──────────────────────────────────────────────────────────
# 1. START SPARK SESSION
#    - vectorized reader disabled for stability on large Parquet
#    - shuffle partitions set to 200 to balance parallelism
# ──────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("StackOverflow-Preprocessing") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("✅ Spark session started")

# ──────────────────────────────────────────────────────────
# 2. LOAD RAW DATA FROM S3
#    Spark reads all .parquet files in the folder automatically
# ──────────────────────────────────────────────────────────
print("📦 Loading raw data from S3...")

df_raw = spark.read.parquet(INPUT_PATH)

print(f"✅ Total raw rows loaded: {df_raw.count():,}")
df_raw.printSchema()

# ──────────────────────────────────────────────────────────
# 3. SELECT ONLY THE COLUMNS WE NEED
#    PostTypeId: 1 = Question, 2 = Answer
#    AcceptedAnswerId: only present on questions
#    ParentId: only present on answers (points to parent question)
# ──────────────────────────────────────────────────────────
df = df_raw.select(
    "Id",
    "PostTypeId",
    "ParentId",
    "AcceptedAnswerId",
    "Title",
    "Body",
    "Score",
    "Tags",
    "CreationDate"
)

# ──────────────────────────────────────────────────────────
# 4. SEPARATE QUESTIONS AND ANSWERS
#    Questions: PostTypeId=1 with a non-null AcceptedAnswerId
#    We only keep questions that have an accepted answer
#    because those represent verified, high-quality Q&A pairs
# ──────────────────────────────────────────────────────────
df_questions = df.filter(
    (F.col("PostTypeId") == 1) &
    (F.col("AcceptedAnswerId").isNotNull()) &
    (F.col("Title").isNotNull()) &
    (F.col("Body").isNotNull())
).select(
    F.col("Id").alias("question_id"),
    F.col("AcceptedAnswerId").alias("accepted_answer_id"),
    F.col("Title").alias("question_title"),
    F.col("Body").alias("question_body"),
    F.col("Score").alias("question_score"),
    F.col("Tags").alias("tags"),
    F.col("CreationDate").alias("created_at")
)

print(f"✅ Questions with accepted answers: {df_questions.count():,}")

# Answers: PostTypeId=2 with a non-null Body
df_answers = df.filter(
    (F.col("PostTypeId") == 2) &
    (F.col("Body").isNotNull())
).select(
    F.col("Id").alias("answer_id"),
    F.col("Body").alias("answer_body"),
    F.col("Score").alias("answer_score")
)

print(f"✅ Total answers: {df_answers.count():,}")

# ──────────────────────────────────────────────────────────
# 5. JOIN QUESTIONS WITH THEIR ACCEPTED ANSWERS
#    Inner join on: question.accepted_answer_id = answer.answer_id
#    This ensures every row has both a question and its best answer
# ──────────────────────────────────────────────────────────
df_pairs = df_questions.join(
    df_answers,
    df_questions["accepted_answer_id"] == df_answers["answer_id"],
    how="inner"
).drop("accepted_answer_id", "answer_id")

print(f"✅ Matched Q&A pairs after join: {df_pairs.count():,}")

# ──────────────────────────────────────────────────────────
# 6. CLEAN TEXT
#    The raw Body field contains HTML from Stack Overflow's editor.
#    We preserve code blocks and inline code by stripping only
#    the backtick fences, keeping the actual code content intact.
#    URLs are replaced with [URL] to reduce noise.
# ──────────────────────────────────────────────────────────
def clean_text(text):
    """
    Clean a single text field:
      - Preserve fenced code blocks ``` ... ``` — strip fences, keep code
      - Preserve inline code `...` — strip backticks, keep content
      - Remove all HTML tags (e.g. <p>, <code>, <pre>)
      - Replace all URLs with [URL]
      - Collapse multiple whitespace characters into one space
    """
    if text is None:
        return None
    # Preserve fenced code blocks — strip the backtick fences, keep the code
    text = re.sub(r"```(?:\w+)?\n?([\s\S]*?)```", r"\1", text)
    # Preserve inline code — strip the backticks, keep the content
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Replace URLs
    text = re.sub(r"https?://\S+", "[URL]", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Register the function as a Spark UDF so it can run on each row
clean_text_udf = F.udf(clean_text, StringType())

# Apply cleaning to all three text columns
df_pairs = df_pairs \
    .withColumn("question_title", clean_text_udf(F.col("question_title"))) \
    .withColumn("question_body",  clean_text_udf(F.col("question_body"))) \
    .withColumn("answer_body",    clean_text_udf(F.col("answer_body")))

print("✅ Text cleaning complete")

# ──────────────────────────────────────────────────────────
# 7. BUILD ALPACA-STYLE PROMPT FORMAT
#    This is the standard format used by Unsloth / LoRA fine-tuning.
#    Each training sample has three fields:
#      instruction → tells the model its role
#      input       → the question (title + body)
#      output      → the accepted answer (ground truth)
# ──────────────────────────────────────────────────────────
df_pairs = df_pairs.withColumn(
    "instruction",
    F.lit("You are a helpful technical assistant. Answer the following programming question.")
).withColumn(
    "input",
    F.concat(
        F.lit("Question: "), F.col("question_title"),
        F.lit("\n\nDetails: "), F.col("question_body")
    )
).withColumn(
    "output",
    F.col("answer_body")
)

# ──────────────────────────────────────────────────────────
# 8. QUALITY FILTERING
#    We remove low-quality samples based on:
#      - question_score >= 1  (at least one upvote from the community)
#      - answer_score   >= 1  (accepted answer also has upvotes)
#      - input length between 50 and 4000 characters
#      - output length between 50 and 4000 characters
#    The length limits remove trivially short posts and
#    extremely long posts that would exceed the LLM context window.
# ──────────────────────────────────────────────────────────
df_pairs = df_pairs \
    .withColumn("input_len",  F.length(F.col("input"))) \
    .withColumn("output_len", F.length(F.col("output")))

df_filtered = df_pairs.filter(
    (F.col("question_score") >= 1) &
    (F.col("answer_score")   >= 1) &
    (F.col("input_len")  >= 50)   &
    (F.col("input_len")  <= 4000) &
    (F.col("output_len") >= 50)   &
    (F.col("output_len") <= 4000)
)

total_filtered = df_filtered.count()
print(f"✅ Samples after quality filtering: {total_filtered:,}")

# ──────────────────────────────────────────────────────────
# 9. EXPLORATORY DATA ANALYSIS (EDA)
#    Three figures required by the assignment:
#      EDA 1 — Input/output length statistics (token proxy)
#      EDA 2 — Question score distribution
#      EDA 3 — Top 50 programming tags
#    All saved to S3 as CSV/JSON for visualization in the report.
# ──────────────────────────────────────────────────────────
print("📊 Running EDA...")

# EDA 1: Length statistics
# Approximate token count = character count / 4 (common rule of thumb)
length_stats = df_filtered.agg(
    F.min("input_len").alias("input_len_min"),
    F.avg("input_len").alias("input_len_avg"),
    F.max("input_len").alias("input_len_max"),
    F.min("output_len").alias("output_len_min"),
    F.avg("output_len").alias("output_len_avg"),
    F.max("output_len").alias("output_len_max")
)
length_stats.write.mode("overwrite").json(EDA_PATH + "length_stats/")
print("✅ EDA 1: Length statistics saved")

# EDA 2: Question score distribution
# Shows how the community rated the questions in our dataset
score_dist = df_filtered \
    .groupBy("question_score") \
    .count() \
    .orderBy("question_score")
score_dist.write.mode("overwrite").csv(EDA_PATH + "score_distribution/", header=True)
print("✅ EDA 2: Score distribution saved")

# EDA 3: Top 50 programming tags
# Tags column is an array, so we explode it to get one tag per row
df_tags = df_filtered.select(F.explode("tags").alias("tag"))
top_tags = df_tags.groupBy("tag").count().orderBy(F.desc("count")).limit(50)
top_tags.write.mode("overwrite").csv(EDA_PATH + "top_tags/", header=True)
print("✅ EDA 3: Top 50 tags saved")




import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess

os.makedirs("/mnt/tmp/plots", exist_ok=True)


# ──────────────────────────────────────────────────────────
# 📊 PLOTS (for report)
# ──────────────────────────────────────────────────────────
print("📈 Generating plots...")

# IMPORTANT: sample to avoid crashing EMR
sample_df = df_filtered.sample(fraction=0.05, seed=42)

# ───── Plot 1: Input Length Distribution
pdf1 = sample_df.select("input_len").toPandas()

plt.figure()
plt.hist(pdf1["input_len"], bins=50)
plt.title("Input Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")

plot1_path = "/mnt/tmp/plots/input_length.png"
plt.savefig(plot1_path)
plt.close()

# ───── Plot 2: Question Score Distribution
pdf2 = df_filtered.groupBy("question_score").count().limit(100).toPandas()

plt.figure()
plt.plot(pdf2["question_score"], pdf2["count"])
plt.title("Question Score Distribution")
plt.xlabel("Score")
plt.ylabel("Count")

plot2_path = "/mnt/tmp/plots/score_distribution.png"
plt.savefig(plot2_path)
plt.close()

# ───── Plot 3: Top Tags
pdf3 = df_tags.groupBy("tag").count() \
    .orderBy(F.desc("count")) \
    .limit(20).toPandas()

plt.figure()
plt.barh(pdf3["tag"], pdf3["count"])
plt.title("Top Programming Tags")

plot3_path = "/mnt/tmp/plots/top_tags.png"
plt.savefig(plot3_path)
plt.close()

print("✅ Plots saved locally")

# Upload to S3
subprocess.run(["aws", "s3", "cp", plot1_path, EDA_PATH + "input_length.png"])
subprocess.run(["aws", "s3", "cp", plot2_path, EDA_PATH + "score_distribution.png"])
subprocess.run(["aws", "s3", "cp", plot3_path, EDA_PATH + "top_tags.png"])

print("✅ Plots uploaded to S3")



# ──────────────────────────────────────────────────────────
# 10. SELECT FINAL COLUMNS FOR OUTPUT
# ──────────────────────────────────────────────────────────
df_final = df_filtered.select(
    "question_id",
    "instruction",
    "input",
    "output",
    "question_score",
    "answer_score",
    "tags",
    "created_at",
    "input_len",
    "output_len"
)

# ──────────────────────────────────────────────────────────
# 11. TRAIN / VALIDATION / TEST SPLIT  —  80 / 10 / 10
#    We use a fixed random seed (42) to make the split
#    reproducible and to prevent data leakage between splits.
#    The rand column is dropped after splitting.
# ──────────────────────────────────────────────────────────
df_final = df_final.withColumn("rand", F.rand(seed=42))

df_train = df_final.filter(F.col("rand") <  0.80).drop("rand")
df_val   = df_final.filter((F.col("rand") >= 0.80) & (F.col("rand") < 0.90)).drop("rand")
df_test  = df_final.filter(F.col("rand") >= 0.90).drop("rand")

train_count = df_train.count()
val_count   = df_val.count()
test_count  = df_test.count()

print(f"✅ Train:      {train_count:,}  ({train_count / total_filtered * 100:.1f}%)")
print(f"✅ Validation: {val_count:,}  ({val_count  / total_filtered * 100:.1f}%)")
print(f"✅ Test:       {test_count:,}  ({test_count / total_filtered * 100:.1f}%)")

# Save split counts to EDA folder for the report
spark.createDataFrame([
    ("train",      train_count),
    ("validation", val_count),
    ("test",       test_count)
], ["split", "count"]) \
    .write.mode("overwrite").csv(EDA_PATH + "split_counts/", header=True)

# ──────────────────────────────────────────────────────────
# 12. SAVE PROCESSED DATA TO S3
#    Each split is saved as Parquet — efficient for downstream
#    loading in the fine-tuning notebook.
# ──────────────────────────────────────────────────────────
print("💾 Saving processed splits to S3...")

df_train.write.mode("overwrite").parquet(OUTPUT_PATH + "train/")
df_val.write.mode("overwrite").parquet(OUTPUT_PATH + "validation/")
df_test.write.mode("overwrite").parquet(OUTPUT_PATH + "test/")

print("✅ All splits saved successfully!")
print(f"   Train      → {OUTPUT_PATH}train/")
print(f"   Validation → {OUTPUT_PATH}validation/")
print(f"   Test       → {OUTPUT_PATH}test/")
print(f"   EDA        → {EDA_PATH}")

# ──────────────────────────────────────────────────────────
# 13. PRINT A SAMPLE ROW TO VERIFY OUTPUT
# ──────────────────────────────────────────────────────────
print("\n📝 Sample training row:")
df_train.select("instruction", "input", "output").show(1, truncate=80)

# ──────────────────────────────────────────────────────────
# 14. STOP SPARK SESSION
# ──────────────────────────────────────────────────────────
spark.stop()
print("✅ Pipeline complete. Spark session stopped.")