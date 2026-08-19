"""
feature_extraction.py
Feature Extraction Module for Sentiment Analysis

This module converts preprocessed text into numerical features using TF-IDF.

What is TF-IDF?
    TF  (Term Frequency)              = How often a word appears in a review
    IDF (Inverse Document Frequency)  = How rare/unique a word is across ALL reviews
    TF-IDF = TF × IDF                = Important words get high scores

    Example:
        "amazing" appears often in one review but rarely overall → HIGH TF-IDF (important!)
        "the" appears everywhere in every review               → LOW TF-IDF (not useful)

Usage:
    python src/feature_extraction.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directory to save the fitted vectorizer
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


# ============================================================
# TF-IDF VECTORIZER
# ============================================================

def create_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2)):
    """
    Create a TF-IDF Vectorizer with optimized parameters.
    
    Parameters:
        max_features (int): Maximum number of features (vocabulary size).
            - 5000 keeps the most important 5000 words/phrases
            - Higher = more detail but slower
            
        ngram_range (tuple): Range of word combinations to consider.
            - (1, 1) = unigrams only:  ["good", "product", "amazing"]
            - (1, 2) = uni + bigrams:  ["good", "product", "good product", "not good"]
            - (1, 3) = uni + bi + tri: adds trigrams like "not very good"
            
            Bigrams are important because "not good" means the OPPOSITE of "good"!
    
    Returns:
        TfidfVectorizer: Configured vectorizer (not yet fitted)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,      # Keep top N words
        ngram_range=ngram_range,        # Unigrams + Bigrams
        min_df=2,                       # Ignore words that appear in less than 2 reviews
        max_df=0.95,                    # Ignore words that appear in more than 95% of reviews
        sublinear_tf=True,              # Apply log scaling to term frequency (reduces impact of very frequent words)
        strip_accents="unicode",        # Handle accented characters
    )

    print(f"   ✅ TF-IDF Vectorizer created")
    print(f"      Max features : {max_features}")
    print(f"      N-gram range : {ngram_range}")
    print(f"      Min DF       : 2 (word must appear in at least 2 docs)")
    print(f"      Max DF       : 0.95 (ignore words in >95% of docs)")

    return vectorizer


def fit_transform_tfidf(vectorizer, texts):
    """
    Fit the vectorizer on training texts and transform them to TF-IDF matrix.
    
    This should be called on TRAINING data only.
    
    Parameters:
        vectorizer (TfidfVectorizer): The vectorizer to fit
        texts (list or pd.Series): List of preprocessed text strings
    
    Returns:
        sparse matrix: TF-IDF feature matrix (rows=reviews, columns=words)
    """
    tfidf_matrix = vectorizer.fit_transform(texts)

    print(f"\n   📊 TF-IDF Matrix Created:")
    print(f"      Shape       : {tfidf_matrix.shape}")
    print(f"      Reviews     : {tfidf_matrix.shape[0]}")
    print(f"      Features    : {tfidf_matrix.shape[1]} unique words/phrases")
    print(f"      Non-zero    : {tfidf_matrix.nnz} values")
    print(f"      Sparsity    : {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")

    return tfidf_matrix


def transform_tfidf(vectorizer, texts):
    """
    Transform new texts using an already-fitted vectorizer.
    
    This should be called on TEST/NEW data.
    
    Parameters:
        vectorizer (TfidfVectorizer): Already fitted vectorizer
        texts (list or pd.Series): List of preprocessed text strings
    
    Returns:
        sparse matrix: TF-IDF feature matrix
    """
    return vectorizer.transform(texts)


# ============================================================
# FEATURE ANALYSIS
# ============================================================

def get_top_features(vectorizer, n=20):
    """
    Get the top N features (words/phrases) in the vocabulary.
    
    Parameters:
        vectorizer (TfidfVectorizer): Fitted vectorizer
        n (int): Number of top features to return
    
    Returns:
        list: Top N feature names
    """
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n   📝 Total vocabulary size: {len(feature_names)} words/phrases")
    print(f"\n   🔤 Top {n} features (alphabetical sample):")

    # Show a sample
    sample = sorted(feature_names)[:n]
    for i, word in enumerate(sample, 1):
        print(f"      {i:>2}. {word}")

    return feature_names


def get_top_tfidf_words(vectorizer, tfidf_matrix, n=15):
    """
    Get words with the highest average TF-IDF scores across all documents.
    These are the most "important" words in the dataset.
    
    Parameters:
        vectorizer (TfidfVectorizer): Fitted vectorizer
        tfidf_matrix: TF-IDF sparse matrix
        n (int): Number of top words
    
    Returns:
        pd.DataFrame: Top N words with their average TF-IDF scores
    """
    feature_names = vectorizer.get_feature_names_out()

    # Calculate mean TF-IDF score for each word across all reviews
    mean_tfidf = tfidf_matrix.mean(axis=0).A1  # Convert sparse to dense array

    # Create DataFrame and sort
    df = pd.DataFrame({
        "word": feature_names,
        "avg_tfidf": mean_tfidf,
    }).sort_values("avg_tfidf", ascending=False).head(n)

    print(f"\n   🏆 Top {n} Most Important Words (by avg TF-IDF):")
    print(f"   {'─' * 40}")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        bar = "█" * int(row["avg_tfidf"] * 200)
        print(f"      {i:>2}. {row['word']:<20} {row['avg_tfidf']:.4f}  {bar}")

    return df


# ============================================================
# SAVE / LOAD VECTORIZER
# ============================================================

def save_vectorizer(vectorizer, filepath=None):
    """Save the fitted vectorizer to disk."""
    if filepath is None:
        filepath = VECTORIZER_PATH

    with open(filepath, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\n   💾 Vectorizer saved to: {filepath}")


def load_vectorizer(filepath=None):
    """Load a fitted vectorizer from disk."""
    if filepath is None:
        filepath = VECTORIZER_PATH

    if not os.path.exists(filepath):
        print(f"   ❌ Vectorizer not found at: {filepath}")
        print(f"   Please train the model first by running: python src/model.py")
        return None

    with open(filepath, "rb") as f:
        vectorizer = pickle.load(f)
    print(f"   ✅ Vectorizer loaded from: {filepath}")
    return vectorizer


# ============================================================
# DEMO — Run this file to see TF-IDF in action
# ============================================================

if __name__ == "__main__":
    from src.preprocess import preprocess_text

    print("\n" + "=" * 65)
    print("  🔢 TF-IDF FEATURE EXTRACTION DEMO")
    print("=" * 65)

    # Sample reviews (simulating preprocessed data)
    raw_reviews = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible quality, worst product I have ever bought.",
        "It's okay, nothing special. Average product.",
        "I love this! Amazing quality and fast delivery!",
        "Do not buy this. Waste of money. Very disappointed.",
        "Great product, highly recommend to everyone!",
        "Not worth the price. Poor quality materials.",
        "Excellent! Exceeded my expectations completely!",
        "Broken on arrival. Terrible customer service.",
        "Good value for money. Happy with my purchase.",
    ]

    sentiments = ["Positive", "Negative", "Neutral", "Positive", "Negative",
                  "Positive", "Negative", "Positive", "Negative", "Positive"]

    # Step 1: Preprocess all reviews
    print("\n   📝 Step 1: Preprocessing reviews...")
    preprocessed = [preprocess_text(review) for review in raw_reviews]

    print(f"\n   Sample preprocessing:")
    for i in range(3):
        print(f"      Original : \"{raw_reviews[i][:50]}...\"")
        print(f"      Cleaned  : \"{preprocessed[i]}\"")
        print()

    # Step 2: Create TF-IDF Vectorizer
    print("   📊 Step 2: Creating TF-IDF Vectorizer...")
    vectorizer = create_tfidf_vectorizer(max_features=100, ngram_range=(1, 2))

    # Step 3: Fit and Transform
    print("\n   🔄 Step 3: Fitting & Transforming...")
    tfidf_matrix = fit_transform_tfidf(vectorizer, preprocessed)

    # Step 4: Analyze features
    print("\n   🔍 Step 4: Analyzing Features...")
    get_top_features(vectorizer, n=10)
    get_top_tfidf_words(vectorizer, tfidf_matrix, n=10)

    # Step 5: Show TF-IDF values for a single review
    print(f"\n   📋 Step 5: TF-IDF Values for Review #1:")
    print(f"      \"{raw_reviews[0][:50]}...\"")
    print(f"      {'─' * 40}")

    feature_names = vectorizer.get_feature_names_out()
    review_vector = tfidf_matrix[0].toarray().flatten()
    non_zero = [(feature_names[i], review_vector[i]) for i in range(len(review_vector)) if review_vector[i] > 0]
    non_zero.sort(key=lambda x: x[1], reverse=True)

    for word, score in non_zero:
        bar = "█" * int(score * 30)
        print(f"      {word:<20} {score:.4f}  {bar}")

    # Step 6: Transform a NEW review (simulating prediction)
    print(f"\n   🔮 Step 6: Transform a NEW review:")
    new_review = "This is the best product I have ever used! Amazing!"
    new_preprocessed = preprocess_text(new_review)
    new_vector = transform_tfidf(vectorizer, [new_preprocessed])

    print(f"      Original : \"{new_review}\"")
    print(f"      Cleaned  : \"{new_preprocessed}\"")
    print(f"      Vector   : {new_vector.shape} sparse matrix")

    print(f"\n{'=' * 65}")
    print(f"  ✅ Feature extraction module ready!")
    print(f"  📦 Import: from src.feature_extraction import create_tfidf_vectorizer")
    print(f"{'=' * 65}\n")
