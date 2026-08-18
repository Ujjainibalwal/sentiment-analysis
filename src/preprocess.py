"""
preprocess.py
Text Preprocessing Module for Sentiment Analysis

This module handles all text cleaning and preprocessing steps:
1. Text Cleaning      - Remove special characters, HTML, URLs, numbers
2. Tokenization       - Split text into individual words
3. Stopword Removal   - Remove common words (the, is, at, etc.)
4. Stemming           - Reduce words to their root form (running → run)
5. Lemmatization      - Reduce words to dictionary form (better → good)

Usage:
    python src/preprocess.py
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (only needed once)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Initialize tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Keep negation words — they flip sentiment meaning!
# "not good" is very different from "good"
NEGATION_WORDS = {"not", "no", "nor", "never", "neither", "nobody", "nothing",
                  "nowhere", "hardly", "barely", "scarcely", "don't", "doesn't",
                  "didn't", "won't", "wouldn't", "couldn't", "shouldn't", "isn't",
                  "aren't", "wasn't", "weren't", "can't", "cannot"}

# Remove negation words from stopwords so they are kept
stop_words = stop_words - NEGATION_WORDS


# ============================================================
# STEP 1: Text Cleaning
# ============================================================

def clean_text(text):
    """
    Clean raw text by removing noise.
    
    Steps:
        - Convert to lowercase
        - Remove HTML tags
        - Remove URLs
        - Remove email addresses
        - Remove numbers
        - Remove special characters & punctuation
        - Remove extra whitespace
    
    Parameters:
        text (str): Raw text input
    
    Returns:
        str: Cleaned text
    
    Example:
        >>> clean_text("This product is AMAZING!!! Visit http://example.com 😍")
        'this product is amazing'
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags: <br>, <p>, </div>, etc.
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs: http://..., https://..., www...
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# STEP 2: Tokenization
# ============================================================

def tokenize(text):
    """
    Split text into individual words (tokens).
    
    Parameters:
        text (str): Cleaned text
    
    Returns:
        list: List of word tokens
    
    Example:
        >>> tokenize("this product is amazing")
        ['this', 'product', 'is', 'amazing']
    """
    return word_tokenize(text)


# ============================================================
# STEP 3: Stopword Removal
# ============================================================

def remove_stopwords(tokens):
    """
    Remove common English stopwords that don't carry sentiment.
    Keeps negation words (not, never, no) as they affect meaning.
    
    Parameters:
        tokens (list): List of word tokens
    
    Returns:
        list: Filtered tokens without stopwords
    
    Example:
        >>> remove_stopwords(['this', 'product', 'is', 'not', 'good'])
        ['product', 'not', 'good']
    """
    return [word for word in tokens if word not in stop_words and len(word) > 1]


# ============================================================
# STEP 4: Stemming
# ============================================================

def stem_tokens(tokens):
    """
    Reduce words to their root form using Porter Stemmer.
    
    Stemming is fast but aggressive — it may produce non-real words.
    
    Parameters:
        tokens (list): List of word tokens
    
    Returns:
        list: Stemmed tokens
    
    Example:
        >>> stem_tokens(['running', 'happily', 'amazing', 'better'])
        ['run', 'happili', 'amaz', 'better']
    """
    return [stemmer.stem(word) for word in tokens]


# ============================================================
# STEP 5: Lemmatization
# ============================================================

def lemmatize_tokens(tokens):
    """
    Reduce words to their dictionary (base) form using WordNet Lemmatizer.
    
    Lemmatization is slower but produces real words.
    
    Parameters:
        tokens (list): List of word tokens
    
    Returns:
        list: Lemmatized tokens
    
    Example:
        >>> lemmatize_tokens(['running', 'happily', 'amazing', 'better'])
        ['running', 'happily', 'amazing', 'better']
    """
    return [lemmatizer.lemmatize(word) for word in tokens]


# ============================================================
# FULL PIPELINE
# ============================================================

def preprocess_text(text, use_stemming=True):
    """
    Complete preprocessing pipeline: clean → tokenize → remove stopwords → stem/lemmatize.
    
    Parameters:
        text (str): Raw text input
        use_stemming (bool): If True, use stemming. If False, use lemmatization.
    
    Returns:
        str: Fully preprocessed text (joined back as string)
    
    Example:
        >>> preprocess_text("This product is AMAZING!!! I absolutely LOVE it! 😍")
        'product amaz absolut love'
    """
    # Step 1: Clean
    text = clean_text(text)

    # Step 2: Tokenize
    tokens = tokenize(text)

    # Step 3: Remove stopwords
    tokens = remove_stopwords(tokens)

    # Step 4: Stem or Lemmatize
    if use_stemming:
        tokens = stem_tokens(tokens)
    else:
        tokens = lemmatize_tokens(tokens)

    # Join tokens back into a string (needed for TF-IDF later)
    return " ".join(tokens)


# ============================================================
# DEMO — Run this file to see preprocessing in action
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  🔤 TEXT PREPROCESSING DEMO — Sentiment Analysis")
    print("=" * 65)

    # Sample reviews to demonstrate each step
    sample_reviews = [
        "This product is AMAZING!!! Best purchase I've ever made. 😍😍😍",
        "Terrible quality... Broke within 2 days. Total WASTE of money!!!",
        "It's okay, nothing special. Does the job. 3/5 stars.",
        "DO NOT buy this! <br>Worst product <b>ever</b>. Visit http://complaints.com",
        "I'm not happy with this purchase. The delivery was never on time.",
    ]

    for i, review in enumerate(sample_reviews, 1):
        print(f"\n{'─' * 65}")
        print(f"  📝 Review #{i}")
        print(f"{'─' * 65}")
        print(f"\n  📥 Original:")
        print(f"     \"{review}\"")

        # Step 1
        cleaned = clean_text(review)
        print(f"\n  🧹 Step 1 — Cleaned:")
        print(f"     \"{cleaned}\"")

        # Step 2
        tokens = tokenize(cleaned)
        print(f"\n  ✂️  Step 2 — Tokenized:")
        print(f"     {tokens}")

        # Step 3
        filtered = remove_stopwords(tokens)
        print(f"\n  🚫 Step 3 — Stopwords Removed:")
        print(f"     {filtered}")

        # Step 4a
        stemmed = stem_tokens(filtered)
        print(f"\n  🌱 Step 4a — Stemmed:")
        print(f"     {stemmed}")

        # Step 4b
        lemmatized = lemmatize_tokens(filtered)
        print(f"\n  📖 Step 4b — Lemmatized:")
        print(f"     {lemmatized}")

        # Full pipeline
        final = preprocess_text(review)
        print(f"\n  ✅ Final Output (full pipeline):")
        print(f"     \"{final}\"")

    print(f"\n{'=' * 65}")
    print(f"  ✅ Preprocessing module ready!")
    print(f"  📦 Import with: from src.preprocess import preprocess_text")
    print(f"{'=' * 65}\n")
