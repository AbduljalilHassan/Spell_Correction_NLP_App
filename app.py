import string
import numpy as np
from collections import Counter
import nltk 
import streamlit as st
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# ----------------------
# 1. Dataset Loading and Preprocessing
# ----------------------
import chardet

with open('Drugs_Dictionary.txt', 'rb') as f:  # Open in binary mode
    rawdata = f.read()
    result = chardet.detect(rawdata)
    encoding = result['encoding'] 

def load_corpus(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    text = text.lower()
    text = nltk.regexp_tokenize(text, r'\w+')

    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text]
    return lemmatized_text

# Build N-gram models
def build_ngram_models(corpus):
    unigrams = nltk.FreqDist(corpus)
    bigrams = nltk.bigrams(corpus)
    bigram_model = nltk.ConditionalFreqDist(bigrams) 
    return unigrams, bigram_model

# ----------------------
# 2. Error Generation 
# ----------------------
def generate_edits(word, max_edit_distance):
    if max_edit_distance == 1:
        return set(swap(word) + delete(word) + insert(word) + replace(word))
    else:
        edits = set()
        edits.update(swap(word))
        edits.update(delete(word))
        edits.update(insert(word))
        edits.update(replace(word))
        for edit in list(edits):
            if max_edit_distance > 1:
                edits.update(generate_edits(edit, max_edit_distance - 1))
        return edits

def swap(word):
    swaps = []
    for i in range(len(word) - 1):
        swapped_word = word[:i] + word[i+1] + word[i] + word[i+2:]
        swaps.append(swapped_word)
    return swaps

def delete(word):
    return [word[:i] + word[i+1:] for i in range(len(word))]

def insert(word):
    alphabet = string.ascii_lowercase
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    return [L + char + R for L, R in splits for char in alphabet]

def replace(word):
    alphabet = string.ascii_lowercase
    splits = [(word[:i], word[i+1:]) for i in range(len(word))]
    return [L + char + R for L, R in splits for char in alphabet]

# ----------------------
# 3. Probabilistic Suggestion + Helper Functions
# ----------------------
def edit_distance(word1, word2):
    dp = np.zeros((len(word1)+1, len(word2)+1))

    for i in range(len(word1)+1):
        dp[i][0] = i  
    for j in range(len(word2)+1):
        dp[0][j] = j  

    for i in range(1, len(word1)+1):
        for j in range(1, len(word2)+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1] 
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    
                    dp[i][j-1],    
                    dp[i-1][j-1]   
                    )

    return int(dp[len(word1)][len(word2)]) 

def bigram_probability(word1, word2, vocabulary, build_ngram_models, unigram_counts):
    if word1 in vocabulary:  
        return build_ngram_models[1][word1][word2] / unigram_counts[word1]
    else:
        return 0.0 

def suggest_corrections(word, vocabulary, build_ngram_models, unigram_counts, max_edit_distance=2):
    if word in vocabulary:
        return [(word, 0)]  

    candidates = generate_edits(word, max_edit_distance)  

    scored_candidates = []
    for candidate in candidates:
        if candidate in vocabulary:
            edit_dist = edit_distance(word, candidate)
            score = unigram_counts[candidate] * bigram_probability(word, candidate, vocabulary, build_ngram_models, unigram_counts) / edit_dist 
            scored_candidates.append((candidate, edit_dist, score))

    # Filter out None values from scored_candidates
    scored_candidates = [candidate for candidate in scored_candidates if candidate[2] is not None]

    return sorted(scored_candidates, key=lambda x: x[2], reverse=True)


# Load your dataset (replace with your actual path)
corpus = load_corpus('Drugs_Dictionary.txt') 
vocabulary = set(corpus)
unigram_counts = Counter(corpus)
build_ngram_models = build_ngram_models(corpus)

# Streamlit App Interface 
def main():
    global max_edit_distance  # Access global variables

    # Set page configuration
    st.set_page_config(page_title="SpellCheck System", layout="wide", page_icon=":pencil2:")

    # Add custom CSS for styling
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(to right, #2980b9, #6ab0de); /* Updated background color */
            color: white; /* Updated font color */
        }
        .sidebar .sidebar-content {
            background: #34495e; /* Updated sidebar background color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main content layout
    st.title("SpellCheck System")

    # Initialize evaluation metrics
    precision = 0
    recall = 0
    f1_score = 0
    accuracy = 0

    # Split the page into two columns
    col1, col2 = st.columns([2, 1])

    # Left column for user input and spell check results
    with col1:
        st.subheader("Spell Check")
        user_input = st.text_area("Enter your text:", height=200)
        if st.button("Check Spelling"):
            if not user_input:
                st.warning("Please enter some text to check.")
            else:
                st.subheader("Spell Check Results")
                input_words = user_input.split()
                correct_count = 0  # Initialize correct count to zero
                misspelled_count = 0  # Initialize misspelled count to zero
                input_words_with_correction = []
                for i, word in enumerate(input_words):
                    if word.lower() not in vocabulary:
                        st.error(f"*{word}* (potentially misspelled)")
                        suggestions = suggest_corrections(word, vocabulary, build_ngram_models, unigram_counts)
                        expander = st.expander(f"Suggestions for '{word}'")
                        for suggestion, edit_dist, score in suggestions[:3]:
                            expander.write(f"- {suggestion} (Edit Distance: {edit_dist})")
                        corrected_word = st.text_input(f"Enter correction for '{word}':", value=word, key=f"correction_{i}")  # Unique key
                        input_words_with_correction.append(corrected_word)
                        misspelled_count += 1  # Increment misspelled count for each potentially misspelled word
                    else:
                        correct_count += 1  # Increment correct count for each correctly spelled word
                        input_words_with_correction.append(word)
                        st.success(f"{word} (correctly spelled)")

                # Calculate total word count
                total_count = len(input_words)

                # Calculate evaluation metrics
                accuracy = round(correct_count / total_count, 2) if total_count != 0 else 0
                precision = round(correct_count / (correct_count + misspelled_count), 2) if (correct_count + misspelled_count) != 0 else 0
                recall = 1.00  # Recall would be 1 since all correctly spelled words are captured
                f1_score = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) != 0 else 0

                # Display evaluation metrics
                st.sidebar.header("Evaluation Results")
                st.sidebar.write(f"Precision: {precision}")
                st.sidebar.write(f"Recall: {recall}") 
                st.sidebar.write(f"F1 Score: {f1_score}")
                st.sidebar.write(f"Accuracy: {accuracy}")

    # Right column for word list and settings sidebar
    with col2:
        st.header("Word List")
        search_term = st.text_input("Search Word List")
        word_list = sorted(list(vocabulary))
        filtered_words = [word for word in word_list if search_term.lower() in word.lower()]
        st.write(filtered_words)

        # Sidebar for settings
        st.sidebar.header("Settings")
        max_edit_distance = st.sidebar.slider("Max Edit Distance", min_value=1, max_value=2, value=2, step=1)
        st.sidebar.write("Adjust the max edit distance for spell correction.")

if __name__ == "__main__":
    main()
