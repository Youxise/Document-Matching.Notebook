import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
from string import punctuation

def corpus_preprocessing(sentences, reducer, stopwords_list):
    """
    Preprocess the corpus by tokenizing, removing stopwords, punctuation, and reducing words.

    Args:
        sentences (list of str): Input text sentences.
        reducer (callable): A function to reduce or normalize words (e.g., stemming or lemmatization).
        stopwords_list (list of str): List of stopwords to exclude from processing.

    Returns:
        tuple: Preprocessed sentences, tokenized sentences, and a set of unique tokens.
    """

    new_sentences = []
    # Preprocessing sentences
    for i in range(len(sentences)):
        words = word_tokenize(sentences[i])
        reduced_words = [reducer(word).lower() for word in words if word.lower() not in stopwords_list and word not in punctuation and len(word) > 1]
        if reduced_words:
            new_sentences.append(' '.join(reduced_words))

    # Process tokens and unique tokens
    tokenized_sentences = [word_tokenize(sentence) for sentence in new_sentences]
    unique_tokens = sorted(set(word for tokens in tokenized_sentences for word in tokens))

    return new_sentences, tokenized_sentences, unique_tokens

def tf_idf(sentences, descriptor_type, normalization_type):
    """
    Compute the TF-IDF matrix for a given set of sentences.

    Args:
        sentences (list of str): Input sentences.
        descriptor_type (str): Type of descriptor (e.g., "Binary" or "Occurrence").
        normalization_type (str): Normalization type for the TF-IDF matrix ("L1", "L2", or "None").

    Returns:
        ndarray: TF-IDF matrix.
    """

    # Set up the vectorizer with the desired normalization
    norm = normalization_type.lower() if normalization_type != "None" else None
    
    if descriptor_type == "Binary":
        # Binary TF-IDF: Transform TF values to binary (0 or 1) before calculating TF-IDF
        class BinaryTfidfVectorizer(TfidfVectorizer):
            def fit_transform(self, sentences, y=None):
                # Call the parent class's fit_transform
                X = super().fit_transform(sentences, y)
                # Convert the term-frequency matrix to binary
                X.data[:] = (X.data > 0).astype(int)
                return X
        
        vectorizer = BinaryTfidfVectorizer(norm=norm, use_idf=True)
    
    elif descriptor_type == "Occurrence":
        # Occurrence TF-IDF
        vectorizer = TfidfVectorizer(norm=norm, use_idf=True)
    
    # Generate the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(sentences).toarray()

    return tfidf_matrix


# Frequency matrix
def create_freq_matrix(tokenized_sentences, unique_tokens, descriptor_type, normalization_type):
    """
    Create a frequency matrix for the corpus.

    Args:
        tokenized_sentences (list of list of str): Tokenized sentences.
        unique_tokens (list of str): List of unique tokens.
        descriptor_type (str): Type of descriptor (e.g., "Binary").
        normalization_type (str): Type of normalization ("L1", "L2").

    Returns:
        ndarray: Normalized frequency matrix.
    """

    # Initialize frequency matrix
    n_sentences = len(tokenized_sentences)
    n_tokens = len(unique_tokens)
    freq_matrix = np.zeros((n_sentences, n_tokens))

    # Map unique tokens to their index for fast lookup
    token_to_index = {token: i for i, token in enumerate(unique_tokens)}

    # Fill the frequency matrix
    for i, sentence in enumerate(tokenized_sentences):
        for token in sentence:
            if token in token_to_index:
                freq_matrix[i, token_to_index[token]] += 1

    # Apply descriptor type
    if descriptor_type == "Binary":
        freq_matrix = np.where(freq_matrix > 0, 1, 0)

    # Normalize the matrix
    if normalization_type == "L1":
        row_sums = freq_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        freq_matrix = freq_matrix / row_sums
    elif normalization_type == "L2":
        row_norms = np.linalg.norm(freq_matrix, axis=1, keepdims=True) # sqrt of sum of all elements squared
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        freq_matrix = freq_matrix / row_norms

    return freq_matrix


def k_nearest_documents(sentences, tokenized_sentences, unique_tokens, descriptor, descriptor_type, normalization_type, k, corpus_matrix, distance_metric="cosine"):
    """
    Find the k most similar documents to a query using the specified descriptor and distance metric.

    Args:
        sentences (list of str): Query sentences.
        tokenized_sentences (list of list of str): Tokenized sentences from the corpus.
        unique_tokens (list of str): Unique tokens in the corpus.
        descriptor (str): Type of descriptor to use ("TF-IDF", "Word2Vec", "FastText", "Doc2Vec", etc.).
        descriptor_type (str): Specific descriptor type ("Binary", "Occurrence").
        normalization_type (str): Type of normalization ("L1", "L2", or "None").
        k (int): Number of similar documents to retrieve.
        corpus_matrix (ndarray): Matrix representation of the corpus.
        distance_metric (str): Metric to compute distances (default: "cosine").

    Returns:
        tuple: A list of tuples (document_index, similarity_score) for the top-k similar documents and the query vector.
    """
    # Create the query vector based on the descriptor
    if descriptor == "TF-IDF":
        query_vector = tf_idf(sentences, descriptor_type, normalization_type)
    elif descriptor == "Word2Vec":
        model = Word2Vec(tokenized_sentences, vector_size=10, window=5, min_count=1, workers=4)
        query_vector = np.array([
            np.mean([model.wv[word] for word in simple_preprocess(sentence) if word in model.wv] or [np.zeros(10)], axis=0)
            for sentence in sentences
        ]).reshape(1, -1)
    elif descriptor == "FastText":
        model = FastText(tokenized_sentences, vector_size=10, window=5, min_count=1, workers=4)
        query_vector = np.array([
            np.mean([model.wv[word] for word in simple_preprocess(sentence) if word in model.wv] or [np.zeros(10)], axis=0)
            for sentence in sentences
        ]).reshape(1, -1)
    elif descriptor == "Doc2Vec":
        tagged_data = [TaggedDocument(words=reduced_words, tags=[f"Doc {i+1}"]) for i, reduced_words in enumerate(sentences)]
        model = Doc2Vec(tagged_data, vector_size=10, window=5, min_count=1, workers=4)
        query_vector = np.array([model.dv[f'Doc {i+1}'] for i in range(len(sentences))]).reshape(1, -1)
    else:  # Frequency matrix as fallback
        freq_matrix = create_freq_matrix(tokenized_sentences, unique_tokens, descriptor_type, normalization_type)
        query_vector = freq_matrix

    # Adjust query vector dimensions to match the corpus matrix
    # Check if the dimension of the query vector columns correspond to that of the corpus matrix.
    # If it's not, zeros will be added to adjust the dimension using np.pad
    if query_vector.shape[1] != corpus_matrix.shape[1]:
        query_vector = np.pad(query_vector, ((0, 0), (0, corpus_matrix.shape[1] - query_vector.shape[1])), mode="constant")

    # Compute pairwise distances between query vector and corpus matrix
    distances = pairwise_distances(query_vector, corpus_matrix, metric=distance_metric).flatten()

    # Identify the k smallest distances (most similar documents)
    top_k_indices = np.argsort(distances)[:k]

    # Pair document indices with similarity scores
    top_k_results = [(idx + 1, distances[idx]) for idx in top_k_indices]

    return top_k_results, query_vector


def descriptor_application(sentences_pp, tokenized_sentences_pp, unique_tokens, descriptor, descriptor_type, normalization_type):
    """
    Generate the appropriate descriptor model and matrix based on the chosen descriptor type.

    Args:
        sentences_pp (list of str): Preprocessed sentences.
        tokenized_sentences_pp (list of list of str): Tokenized preprocessed sentences.
        unique_tokens (list of str): Unique tokens in the corpus.
        descriptor (str): Descriptor type ("TF-IDF", "Word2Vec", "FastText", "Doc2Vec").
        descriptor_type (str): Specific descriptor type ("Binary", "Occurrence").
        normalization_type (str): Normalization type ("L1", "L2", "None").

    Returns:
        tuple: The descriptor model and the corresponding feature matrix.
    """
    if descriptor == "TF-IDF":
        # Use TF-IDF descriptor
        model = tf_idf
        matrix = tf_idf(sentences_pp, descriptor_type, normalization_type)
    elif descriptor == "Word2Vec":
        # Use Word2Vec model to compute sentence vectors
        model = Word2Vec(tokenized_sentences_pp, vector_size=10, window=5, min_count=1, workers=4)
        matrix = np.array([
            np.mean([model.wv[word] for word in simple_preprocess(sentence) if word in model.wv] or [np.zeros(10)], axis=0)
            for sentence in sentences_pp
        ])
    elif descriptor == "FastText":
        # Use FastText model to compute sentence vectors
        model = FastText(tokenized_sentences_pp, vector_size=10, window=5, min_count=1, workers=4)
        matrix = np.array([
            np.mean([model.wv[word] for word in simple_preprocess(sentence) if word in model.wv] or [np.zeros(10)], axis=0)
            for sentence in sentences_pp
        ])
    elif descriptor == "Doc2Vec":
        # Use Doc2Vec model to compute document embeddings
        tagged_data = [TaggedDocument(words=sent, tags=[f"Doc {i+1}"]) for i, sent in enumerate(tokenized_sentences_pp)]
        model = Doc2Vec(tagged_data, vector_size=10, window=5, min_count=1, workers=4)
        matrix = np.array([model.dv[f"Doc {i+1}"] for i in range(len(sentences_pp))])
    else:
        # Default to a frequency matrix
        freq_matrix = create_freq_matrix(tokenized_sentences_pp, unique_tokens, descriptor_type, normalization_type)
        model = create_freq_matrix
        matrix = freq_matrix

    return model, matrix

def kullback_leibler_divergence(p, q):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions.

    Args:
        p (ndarray): The first probability distribution.
        q (ndarray): The second probability distribution.

    Returns:
        float: KL divergence value.
    """
    # Ensure non-negative values and avoid division by zero
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    
    return np.sum(p * np.log(p / q))

def compute_kl_similarity_matrix(matrix):
    """
    Compute the KL divergence-based similarity matrix for the given matrix.

    Args:
        matrix (ndarray): Input matrix with rows as probability distributions.

    Returns:
        ndarray: Pairwise KL divergence matrix.
    """
    n_docs = matrix.shape[0]
    similarity_matrix = np.zeros((n_docs, n_docs))

    # Compute pairwise KL divergence
    for i in range(n_docs):
        for j in range(n_docs):
            similarity_matrix[i, j] = kullback_leibler_divergence(matrix[i], matrix[j])

    return similarity_matrix