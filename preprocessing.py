import re

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())


def preprocess_sentences(sentences):
    unique_tokens = set()
    sent_tokens = []

    for sentence in sentences:
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        sent_tokens.append(tokens)
        unique_tokens.update(tokens)
    return sent_tokens, sorted(unique_tokens)



# Frequency matrix
def create_freq_matrix(word_of_sentences, unique_tokens, matrix_type, normalization_type):

    freq_matrix = []
    if matrix_type == 'Binary':
        for sent_tokens in word_of_sentences:            
            matrix_row = [1 if token in sent_tokens else 0 for token in unique_tokens]
            freq_matrix.append(matrix_row)
    else: # Occurence
        for sent_tokens in word_of_sentences:  
            matrix_row = [sent_tokens.count(token) for token in unique_tokens]
            freq_matrix.append(matrix_row)

    # Appliquer la normalisation
    if normalization_type == "Probability":
        for row in freq_matrix:
            row_sum = sum(row)
            for i in range(len(row)):
                row[i] = row[i] / row_sum if row_sum != 0 else 0
    elif normalization_type == "L2":
        for row in freq_matrix:
            row_sum = sum(x**2 for x in row) ** 0.5
            for i in range(len(row)):
                row[i] = row[i] / row_sum if row_sum != 0 else 0

    return freq_matrix

# ----------------------------------------------------------------------------#

# Similarity matrix
def create_similarity_matrix(sentences, bow_matrix, distance):
    
    distance_matrix = [[0] * len(sentences) for _ in range(len(sentences))]

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                distance_matrix[i][j] = distance(bow_matrix[i], bow_matrix[j])

    return distance_matrix