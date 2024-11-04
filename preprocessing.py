import re

# Split text into sentences
def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

# Tokenize sentences and gets all the vocabulary
def preprocess_sentences(sentences):
    unique_tokens = set()
    sent_tokens = []

    for sentence in sentences:
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        sent_tokens.append(tokens)
        unique_tokens.update(tokens)
    return sent_tokens, sorted(unique_tokens)
