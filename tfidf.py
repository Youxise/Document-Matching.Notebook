import math

# Fonction pour calculer TF-IDF_New pour un document
def tfidf_new(liste_mots_differents_corpus, document, df):
    """
    Cette fonction retourne un vecteur caractéristique du document de taille nb_mots_differents_corpus,
    où chaque bin contient le score TF-IDF_new.
    - liste_mots_differents_corpus : liste des mots uniques du corpus
    - document : liste de mots du document à analyser
    - df : dictionnaire donnant le nombre de documents contenant chaque mot (Document Frequency)
    """
    # Calculer la fréquence des mots dans le document (TF normalisé)
    tf_doc = {}
    for mot in document:
        tf_doc[mot] = tf_doc.get(mot, 0) + 1
    total_words = len(document)
    tf_doc = {mot: count / total_words for mot, count in tf_doc.items()}

    # Initialiser le vecteur TF-IDF_new avec des zéros pour chaque mot unique du corpus
    tfidf_new_vector = {mot: 0 for mot in liste_mots_differents_corpus}

    # Calculer le TF-IDF pour chaque mot du document
    for mot in document:
        if mot in liste_mots_differents_corpus:
            # Calcul de l'IDF avec une gestion des valeurs inconnues dans df
            idf = math.log10(len(df) / (1 + df.get(mot, 0)))
            tfidf_new_vector[mot] = tf_doc.get(mot, 0) * idf

    return tfidf_new_vector

# Calculate Document Frequency (DF) without using libraries
def calculate_document_frequency(tf_occ):

    # Initialize DF list with zeroes, one for each term
    df = [0] * len(tf_occ[0])
    
    # Iterate over each row in tf_occ matrix
    for row in tf_occ:
        for i in range(len(row)):
            if row[i] > 0:
                df[i] += 1  # Increment DF count for the term if it appears in the document
    
    return df

# Updated calculate_tfidf function to handle division by zero in IDF calculation
def calculate_tfidf(freq_matrix, num_documents):
    
    # Calculate Document Frequency (DF) using the custom function
    df = calculate_document_frequency(freq_matrix)

    # Calculate Inverse Document Frequency (IDF) with handling for zero values in DF
    idf = [
        math.log10(num_documents / (df[i] if df[i] > 0 else 1))
        for i in range(len(df))
    ]

    # Calculate the TF-IDF matrices
    similarity_matrix = [[freq_matrix[row][i] * idf[i] for i in range(len(idf))] for row in range(len(freq_matrix))]

    return similarity_matrix
