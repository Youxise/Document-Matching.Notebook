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

chiraq_text = """La confiance que vous venez de me témoigner, je veux y répondre en m'engageant dans l'action avec détermination.
Mes chers compatriotes de métropole, d'outre-mer et de l'étranger,
Nous venons de vivre un temps de grave inquiétude pour la Nation.
Mais ce soir, dans un grand élan la France a réaffirmé son attachement aux valeurs de la République.
Je salue la France, fidèle à elle-même, fidèle à ses grands idéaux, fidèle à sa vocation universelle et humaniste.
Je salue la France qui, comme toujours dans les moments difficiles, sait se retrouver sur l'essentiel. Je salue les Françaises et les Français épris de solidarité et de liberté, soucieux de s'ouvrir à l'Europe et au monde, tournés vers l'avenir.
J'ai entendu et compris votre appel pour que la République vive, pour que la Nation se rassemble, pour que la politique change. Tout dans l'action qui sera conduite, devra répondre à cet appel et s'inspirer d'une exigence de service et d'écoute pour chaque Française et chaque Français.
Ce soir, je veux vous dire aussi mon émotion et le sentiment que j'ai de la responsabilité qui m'incombe.
Votre choix d'aujourd'hui est un choix fondateur, un choix qui renouvelle notre pacte républicain. Ce choix m'oblige comme il oblige chaque responsable de notre pays. Chacun mesure bien, à l'aune de notre histoire, la force de ce moment exceptionnel.
Votre décision, vous l'avez prise en conscience, en dépassant les clivages traditionnels, et, pour certains d'entre vous, en allant au-delà même de vos préférences personnelles ou politiques.
La confiance que vous venez de me témoigner, je veux y répondre en m'engageant dans l'action avec détermination.
Président de tous les Français, je veux y répondre dans un esprit de rassemblement. Je veux mettre la République au service de tous. Je veux que les valeurs de liberté, d'égalité et de fraternité reprennent toute leur place dans la vie de chacune et de chacun d'entre nous.
La liberté, c'est la sécurité, la lutte contre la violence, le refus de l'impunité. Faire reculer l'insécurité est la première priorité de l'Etat pour les temps à venir.
La liberté, c'est aussi la reconnaissance du travail et du mérite, la réduction des charges et des impôts.
L'égalité, c'est le refus de toute discrimination, ce sont les mêmes droits et les mêmes devoirs pour tous.
La fraternité, c'est sauvegarder les retraites. C'est aider les familles à jouer pleinement leur rôle. C'est faire en sorte que personne n'éprouve plus le sentiment d'être laissé pour compte.
La France, forte de sa cohésion sociale et de son dynamisme économique, portera en Europe et dans le monde l'ambition de la paix, des libertés et de la solidarité.
Dans les prochains jours, je mettrai en place un gouvernement de mission, un gouvernement qui aura pour seule tâche de répondre à vos préoccupations et d'apporter des solutions à des problèmes trop longtemps négligés. Son premier devoir sera de rétablir l'autorité de l'Etat pour répondre à l'exigence de sécurité, et de mettre la France sur un nouveau chemin de croissance et d'emploi.
C'est par une action forte et déterminée, c'est par la solidarité de la Nation, c'est par l'efficacité des résultats obtenus, que nous pourrons lutter contre l'intolérance, faire reculer l'extrémisme, garantir la vitalité de notre démocratie. Cette exigence s'impose à chacun d'entre nous. Elle impliquera, au cours des prochaines années, vigilance et mobilisation de la part de tous.
Mes chers compatriotes,
Le mandat que vous m'avez confié, je l'exercerai dans un esprit d'ouverture et de concorde, avec pour exigence l'unité de la République, la cohésion de la Nation et le respect de l'autorité de l'Etat.
Les jours que nous venons de vivre ont ranimé la vigueur nationale, la vigueur de l'idéal démocratique français. Ils ont exprimé une autre idée de la politique, une autre idée de la citoyenneté.
Chacune et chacun d'entre vous, conscient de ses responsabilités, par un choix de liberté, a contribué, ce soir, à forger le destin de la France.
Il y a là un espoir qui ne demande qu'à grandir, un espoir que je veux servir.
Vive la République !
Vive la France !"""

from preprocessing import split_into_sentences

sentences = split_into_sentences(chiraq_text)
#matrices1, matrices2, matrices3, tokens = create_matrices(sentences)
#print(matrices1)


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
