To run the application, open the terminal and run the command : streamlit run app.py

I. Fonctionnalités principales

1. Sélection et navigation dans le corpus
- L'utilisateur peut choisir un corpus de documents en fonction de la langue (« Corpus Français » ou « Corpus Anglais »).
- Il peut soit choisir de faire une recherche par documents (dans un seul corpus) ou bien une recherche par phrases (dans un seul document) 
- Navigation par dossiers et fichiers via une barre latérale.

2. Prétraitement des documents

- Suppression des mots vides (stopwords) : les mots vides peuvent être supprimés selon la langue (« French », « English » ou aucun mot vide).
- Réduction lexicale : choix entre lemmatisation et diverses techniques de stemming (« Lovins », « Lancaster », « Porter », « Snowball »).

3. Méthodes de descripteurs

L'application propose différents descripteurs pour modéliser les documents :
BoW (Bag of Words)
TF-IDF
Word2Vec
FastText
Doc2Vec
Chaque descripteur peut être configuré pour être binaire ou basé sur les occurrences, avec une normalisation (« L1 », « L2 » ou aucune).
Remarque : seuls BoW et TF-IDF sont concernés par le type de comptage et la normalisation.

4. Visualisation des matrices

- Affichage des matrices de descripteurs et de similarité.
- Exploration visuelle :
a. Consultation des contenus de tokens ou documents avant et après prétraitement.
b. Visualisation des vectorisations des tokens à l'aide de réduction dimensionnelle (PCA).

5. Analyse des similarités

Métriques de distance : choix parmi plusieurs mesures de distance, telles que « Cosine », « Euclidean », « Manhattan », « Jaccard », « Kullback-Leibler », etc.
Recherche des k documents les plus similaires à un document ou une phrase donnée.

6. Génération de Word Clouds

Les mots les plus fréquents ou pertinents (à partir de TF-IDF) sont affichés dans un format visuel attractif

7. Chatbot

Interaction avec un chatbot simple qui répond aux questions en identifiant les phrases ou documents les plus proches du contenu fourni.

II. Fonctionnement et méthodes utilisées

Prétraitement des textes

- Tokenisation : Les phrases sont segmentées en phrases et en tokens (mots individuels) avec NLTK.
- Suppression des mots vides : Basée sur des listes de mots courants en français ou anglais avec NLTK.
- Réduction lexicale : Les mots sont réduits à leur racine ou forme canonique avec NLTK.

Application des descripteurs

- BoW et TF-IDF : Transformation des documents en matrices creuses où chaque ligne correspond à un document et chaque colonne à un mot unique.
- Word2Vec, FastText, et Doc2Vec : Génération d’embeddings (représentations vectorielles) pour capturer les relations sémantiques entre mots ou documents.
- Calcul des similarités
1. Pairwise Distances : Utilisation de la fonction « pairwise_distances » de scikit-learn pour comparer les vecteurs des documents.
2. Intégration de la distance de Kullback-Leibler.

Visualisation des embeddings avec PCA

- Réduction des embeddings à deux dimensions pour une représentation graphique.

Recherche des documents les plus proches

- Identification des k documents les plus similaires en fonction d’une requête utilisateur, avec ajustement du nombre k via un curseur.

Chatbot

- La recherche se fait dans le même corpus que le document sélectionné
- Identification du document le plus similaire en fonction d’une requête utilisateur
- Ré-application de l’identification au cas où on est dans l’option “Par documents” pour retrouver la phrase la plus similaire
- Simulation de compréhension des requêtes en utilisant un système de remplacement de mots clés (“Pourquoi” devient “Parce que” par ex.)

Génération de nuages de mots

- Création de nuages de mots à partir des textes du corpus ou d’un document.
- Paramètres configurables : nombre maximal de mots, couleur de fond, schéma de couleurs, méthodes avec les différents types de score TF-IDF
- Un nuage de mots en forme de carte du pays sélectionné
