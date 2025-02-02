import os
import nltk
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from PIL import Image 

# Import custom modules for preprocessing, etc.
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from stemming.lovins import stem
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Import optimized functions
from matrix import corpus_preprocessing, descriptor_application, k_nearest_documents, compute_kl_similarity_matrix
from folder_manage import list_files, list_folders


# Set up the Streamlit page layout and configuration
st.session_state["base_path"] = os.getcwd()  # Start at the current working directory
base_path = st.session_state.get("base_path", ".")

st.set_page_config(
    page_title="Document Matching",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# Display the main title of the application
st.title("Document Similarity Analysis")

language = st.sidebar.radio("Select language:", ["French", "English"], index=None)
search_type = st.sidebar.radio("Select searching type:", ["Per document", "Per sentence"], index=None)

if language == "French":
    folder_path = "Corpus_Francais"
    mask_map = np.array(Image.open('france-map.jpg'))
else:
    folder_path = "Corpus_Anglais"
    mask_map = np.array(Image.open('usa-map.jpg'))


corpus = list_folders(folder_path)

# Folder navigation
if corpus:
    base_path = os.path.join(base_path, folder_path)
    selected_folder = st.sidebar.selectbox("Select corpus:", corpus)
    base_path = os.path.join(base_path, selected_folder) # update the path

# Save the updated path in session state
st.session_state["base_path"] = base_path

documents = list_files(st.session_state["base_path"])

# File navigation
if documents:
    document = st.sidebar.selectbox(
        "Select document:",
        options=documents,
    )
    document = os.path.join(st.session_state["base_path"], document) # update the path
# If folder is empty
else:
    st.sidebar.write("No files found in the specified folder.")

# Descriptor
descriptor = st.sidebar.selectbox("Select descriptor to use:", ["BoW", "TF-IDF", "Word2Vec", "FastText", "Doc2Vec"])
descriptor_type = st.sidebar.selectbox("Select descriptor type:", ["Binary", "Occurrence"])
normalization_type = st.sidebar.selectbox("Select normalization method:", ["L1", "L2", "None"])
distance_type = st.sidebar.selectbox("Select distance metric:", ["Euclidean", "Manhattan", "Cosine", "Jaccard", "Hamming", "Bray-Curtis", "Kullback-Leibler"])

# Preprocessing options
stopwords_option = st.sidebar.selectbox("Select stopwords to remove:", ["French", "English", "None"])

reducing_type = st.sidebar.radio("Select reducing type:", ["Stemming", "Lemmatization", "None"], index=None)
stemming_option = st.sidebar.selectbox("Select stemming:", ["Lovins", "Lancaster", "Porter", "Snowball"])

input_text = None

# Load the corpus documents
if search_type == "Per document":
    sentences = []
    print(st.session_state["base_path"])
    for filename in os.listdir(st.session_state["base_path"]):
        file_path = os.path.join(st.session_state["base_path"], filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                sentences.append(file.read().lower())
else:
# Load the document and tokenize sentences
    with open(document, 'r', encoding='utf-8') as file:
        input_text = file.read().lower()
    
        # Split the input text into sentences
        sentences = sent_tokenize(input_text)

# Remove stopwords
stopwords_list = []
if stopwords_option == "French":
    stopwords_list = stopwords.words('french')
elif stopwords_option == "English":
    stopwords_list = stopwords.words('english')

# Reducing words
def same(text):
    return text

reducer = same
if reducing_type == "Stemming":
    stemming_functs = {
        "Lovins": stem,
        "Porter": PorterStemmer().stem,
        "Lancaster": LancasterStemmer().stem,
        "Snowball": SnowballStemmer("english").stem if language == "English" else SnowballStemmer("french").stem 
    }
    reducer = stemming_functs[stemming_option]
else:
    reducer = WordNetLemmatizer().lemmatize

sentences_pp, tokenized_sentences_pp, unique_tokens = corpus_preprocessing(sentences, reducer, stopwords_list)

model, matrix = descriptor_application(sentences_pp, tokenized_sentences_pp, unique_tokens, descriptor, descriptor_type, normalization_type)

if input_text:
        st.text_area("Corpus content:", input_text, height=300, disabled=True)

st.write(f"The text contains {len(sentences_pp)} documents and {len(unique_tokens)} unique tokens.")

# Display descriptor matrix
st.subheader(f"{descriptor} Matrix")
st.dataframe(pd.DataFrame(matrix, index=[f'Doc {i+1}' for i in range(len(sentences_pp))]))

visualize_tok_content = st.number_input("Enter token number:", min_value=1, max_value=len(unique_tokens), step=1) - 1
st.text_area("Token content:", unique_tokens[visualize_tok_content], height=68, disabled=True)

# Display content
visualize_doc_content = st.number_input("Enter document number: ", min_value=1, max_value=len(sentences), step=1) - 1
st.text_area("Document content before preprocessing:", sentences[visualize_doc_content], height=100, disabled=True)

# Display content
visualize_doc_content = st.number_input("Enter document number:  ", min_value=1, max_value=len(sentences_pp), step=1) - 1
st.text_area("Document content after preprocessing:", sentences_pp[visualize_doc_content], height=100, disabled=True)


# Display similarity matrix
st.subheader("Similarity Matrix")
if distance_type.lower() == "kullback-leibler":
    similarity_matrix = compute_kl_similarity_matrix(matrix)
else:
    similarity_matrix = pairwise_distances(matrix, metric= re.sub(r'[^\w\s]', '', distance_type.lower()))
st.dataframe(pd.DataFrame(similarity_matrix, index=[f'Doc {i+1}' for i in range(len(sentences_pp))], columns=[f'Doc {i+1}' for i in range(len(sentences))]))

if descriptor in ["Word2Vec", "FastText", "Doc2Vec"]:
        
        # Récupérez les embeddings pour le vocabulaire
        word_vectors = model.wv  # Récupérer les vecteurs pour Word2Vec ou FastText
        embeddings = []
        words = []

        for word in unique_tokens:
            if word in word_vectors:
                embeddings.append(word_vectors[word])
                words.append(word)

        # Réduction dimensionnelle à 2D avec PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Afficher le scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

        # Annoter les points avec les mots
        for i, word in enumerate(words):
            ax.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8)

        st.pyplot(fig)

# K-nearest documents
st.subheader("Find Most Similar Documents")
doc_index = st.number_input("Enter document number:", min_value=1, max_value=len(sentences_pp), step=1) - 1
doc_query = st.text_area("Document content:", sentences_pp[doc_index], height=300, disabled=False)
k = st.slider("Select number of similar documents:", 1, len(sentences_pp) - 1, 3)
if st.button("Find"):
    sentences_tab1, tokenized_sentences_tab1, unique_tokens_tab1 = corpus_preprocessing([doc_query], reducer, stopwords_list)
    k_closest, vector_query = k_nearest_documents(sentences_tab1, tokenized_sentences_tab1, unique_tokens_tab1, descriptor, descriptor_type, normalization_type, k, matrix, "cosine")
    k_closest_dataframe = pd.DataFrame(k_closest, columns=["Document Number", "Similarity Score"])
    k_closest_dataframe["Document content"] = k_closest_dataframe["Document Number"].apply(lambda x: sentences_pp[x-1])
    st.write(k_closest_dataframe)

# ------------------------------------------------------- Chatbot -------------------------------------------------- #


st.title("Chatbot")

query = st.text_area("Type a query:", height=100, disabled=False)
response_functs = {
        "hello,": "Hi there! ",
        "hi,": "Hello there! ",
        "bonjour,": "Bonjour! ",
        "bonsoir,": "Bonsoir! ",
        "salut,": "Salutations! ",
        "pourquoi": "Car ",
        "why": "Because ",
        "comment": "Après analyse, ",
        "how": "After analysis, ",
        "peux-tu": "Oui bien sûr. ",
        "can": "Sure! "
    }
def replace_words(text, responses):

    words = text.split()
    replaced_text = []

    for word in words:
        if word.lower() in responses:
            replaced_text.append(responses[word.lower()])

    return " ".join(replaced_text)

if st.button("Ask"):
    
    chatbot_response = replace_words(query, response_functs)
    
    sentences_tab2_pp, tokenized_sentences_tab2, unique_tokens_tab2 = corpus_preprocessing([query], reducer, stopwords_list)

    if descriptor != "FastText":
    # Filtering query words not in vocabulary
        for i in range(len(tokenized_sentences_tab2)):
            reduced_words = [word for word in tokenized_sentences_tab2[i] if word in unique_tokens]
            sentences_tab2_pp[i] = ' '.join(reduced_words)
        
        tokenized_sentences_tab2 = [word_tokenize(sentence) for sentence in sentences_tab2_pp]
        unique_tokens_tab2 = sorted(set(word for tokens in tokenized_sentences_tab2 for word in tokens))
    
    # Searching for the most similar document
    k_closest = k_nearest_documents(sentences_tab2_pp, tokenized_sentences_tab2, unique_tokens_tab2, descriptor, descriptor_type, normalization_type, 1, matrix, "cosine")

    # Searching for the best response based on the most significant query words
    top_doc_idx = k_closest[0][0][0] - 1
    top_doc = sentences[top_doc_idx]
    if search_type == "Per sentence":
        st.write(chatbot_response + top_doc)
    else:
        # Searching for the similar sentence
        similar_doc_sents = sent_tokenize(top_doc)
        similar_doc_sents_pp, similar_doc_sents_tokenized, similar_doc_unique_tokens = corpus_preprocessing(similar_doc_sents, reducer, stopwords_list)
        model_tab2, matrix_tab2 = descriptor_application(similar_doc_sents_pp, similar_doc_sents_tokenized, similar_doc_unique_tokens, descriptor, descriptor_type, normalization_type)

        similar_sent = k_nearest_documents(sentences_tab2_pp, tokenized_sentences_tab2, unique_tokens_tab2, descriptor, descriptor_type, normalization_type, 1, matrix_tab2, "cosine")
        top_sent_idx = similar_sent[0][0][0] - 1
        top_sent = similar_doc_sents[top_sent_idx]
        st.write(chatbot_response + top_sent)

# ------------------------------------------------------- Word Cloud -------------------------------------------------- #

st.title("Word Cloud")  
# Display content
max_words = st.number_input("Max words:  ", min_value=1, max_value=len(unique_tokens), step=1)

background_color = st.selectbox("Select background color:", ["Black", "White"])

color = st.selectbox("Select color map:", ['viridis', 'inferno', 'plasma', 'magma', 'Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd', 'afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter', 'BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic', 'Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3', 'Vega10',
                             'Vega20', 'Vega20b', 'Vega20c', 'gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd'])

cloud_method = st.selectbox("Select method: ", ["TF-IDF score sum", "TF-IDF score mean", "TF-IDF score max", "None"])

from sklearn.feature_extraction.text import TfidfVectorizer

if st.button("Generate"):
    if search_type == "Per document":
        full_text = "".join([item for sublist in sentences for item in sublist])
    else:
        full_text = input_text
    fig = plt.figure()

    word_cloud =  WordCloud(stopwords=stopwords_list, background_color=background_color.lower(),mask=mask_map, colormap=color, contour_width=2, width=800, height=400, max_words=max_words, contour_color='firebrick')
    
    if cloud_method == "None":
        word_cloud.generate(full_text)
    else:
        vectorizer_cloud = TfidfVectorizer(stop_words=stopwords_list)
        matrix_cloud = vectorizer_cloud.fit_transform(sent_tokenize(full_text))
        tokens_cloud = vectorizer_cloud.get_feature_names_out()
        score_cloud = matrix_cloud.toarray()

        cloud_method_functs = {
    "TF-IDF score sum": word_cloud.generate_from_frequencies(dict(zip(tokens_cloud, score_cloud.sum(axis=0)))),
    "TF-IDF score mean": word_cloud.generate_from_frequencies(dict(zip(tokens_cloud, score_cloud.mean(axis=0)))),
    "TF-IDF score max": word_cloud.generate_from_frequencies(dict(zip(tokens_cloud, score_cloud.max(axis=0))))
}
        
        cloud_method_functs[cloud_method]
    plt.axis("off")
    plt.imshow(word_cloud, interpolation="bilinear")
    st.pyplot(fig)