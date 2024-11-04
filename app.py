import streamlit as st
import pandas as pd
from preprocessing import split_into_sentences, preprocess_sentences
from matrix import k_nearest_documents, create_freq_matrix, create_similarity_matrix
from tfidf import calculate_tfidf, tfidf_new
from distances import distance_euclidean, distance_manhattan, distance_cosinus, distance_jaccard, distance_hamming, distance_bray_curtis, distance_kullback_leibler

# Tab
st.set_page_config(
        page_title="Document Matching",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

# Title of the application
st.title("Document Similarity Analysis")

# Selection
descriptor = st.sidebar.selectbox("Select descriptor to use:", 
                                  ["BoW", "TF-IDF"])
descriptor_type = st.sidebar.selectbox("Select descriptor type:", 
                                       ["Binary", "Occurrence"])
normalization_type = st.sidebar.selectbox("Select normalization method:", 
                                          ["None", "Probability", "L2"])
corpus_type = st.sidebar.selectbox("Select corpus:", 
                                   ["Chirac", "Obama", "CatsDogs", "Custom", "File"])
distance_type = st.sidebar.selectbox("Select distance metric:",
                                      ["Euclidean", "Manhattan", "Cosine", "Jaccard", "Hamming", "Bray-Curtis", "Kullback-Leibler"])

if corpus_type == "Chirac" or corpus_type == "Obama" or corpus_type == "CatsDogs":
    with open(corpus_type + ".txt", 'r', encoding='utf-8') as file:
        input_text = file.read()

elif corpus_type == "Custom":
    input_text = st.text_area("Enter your text:")
else:
    uploaded_file = st.file_uploader("Upload your file:", type="txt")
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")
    else:
        input_text = """La confiance que vous venez de me témoigner, je veux y répondre en m'engageant dans l'action avec détermination.
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

# Process if text is provided
if input_text:
    with st.spinner("Processing..."):
        sentences = split_into_sentences(input_text)
        sents_tokens, unique_tokens = preprocess_sentences(sentences)

        st.write(f"The text contains {len(sentences)} sentences and {len(unique_tokens)} tokens.")
        
        if descriptor == "BoW":
            st.header("Bag of Words")
            matrix = create_freq_matrix(sents_tokens, unique_tokens, descriptor_type, normalization_type)

        else:
            st.header("TF-IDF")
            matrix = create_freq_matrix(sents_tokens, unique_tokens, descriptor_type, normalization_type)

            # Calculate TF-IDF
            matrix = calculate_tfidf(matrix, len(sentences))

            # Display TF-IDF Matrix
            st.subheader("TF-IDF matrix")
            st.dataframe(pd.DataFrame(matrix, columns=unique_tokens, index=[f'Doc {i+1}' for i in range(len(sentences))]))

            # Document Frequency (DF) for each word
            df = (pd.DataFrame(matrix, columns=unique_tokens) > 0).sum(axis=0)
            
            selected_sentence = st.selectbox(
                "Select a sentence to calculate the TF-IDF New vector:",
                options=sentences,
                format_func=lambda x: x[:100] + "..." if len(x) > 100 else x  # Show a part of the sentence if long
            )

            # Calculate TF-IDF_New for the selected sentence
            if selected_sentence:
                tfidf_new_vector = tfidf_new(unique_tokens, selected_sentence.split(), df)
                st.subheader("TF-IDF New")
                st.dataframe(tfidf_new_vector)

        similarity_matrix = []
        # Distance Calculation Based on User Choice
        if distance_type == "Euclidean":
            similarity_matrix = create_similarity_matrix(sentences, matrix, distance_euclidean)
        elif distance_type == "Manhattan":
            similarity_matrix = create_similarity_matrix(sentences, matrix, distance_manhattan)
        elif distance_type == "Cosine":
            similarity_matrix = create_similarity_matrix(sentences, matrix, distance_cosinus)
        elif distance_type == "Jaccard":
            similarity_matrix = create_similarity_matrix(sentences, matrix, distance_jaccard)
        elif distance_type == "Hamming":
            similarity_matrix = create_similarity_matrix(sentences, matrix, distance_hamming)
        elif distance_type == "Bray-Curtis":
            similarity_matrix = create_similarity_matrix(sentences, matrix, distance_bray_curtis)
        elif distance_type == "Kullback-Leibler":
            similarity_matrix = create_similarity_matrix(sentences, matrix, distance_kullback_leibler)
        
        similarity_df = pd.DataFrame(similarity_matrix, 
                                     columns=[f'Doc {i+1}' for i in range(len(sentences))],
                                     index=[f'Doc {i+1}' for i in range(len(sentences))])
        st.subheader("Similarity matrix")
        st.dataframe(similarity_df)

        st.subheader("Document similarity")
        # Select a document to find the closest ones
        doc_query = st.number_input("Enter the document number (1 to N):", 
                                    min_value=1, max_value=len(sentences), step=1) - 1
        k = st.slider("Select the number of similar documents to display:", 1, len(sentences)-1, 3)

        # Calculate the closest documents
        if st.button("Find"):
            k_closest = k_nearest_documents(doc_query, k, similarity_matrix)
            st.dataframe(pd.DataFrame(k_closest, 
                                      columns=['Document number', 'Similarity score'],
                                      index=None))