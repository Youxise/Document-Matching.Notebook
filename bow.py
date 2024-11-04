


# ----------------------------------------------------------------------------#

# K most similar documents
def k_nearest_documents(doc_query, k, similarity_matrix):
    # si le document est parmis la liste
    similarites = similarity_matrix[doc_query]
    similarites_idx = [(i, similarites[i]) for i in range(len(similarites)) if i != doc_query]
    similarites_idx.sort(key=lambda x: x[1], reverse=True)
    return similarites_idx[:k]