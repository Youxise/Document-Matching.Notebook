import math

# Euclidean
def distance_euclidean(vect1, vect2):
    s = 0
    for i in range(len(vect1)):
        s += math.pow(vect1[i] - vect2[i], 2)
    
    return math.sqrt(s)

# -----------------------------------------------------------#

# Manhattan
def distance_manhattan(vect1, vect2):
    s = 0
    for i in range(len(vect1)):
        s += abs(vect1[i] - vect2[i])
    
    return s

# -----------------------------------------------------------#

# Cosine
def distance_cosinus(vect1, vect2):
    somme_norm1 = 0
    somme_norm2 = 0
    produit_scalaire = 0
    for i in range(len(vect1)):
        somme_norm1 += vect1[i]**2
        somme_norm2 += vect2[i]**2
        produit_scalaire += vect1[i]*vect2[i]

    return (1 - (produit_scalaire/(math.sqrt(somme_norm1)*math.sqrt(somme_norm2))))

# -----------------------------------------------------------#

# Jaccard
def distance_jaccard(vect1, vect2):
    somme_union = 0
    somme_intersec = 0
    for i in range(len(vect1)):
        if vect1[i] != vect2[i]:
            somme_union += 1
        elif vect1[i] != 0 or vect2[i] != 0: # s'assurer que les deux mots se trouvent au moins une fois dans les documents
            somme_intersec += 1
            
    return (1 - (somme_intersec/somme_union))

# -----------------------------------------------------------#

# Hamming
def distance_hamming(vect1, vect2):
    somme_diff = 0
    for i in range(len(vect1)):
        if vect1[i] != vect2[i]:
            somme_diff += 1
            
    return somme_diff

# -----------------------------------------------------------#

# Bray-Curtis
def distance_bray_curtis(vect1, vect2):
    somme_abs = 0
    somme = 0
    for i in range(len(vect1)):
        somme_abs += abs(vect1[i] - vect2[i])
        somme += vect1[i] + vect2[i]

    if somme != 0:
        return somme_abs / somme
    else:
        return 0
    
# -----------------------------------------------------------#

# Kullback Leibler
def distance_kullback_leibler(vect1, vect2):
    somme = 0
    for i in range(len(vect1)):
        if vect1[i] > 0 and vect2[i] > 0:
            somme += vect1[i] * math.log(vect1[i]/vect2[i])

    return somme