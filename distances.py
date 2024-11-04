import math

# Euclidean
def distance_euclidean(vect1, vect2):
    s = 0
    for i in range(len(vect1)):
        s += (vect1[i] - vect2[i]) ** 2
    
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
    norm1 = sum(v ** 2 for v in vect1)
    norm2 = sum(v ** 2 for v in vect2)
    dot_product = sum(v1 * v2 for v1, v2 in zip(vect1, vect2))
    
    if norm1 == 0 or norm2 == 0:
        return 1  # Maximal dissimilarity if any vector has no magnitude
    
    return 1 - (dot_product / (math.sqrt(norm1) * math.sqrt(norm2)))

# -----------------------------------------------------------#

# Jaccard
def distance_jaccard(vect1, vect2):
    union_count = 0
    intersection_count = 0
    
    for i in range(len(vect1)):
        if vect1[i] != 0 or vect2[i] != 0:
            union_count += 1
            if vect1[i] == vect2[i]:
                intersection_count += 1

    return 1 - (intersection_count / union_count) if union_count != 0 else 1

# -----------------------------------------------------------#

# Hamming
def distance_hamming(vect1, vect2):
    if len(vect1) != len(vect2):
        raise ValueError("Vectors must be of the same length for Hamming distance.")
    
    return sum(1 for v1, v2 in zip(vect1, vect2) if v1 != v2)

# -----------------------------------------------------------#

# Bray-Curtis
def distance_bray_curtis(vect1, vect2):
    sum_abs_diff = sum(abs(v1 - v2) for v1, v2 in zip(vect1, vect2))
    sum_values = sum(v1 + v2 for v1, v2 in zip(vect1, vect2))

    return sum_abs_diff / sum_values if sum_values != 0 else 0
    
# -----------------------------------------------------------#

# Kullback Leibler
def distance_kullback_leibler(vect1, vect2):
    kl_divergence = 0
    for v1, v2 in zip(vect1, vect2):
        if v1 > 0 and v2 > 0:
            kl_divergence += v1 * math.log(v1 / v2)
    
    return kl_divergence