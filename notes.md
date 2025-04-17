# 📚 Problem Description: Limitations of K-Means for Book Shelf Clustering

I'm organizing my book collection onto a fixed number of shelves by clustering books based on metadata like genre, author, series, and language. I initially chose **K-Means clustering** because it provides a straightforward mapping between the number of clusters and the number of shelves.

However, after testing and reviewing the results, I encountered several limitations with K-Means in this context:

## ⚠️ Issues with K-Means

### 1. Assumes Spherical, Equally-Sized Clusters
- K-Means tends to form clusters of similar size and shape.
- This doesn't match the natural structure of my library, where some clusters (e.g., large series or popular genres) are much larger than others.

### 2. Insensitive to Categorical/One-Hot Features
- Most of my book metadata is **categorical** (e.g., genre tags, language, author, series).
- After one-hot encoding, K-Means doesn’t handle these features well. All distances are treated equally, even if some features (like series or language) should weigh more heavily in grouping decisions.

### 3. Sensitive to Initialization
- K-Means results vary across runs due to random initialization.
- Even when fixing the seed, it often splits books from the same series or author into different clusters.

### 4. Poor Interpretability
- Final cluster groupings don’t always make intuitive sense.
- Books that seem unrelated end up on the same "shelf," while logical groups are split.

---

## 🔁 Planned Changes

To address these limitations, I plan to experiment with the following improvements:

### ✅ Switch to Hierarchical Clustering
- Hierarchical clustering doesn’t require assumptions about cluster size or shape.
- It produces a dendrogram showing nested groupings, which can be **cut at the number of shelves** I have to form clusters.
- This approach better respects relationships in categorical data (e.g., series, authors).

### ✅ Apply Feature Weights
- Not all features are equally important.
- I plan to **weight features like author, series, and language more heavily** than general genre tags so that books from the same series or language are more likely to be grouped together.

### ✅ Improve Distance Metric
- Euclidean distance isn’t ideal for high-dimensional binary vectors.
- I’ll explore **Hamming distance** or **cosine similarity**, which are more appropriate for sparse binary data.

### ✅ Optional: Use Agglomerative Clustering with Precomputed Distance
- If using a custom similarity matrix (e.g., from cosine or Hamming distances), I can feed this directly into agglomerative clustering.

---

These changes should allow shelf groupings that are more natural, interpretable, and useful — especially for keeping series and multilingual books together while making good use of shelf space.



| Challenge                                  | K-Means Struggles | Why                                                                 |
|--------------------------------------------|-------------------|----------------------------------------------------------------------|
| **Sparse features** (e.g. tags, genres)     | 😬 Yes            | K-Means relies on Euclidean distance — doesn’t handle lots of 0s well |
| **Binary features** (tag/no tag)            | 😬 Yes            | Euclidean treats 0→1 and 1→0 equally, even when meaning differs      |
| **Non-spherical clusters**                 | 😬 Yes            | K-Means assumes clusters are round and roughly equal in size         |
| **Differing cluster sizes**                | 😬 Yes            | K-Means favors similar-sized clusters — splits large groups, merges small ones |
| **Need to keep authors/series together**   | 😬 Yes            | No built-in sense of hierarchy or structure                          |
| **Random cluster seeds**                   | 😬 Yes            | Results vary run-to-run, especially with sparse data                 |


| Feature                                          | Agglomerative Clustering |
|--------------------------------------------------|---------------------------|
| Doesn’t assume shape/size of clusters?           | ✅                         |
| Works well with sparse/binary data?              | ✅ (especially with cosine/Hamming distance) |
| Lets you preserve structure (series, authors)?   | ✅ Often groups them naturally |
| Avoids random init/seeds?                        | ✅ Deterministic process   |
| Lets you choose how many shelves later?          | ✅ Cut the dendrogram wherever you want |
