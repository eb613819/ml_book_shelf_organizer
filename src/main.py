# Machine Learning Book Shelf Organizer
#   Evan Brooks
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import VarianceThreshold
import scipy.cluster.hierarchy as sch

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)

output_dir = os.path.join(parent_dir, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ------------------Load Data-----------------------------------------------

data_location = os.path.join(script_dir, "../data/books_calibre.csv")

data = pd.read_csv(data_location, index_col=0)
print(data.columns)

#-------------------Preprocess Data-----------------------------------------

# remove quantities column; its not useful
if 'quantity' in data.columns:
    data = data.drop(columns=['quantity'])
# remove series column; its missing values
if 'series' in data.columns:
    data = data.drop(columns=['series'])
# remove rating column; its missing values
if 'rating' in data.columns:
    data = data.drop(columns=['rating'])
# remove comments column; its not useful
if 'comments' in data.columns:
    data = data.drop(columns=['comments'])
# remove isbn column; its not useful
if 'isbn' in data.columns:
    data = data.drop(columns=['isbn'])
# remove identifiers column; its not useful
if 'identifiers' in data.columns:
    data = data.drop(columns=['identifiers'])
# remove pubdate column; its not useful
if 'pubdate' in data.columns:
    data = data.drop(columns=['pubdate'])
# remove publisher column; its not useful
if 'publisher' in data.columns:
    data = data.drop(columns=['publisher'])
# remove uuid column; its not useful
if 'uuid' in data.columns:
    data = data.drop(columns=['uuid'])

# drop any rows where the index (book title) is 'books_calibre'
data = data[data.index != 'books_calibre']

# one hot encode genres
data['tags_list'] = data['tags'].str.split(',').apply(lambda x: [tag.strip() for tag in x])
mlb = MultiLabelBinarizer()
encoded_tags = pd.DataFrame(mlb.fit_transform(data['tags_list']),
                              columns=mlb.classes_,
                              index=data.index)
data = pd.concat([data, encoded_tags], axis=1)
data = data.drop(columns=['tags', 'tags_list'])

# one hot encode authors and languages
data = pd.get_dummies(data, columns=['authors', 'languages'], dummy_na=False)



#-------------------Feature Selection---------------------------------------
min_books = 2
max_books_percent= 0.5
num_books = len(data)
max_books = num_books * max_books_percent

feature_counts = data.apply(lambda x: x.sum(), axis=0)
features_to_keep = feature_counts[(feature_counts >= min_books) & (feature_counts <= max_books)].index
data = data[features_to_keep]
#-------------------Feature Scaling-----------------------------------------
author_cols = [col for col in data.columns if col.startswith('authors_')]
languages_cols = [col for col in data.columns if col.startswith('languages_')]
data[author_cols] *= 3.0
data[languages_cols] *= 2
data['Language'] *= 2
data['Classics'] *= 2
data['Romance'] *= 2
data['Science Fiction'] *= 2
data['Young Adult Fiction'] *= 2
data['Philosophy'] *= 2
print(data.head())
#-------------------Clustering----------------------------------------------

X = data
y = data.index

clustering = AgglomerativeClustering(
    metric='cosine',
    linkage='average',
    distance_threshold=0.72,
    n_clusters=None          
)

labels = clustering.fit_predict(X)
num_shelves = len(set(labels))

plt.figure(figsize=(15, 15))
sch.dendrogram(
    sch.linkage(X, method='average', metric='cosine'),
    labels=data.index,
    leaf_rotation=90,  
    leaf_font_size=8
)
plt.title('Dendrogram of Agglomerative Clustering')
plt.xlabel('Books')
plt.ylabel('Distance')
plt.savefig(os.path.join(output_dir, "dendro.png"))

data['cluster'] = labels
cluster_labels = data['cluster']
X['cluster'] = labels

output_data = X.copy()
output_file_all_features = os.path.join(output_dir, "books_with_features_and_clusters.csv")
output_data.to_csv(output_file_all_features, index=False)

shelves = {}
for cluster in range(num_shelves):
    shelf_books = data[data['cluster'] == cluster].index.tolist() 
    shelves[f'Shelf_{cluster+1}'] = shelf_books

shelves_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in shelves.items()]))

output_file = os.path.join(output_dir, "books_by_shelf.csv")
shelves_df.to_csv(output_file, index=False)

#-------------------Dimensionality Reduction and Visualization--------------
# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis')
plt.title('PCA Visualization of Book Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig(os.path.join(output_dir, "pca_clusters.png"))
plt.close()

# t-SNE
tsne = TSNE(n_components=2, random_state=42, n_iter=300, verbose=0)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=cluster_labels, palette='viridis')
plt.title('t-SNE Visualization of Book Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.savefig(os.path.join(output_dir, "tsne_clusters.png"))
plt.close()

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=cluster_labels, palette='viridis')
plt.title('UMAP Visualization of Book Clusters')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig(os.path.join(output_dir, "umap_clusters.png"))
plt.close()
