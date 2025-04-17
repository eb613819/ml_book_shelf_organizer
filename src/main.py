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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import VarianceThreshold

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

X = data
y = data.index

print(X.head())

#-------------------Feature Selection---------------------------------------
selector = VarianceThreshold(threshold=0.2)
X_selected = selector.fit_transform(X)
selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features, index=data.index)
#-------------------Clustering----------------------------------------------


num_shelves = 12
kmeans = KMeans(n_clusters=num_shelves, random_state=42)

kmeans.fit(X)

data['cluster'] = kmeans.labels_
cluster_labels = data['cluster']

print(data[['cluster']].head())

output_data = data.copy()
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
