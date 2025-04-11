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


# ------------------Load Data-----------------------------------------------

script_dir = os.path.dirname(os.path.realpath(__file__))
data_location = os.path.join(script_dir, "../data/books.csv")

data = pd.read_csv(data_location, index_col=0)
print(data.columns)

# ------------------Preprocess Data-----------------------------------------

# remove quantities column; its not useful
if 'quantity' in data.columns:
    data = data.drop(columns=['quantity'])

data = pd.get_dummies(data, columns=['author', 'series'])

cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})

X = data
y = data.index

print(X.head())
