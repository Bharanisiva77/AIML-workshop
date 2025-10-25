# k Means Clustering
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import kagglehub

path = kagglehub.dataset_download("shwetabh123/mall-customers")

df = pd.read_csv(f"{path}/Mall_Customers.csv")

df["Genre"]=df["Genre"].map({"Female":0,"Male":1})
features = ["Age","Annual Income (k$)","Spending Score (1-100)"]
x=StandardScaler().fit_transform(df[features])

kmeans = KMeans(n_clusters=5,random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(x)

tsne = TSNE(n_components=2,random_state=42)
x_embedded=tsne.fit_transform(x)

plt.scatter(x_embedded[:,0],x_embedded[:,1],c=df['Cluster'],cmap="tab10")
plt.title("Customer Segmentation with K Means and t-SNE")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()