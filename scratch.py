import json
import numpy as np
import pandas as pd
import requests
import warnings
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
# from km_model import KMModel


smog_ds = json.loads(requests.get('https://public-esa.ose.gov.pl/api/v1/smog').text)

locations = [{'longitude': item['school']['longitude'], 'latitude': item['school']['latitude']} for item in smog_ds['smog_data']]
#I want to find minimum value of longitude and latitude in locations. Copilot provide me a code

df = pd.DataFrame(locations)

# Convert longitude and latitude to numeric values
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df.dropna(subset=['longitude', 'latitude'], inplace=True)
df = df[(df['longitude'] != 0) & (df['latitude'] != 0)]
coordinates = df.loc[:, ['longitude', 'latitude']]
plt.scatter(df.loc[:, 'longitude'], df.loc[:, 'latitude'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Smog Data Locations')
plt.show()
#
# WCSS = []
#
# for k in range(1,17):
#     kmeans = KMeans(n_clusters=k, n_init=100)
#     kmeans.fit(coordinates)
#     WCSS.append(kmeans.inertia_)
#
# plt.plot(range(1, 17), WCSS)
# plt.xlabel('Number w cluster')
# plt.ylabel('WCSS')
# plt.grid()
# plt.show()

kmeans = KMeans(n_clusters=6, n_init=300, random_state=1)
kmeans.fit_predict(coordinates.values)
h = 0.001
x_min, x_max = coordinates['longitude'].min(), coordinates['longitude'].max()
y_min, y_max = coordinates['latitude'].min(), coordinates['latitude'].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.figure(1, figsize=(10,4))
plt.clf()
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel1, origin='lower')
plt.scatter(x=coordinates['longitude'], y=coordinates['latitude'], c=labels, s=100)
plt.scatter(x=centroids[:, 0], y=centroids[:, 1], s=300, c='red')
plt.xlabel('Longitude (x)')
plt.ylabel('Latitude (y)')
plt.title('Clustering')
plt.show()


