# Mall Customer Segmentation

## Overview
This project demonstrates how to perform customer segmentation using the K-Means clustering algorithm. Customer segmentation is a key marketing strategy to identify distinct groups of customers based on their behaviors, which helps businesses tailor their strategies for better customer satisfaction and profit maximization.

## Dataset
The dataset used for this project is `Mall_Customers.csv`. It contains the following attributes:
- **CustomerID**: Unique ID assigned to each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Annual Income (k$)**: Annual income of the customer in thousands of dollars
- **Spending Score (1-100)**: Score assigned to the customer based on their spending behavior and purchasing data

## Project Steps

### 1. Data Loading and Exploration
- Load the dataset using Pandas.
- Display the first 5 rows to understand the structure of the data.
- Check the shape of the dataset to understand its dimensions.
- Inspect for missing values and general data information.

### 2. Data Preprocessing
- Extract relevant features for clustering: `Annual Income` and `Spending Score`.

### 3. Elbow Method to Determine Optimum Number of Clusters
- Compute the Within-Cluster Sum of Squares (WCSS) for different numbers of clusters (1 to 10).
- Plot the WCSS values to identify the "elbow point."

### 4. K-Means Clustering
- Train the K-Means model with the optimal number of clusters (5).
- Predict the cluster labels for the data points.

### 5. Visualization
- Visualize the clusters in a scatter plot.
- Highlight the centroids of the clusters.

## Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# loading the data from csv file to a Pandas DataFrame
customer_data = pd.read_csv('/content/Mall_Customers.csv')

# first 5 rows in the dataframe
print(customer_data.head())

# finding the number of rows and columns
print(customer_data.shape)

# getting some information about the dataset
print(customer_data.info())

# checking for missing values
print(customer_data.isnull().sum())

# Selecting relevant features for clustering
X = customer_data.iloc[:, [3, 4]].values
print(X)

# finding wcss value for different number of clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# plot an elbow graph
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Optimum Number of Clusters = 5

# Training the k-Means Clustering Model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)
print(Y)

# Visualizing all the Clusters
plt.figure(figsize=(8, 8))
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='blue', label='Cluster 5')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
```

## Results
The K-Means algorithm identified 5 distinct clusters of customers based on their annual income and spending score. Each cluster represents a unique group of customers with similar purchasing behaviors, allowing businesses to design targeted marketing strategies.

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run
1. Clone the repository.
2. Install the required libraries using:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Place the `Mall_Customers.csv` file in the appropriate directory.
4. Run the Python script.
