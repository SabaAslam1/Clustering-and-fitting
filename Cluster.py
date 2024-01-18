import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import numpy as np

# Create a StandardScaler instance
scaler = StandardScaler()


def read_data(file_paths, selected_country, start_year, end_year):
    dataframes_list = []

    for path in file_paths:
        file_name = path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path, skiprows=4)
        df = df.rename(columns={'Country Name': 'Country'})
        df = df.set_index('Country')
        df_selected_country = df.loc[selected_country, str(start_year):str(end_year)].transpose().reset_index()
        df_selected_country = df_selected_country.rename(columns={'index': 'Year', selected_country: file_name})
        dataframes_list.append(df_selected_country)

    # Concatenate all DataFrames based on the 'Year' column
    result_df = pd.concat(dataframes_list, axis=1)

    # Replace null values with the mean of each column
    result_df = result_df.apply(lambda col: col.fillna(col.mean()))

    return result_df


def calculate_silhouette_score(xy, n_clusters):
    """Calculates silhouette score for n_clusters."""
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


def plot_scatter_matrix(dataframe):
    """Plots scatter matrix for the given DataFrame."""
    pd.plotting.scatter_matrix(dataframe, figsize=(9.0, 9.0))
    plt.tight_layout()
    plt.show()


def plot_heatmap(correlation_matrix):
    """Plots heatmap for the given correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Scaled Data')
    plt.show()


def plot_cluster(dataframe, x_col, y_col, n_clusters, title, save_filename):
    """Plots clusters for the given DataFrame and columns."""
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(dataframe)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    plt.figure(figsize=(6.0, 6.0))
    plt.scatter(dataframe[x_col], dataframe[y_col], c=labels, cmap="tab10")
    xc, yc = cen[:, 0], cen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    plt.xlabel(x_col, fontweight='bold')
    plt.ylabel(y_col, fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.savefig(save_filename, dpi=300)
    plt.show()


# Example usage
file_paths = [
    "DeathInjuries.csv", 'Slums.csv',
    'Population density.csv', 'Urban Population.csv']
selected_country = "India"
start_year = 2000
end_year = 2022

result_df = read_data(file_paths, selected_country, start_year, end_year)
result_df = result_df.drop('Year', axis=1)
result_df_scaled = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns)

print(result_df_scaled)

plot_scatter_matrix(result_df_scaled)

correlation_matrix = result_df_scaled.corr()
plot_heatmap(correlation_matrix)

# Cluster plots
cluster_df_1 = result_df[['Urban Population', 'Population density']]
plot_cluster(cluster_df_1, "Urban Population", "Population density", 2,
             "Cluster between Population density & Urban Population",
             'Cluster_urban_vs_Density.png')

cluster_df_2 = result_df[['Population density', 'Slums']]
plot_cluster(cluster_df_2, "Population density", "Slums", 2,
             "Cluster between Slum population & urban population density",
             'Cluster_Slums_vs_Density.png')
