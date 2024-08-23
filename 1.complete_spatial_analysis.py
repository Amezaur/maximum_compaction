import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway, gaussian_kde, zscore
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans

# Replace with the actual path to your Excel file
file_path = 'C:/Users/Zamfi/Dropbox/PC (2)/Desktop/screen_coordinates.xlsx'

# Load the data
data = pd.read_excel(file_path)

# Helper function to calculate the centroids of each category
def calculate_centroids(data):
    centroids = {}
    for category in data['Category'].unique():
        category_data = data[data['Category'] == category]
        centroid = category_data[['X', 'Y']].mean().to_numpy()
        centroids[category] = centroid
    return centroids

# Helper function to calculate variance and standard deviation for each category
def calculate_variance_std(data):
    variance_std = {}
    for category in data['Category'].unique():
        category_data = data[data['Category'] == category]
        variance = category_data[['X', 'Y']].var().to_numpy()
        std_dev = category_data[['X', 'Y']].std().to_numpy()
        variance_std[category] = (variance, std_dev)
    return variance_std

# Helper function to calculate convex hull areas for each category
def calculate_convex_hull_areas(data):
    convex_hull_areas = {}
    for category in data['Category'].unique():
        category_data = data[data['Category'] == category]
        if not category_data.empty:
            points = category_data[['X', 'Y']].to_numpy()
            if len(points) > 2:  # ConvexHull requires at least 3 points
                hull = ConvexHull(points)
                area = hull.volume
                convex_hull_areas[category] = area
            else:
                print(f"Not enough points for category {category}. Points given: {len(points)}")
                convex_hull_areas[category] = 0
        else:
            print(f"No data for category {category}.")
            convex_hull_areas[category] = 0
    return convex_hull_areas

# Helper function for nearest neighbor analysis for each category
def nearest_neighbor_analysis(data):
    nn_analysis = {}
    for category in data['Category'].unique():
        category_data = data[data['Category'] == category]
        points = category_data[['X', 'Y']].to_numpy()
        distance_matrix = squareform(pdist(points))
        np.fill_diagonal(distance_matrix, np.inf)
        mean_distance = np.mean(np.min(distance_matrix, axis=1))
        std_dev = np.std(np.min(distance_matrix, axis=1))
        nn_analysis[category] = (mean_distance, std_dev)
    return nn_analysis

# Helper function to perform K-means clustering
def kmeans_clustering(data, n_clusters=3):
    kmeans_results = {}
    for category in data['Category'].unique():
        category_data = data[data['Category'] == category]
        points = category_data[['X', 'Y']].to_numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
        kmeans_results[category] = (points, kmeans.labels_, kmeans.cluster_centers_)
    return kmeans_results

# Helper function to calculate the PDF of the spatial distribution
def calculate_pdf(data):
    pdf_results = {}
    for category in data['Category'].unique():
        category_data = data[data['Category'] == category]
        points = category_data[['X', 'Y']].to_numpy()
        kde = gaussian_kde(points.T)
        pdf_results[category] = kde
    return pdf_results

# Function to perform within-category comparisons
def compare_within_category(data):
    results = {}
    for category in data['Category'].unique():
        category_data = data[data['Category'] == category]
        group_labels = category_data['Group'].values
        unique_groups = np.unique(group_labels)
        group_combinations = combinations(unique_groups, 2)
        distances = pdist(category_data[['X', 'Y']].values)
        distances_matrix = squareform(distances)
        comparisons = {}
        for group1, group2 in group_combinations:
            group1_mask = group_labels == group1
            group2_mask = group_labels == group2
            between_group_distances = distances_matrix[group1_mask][:, group2_mask]
            comparisons[(group1, group2)] = np.mean(between_group_distances)
        results[category] = comparisons
    return results

# Function to perform between-category comparisons
def compare_between_categories(data):
    results = {}
    categories = data['Category'].unique()
    category_combinations = combinations(categories, 2)
    for category1, category2 in category_combinations:
        category1_data = data[data['Category'] == category1][['X', 'Y']].values
        category2_data = data[data['Category'] == category2][['X', 'Y']].values
        _, p_value = f_oneway(category1_data, category2_data)
        results[(category1, category2)] = p_value
    return results

# Function to calculate packing efficiency with deviation
def calculate_packing_efficiency_with_deviation(data):
    efficiency_results = {}
    np.random.seed(42)  # For reproducibility
    for category in data['Category'].unique():
        category_data = data[data['Category'] == category]
        for group in category_data['Group'].unique():
            group_data = category_data[category_data['Group'] == group]
            niche_sizes = np.random.normal(group_data['AverageSize'].iloc[0], group_data['AverageDeviation'].iloc[0], len(group_data))
            niche_area = np.sum(niche_sizes)
            plane_area = group_data['PlaneSize'].iloc[0]
            efficiency = niche_area / plane_area
            if category not in efficiency_results:
                efficiency_results[category] = []
            efficiency_results[category].append(efficiency)
    return efficiency_results

# Function to detect outliers using z-score method and remove them
def detect_and_remove_outliers(data):
    non_outliers_results = {}
    for category in data['Category'].unique():
        category_data = data[data['Category'] == category].copy()
        category_data['Packing_Efficiency'] = category_data['AverageSize'] / category_data['PlaneSize']
        z_scores = zscore(category_data['Packing_Efficiency'])
        category_data['Z_Score'] = z_scores
        non_outliers = category_data[(z_scores > -2) & (z_scores < 2)]
        non_outliers_results[category] = non_outliers
        outliers = category_data[(z_scores <= -2) | (z_scores >= 2)]
        print(f"\nOutliers in {category} packing efficiency:")
        print(outliers)
        mean_efficiency = non_outliers['Packing_Efficiency'].mean()
        std_efficiency = non_outliers['Packing_Efficiency'].std()
        print(f"Recalculated Mean Efficiency ({category}): {mean_efficiency}")
        print(f"Recalculated Std Dev ({category}): {std_efficiency}")
    return non_outliers_results

# Calculate metrics
centroids = calculate_centroids(data)
variance_std = calculate_variance_std(data)
convex_hull_areas = calculate_convex_hull_areas(data)
nn_analysis = nearest_neighbor_analysis(data)
kmeans_results = kmeans_clustering(data)
pdf_results = calculate_pdf(data)
within_category_results = compare_within_category(data)
between_category_results = compare_between_categories(data)
packing_efficiency_results = calculate_packing_efficiency_with_deviation(data)
non_outliers_results = detect_and_remove_outliers(data)

# Print metrics
print("\nCentroids:")
for category, centroid in centroids.items():
    print(f"Category: {category}, Centroid: {centroid}")

print("\nVariance and Standard Deviation:")
for category, (variance, std_dev) in variance_std.items():
    print(f"Category: {category}, Variance: {variance}, Standard Deviation: {std_dev}")

print("\nConvex Hull Areas:")
for category, area in convex_hull_areas.items():
    print(f"Category: {category}, Convex Hull Area: {area}")

print("\nNearest Neighbor Analysis:")
for category, (mean_distance, std_dev) in nn_analysis.items():
    print(f"Category: {category}, Mean Nearest Neighbor Distance: {mean_distance}, Std Dev: {std_dev}")

print("\nK-Means Clustering:")
for category, (points, labels, centers) in kmeans_results.items():
    print(f"Category: {category}, Cluster Centers: {centers}")

print("\nWithin-Category Comparisons:")
for category, comparisons in within_category_results.items():
    print(f"Category: {category}")
    for (group1, group2), mean_distance in comparisons.items():
        print(f"  Groups {group1} vs {group2}: Mean Distance = {mean_distance}")

print("\nBetween-Category Comparisons:")
for (category1, category2), p_value in between_category_results.items():
    print(f"Categories {category1} vs {category2}: p-value = {p_value}")

print("\nPacking Efficiency with Deviation:")
for category, efficiencies in packing_efficiency_results.items():
    mean_efficiency = np.mean(efficiencies)
    std_efficiency = np.std(efficiencies)
    print(f"Category: {category}, Mean Efficiency: {mean_efficiency}, Std Dev: {std_efficiency}")

# Create a PDF to save all plots
with PdfPages('analysis_plots.pdf') as pdf:

    # Function to plot and save centroids
    def plot_centroids(data, centroids):
        plt.figure(figsize=(10, 6))
        colors = {
            'control': 'green',
            'mutant_arm': 'purple',
            'mutant_hpo': 'blue',
            'mutant_inr': 'hotpink',
            'mutant_n': 'turquoise'
        }
        for idx, category in enumerate(data['Category'].unique()):
            category_data = data[data['Category'] == category]
            plt.scatter(category_data['X'], category_data['Y'], label=f'{category} Data Points', color=colors[category])
        for idx, (category, centroid) in enumerate(centroids.items()):
            plt.scatter(centroid[0], centroid[1], label=f'{category} Centroid', color=colors[category], marker='x', s=100, linewidths=3)
        plt.xlabel('X')
        plt.ylabel('Y')
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Centroids of Categories')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Function to plot nearest neighbor analysis
    def plot_nearest_neighbor_analysis(nn_analysis):
        categories = list(nn_analysis.keys())
        mean_distances = [nn_analysis[cat][0] for cat in categories]
        std_devs = [nn_analysis[cat][1] for cat in categories]
        colors = {
            'control': 'green',
            'mutant_arm': 'purple',
            'mutant_hpo': 'blue',
            'mutant_inr': 'hotpink',
            'mutant_n': 'turquoise'
        }

        plt.figure(figsize=(10, 6))
        plt.bar(categories, mean_distances, yerr=std_devs, capsize=5, color=[colors[cat] for cat in categories])
        plt.title('Nearest Neighbor Analysis')
        plt.xlabel('Category')
        plt.ylabel('Mean Nearest Neighbor Distance')
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Function to plot variance and standard deviation
    def plot_variance_std(variance_std):
        categories = list(variance_std.keys())
        variances = [variance_std[cat][0][0] for cat in categories]
        std_devs = [variance_std[cat][1][0] for cat in categories]
        colors = {
            'control': 'green',
            'mutant_arm': 'purple',
            'mutant_hpo': 'blue',
            'mutant_inr': 'hotpink',
            'mutant_n': 'turquoise'
        }

        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        axs[0].bar(categories, variances, color=[colors[cat] for cat in categories])
        axs[0].set_title('Variance of Categories')
        axs[0].set_ylabel('Variance')
        axs[0].grid(True)
        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        axs[1].bar(categories, std_devs, color=[colors[cat] for cat in categories])
        axs[1].set_title('Standard Deviation of Categories')
        axs[1].set_ylabel('Standard Deviation')
        axs[1].grid(True)
        handles, labels = axs[1].get_legend_handles_labels()
        if handles:
            axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Function to plot convex hull areas
    def plot_convex_hull_areas(convex_hull_areas):
        categories = list(convex_hull_areas.keys())
        areas = [convex_hull_areas[cat] for cat in categories]
        colors = {
            'control': 'green',
            'mutant_arm': 'purple',
            'mutant_hpo': 'blue',
            'mutant_inr': 'hotpink',
            'mutant_n': 'turquoise'
        }

        plt.figure(figsize=(10, 6))
        plt.bar(categories, areas, color=[colors[cat] for cat in categories])
        plt.title('Convex Hull Areas of Categories')
        plt.xlabel('Category')
        plt.ylabel('Area')
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Function to plot K-means clustering results
    def plot_kmeans_clustering(data, kmeans_results):
        plt.figure(figsize=(10, 6))
        categories = data['Category'].unique()
        colors = {
            'control': 'green',
            'mutant_arm': 'purple',
            'mutant_hpo': 'blue',
            'mutant_inr': 'hotpink',
            'mutant_n': 'turquoise'
        }

        for idx, category in enumerate(categories):
            category_data = data[data['Category'] == category]
            points, labels, centers = kmeans_results[category]
            plt.scatter(points[:, 0], points[:, 1], label=f'{category} data', color=colors[category], alpha=0.5, marker='o')

            for center in centers:
                plt.scatter(center[0], center[1], label=f'{category} cluster center', color=colors[category], edgecolor='black', s=200, marker='X')

        plt.title('K-means Clustering of Categories')
        plt.xlabel('X')
        plt.ylabel('Y')

        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Function to plot PDF of spatial distribution
    def plot_pdf(pdf_results, data):
        plt.figure(figsize=(10, 6))
        categories = data['Category'].unique()
        colors = {
            'control': 'green',
            'mutant_arm': 'purple',
            'mutant_hpo': 'blue',
            'mutant_inr': 'hotpink',
            'mutant_n': 'turquoise'
        }

        for idx, category in enumerate(categories):
            category_data = data[data['Category'] == category]
            kde = pdf_results[category]
            x_min, x_max = category_data['X'].min(), category_data['X'].max()
            y_min, y_max = category_data['Y'].min(), category_data['Y'].max()
            X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kde(positions).T, X.shape)
            plt.contourf(X, Y, Z, cmap='Blues', alpha=0.5)
            plt.scatter(category_data['X'], category_data['Y'], label=f'{category} data', color=colors[category], alpha=0.5, marker='o')

        plt.title('PDF of Spatial Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Function to plot individual PDF of spatial distribution for each genotype
    def plot_individual_pdfs(pdf_results, data):
        categories = data['Category'].unique()
        colors = {
            'control': 'green',
            'mutant_arm': 'purple',
            'mutant_hpo': 'blue',
            'mutant_inr': 'hotpink',
            'mutant_n': 'turquoise'
        }

        for idx, category in enumerate(categories):
            plt.figure(figsize=(10, 6))
            category_data = data[data['Category'] == category]
            kde = pdf_results[category]
            x_min, x_max = category_data['X'].min(), category_data['X'].max()
            y_min, y_max = category_data['Y'].min(), category_data['Y'].max()
            X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kde(positions).T, X.shape)
            plt.contourf(X, Y, Z, cmap='Blues', alpha=0.5)
            plt.scatter(category_data['X'], category_data['Y'], label=f'{category} data', color=colors[category], alpha=0.5, marker='o')

            plt.title(f'PDF of Spatial Distribution for {category}')
            plt.xlabel('X')
            plt.ylabel('Y')
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    # Function to plot packing efficiency
    def plot_packing_efficiency(non_outliers_results):
        categories = non_outliers_results.keys()
        mean_efficiencies = [non_outliers_results[cat]['Packing_Efficiency'].mean() for cat in categories]
        std_efficiencies = [non_outliers_results[cat]['Packing_Efficiency'].std() for cat in categories]
        colors = {
            'control': 'green',
            'mutant_arm': 'purple',
            'mutant_hpo': 'blue',
            'mutant_inr': 'hotpink',
            'mutant_n': 'turquoise'
        }

        plt.figure(figsize=(10, 6))
        plt.bar(categories, mean_efficiencies, yerr=std_efficiencies, capsize=5, color=[colors[cat] for cat in categories])
        plt.title('Packing Efficiency by Category')
        plt.xlabel('Category')
        plt.ylabel('Mean Packing Efficiency')
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Call the plot functions
    plot_centroids(data, centroids)
    plot_nearest_neighbor_analysis(nn_analysis)
    plot_variance_std(variance_std)
    plot_convex_hull_areas(convex_hull_areas)
    plot_kmeans_clustering(data, kmeans_results)
    plot_pdf(pdf_results, data)
    plot_individual_pdfs(pdf_results, data)
    plot_packing_efficiency(non_outliers_results)

print("All plots saved to analysis_plots.pdf")
