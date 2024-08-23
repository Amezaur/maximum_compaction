import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

# Load the data
file_path = 'screen_coordinates.xlsx'
data = pd.read_excel(file_path)

# Drop rows with NaN values in 'PlaneSize'
data.dropna(subset=['PlaneSize'], inplace=True)

# Define bins and labels for PlaneSizeRange
bins = [0, 1600, 2400, np.inf]
labels = ['Small', 'Medium', 'Large']

# Ensure 'Group' column is correctly set as category or numeric type
data['Group'] = data['Group'].astype(str)

# Add 'NumberOfNiches' column by counting the occurrences per 'Group' and 'Category'
data['NumberOfNiches'] = data.groupby(['Group', 'Category'])['AverageSize'].transform('count')

# Aggregate data to have one row per ovary
agg_data = data.groupby(['Group', 'Category', 'PlaneSize']).agg({
    'NumberOfNiches': 'first',
    'AverageSize': 'first',
    'AverageDeviation': 'first'
}).reset_index()

# Add 'Interaction' column as the product of features
agg_data['Interaction'] = (agg_data['NumberOfNiches'] *
                           agg_data['AverageSize'] *
                           agg_data['AverageDeviation'])

# Apply bins and labels to the aggregated data
agg_data['PlaneSizeRange'] = pd.cut(agg_data['PlaneSize'], bins=bins, labels=labels)

# Define the segments
segments = {
    'Less_than_18': agg_data[agg_data['NumberOfNiches'] < 18],
    'Control_18_20': agg_data[(agg_data['NumberOfNiches'] >= 18) & (agg_data['NumberOfNiches'] <= 20)],
    'More_than_20': agg_data[agg_data['NumberOfNiches'] > 20]
}

# Create a new directory to save the outputs
output_dir = 'rf_analysis_results'
os.makedirs(output_dir, exist_ok=True)

# Initialize SHAP analysis
for segment_name, segment_data in segments.items():
    if segment_data.empty:
        print(f"No data for segment {segment_name}")
        continue

    # Define all features for analysis, even if some are not used in final model training
    features = ['Interaction', 'AverageSize', 'AverageDeviation', 'NumberOfNiches']
    X = segment_data[features]

    # Train a basic RandomForest model to get SHAP values and feature importances
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, segment_data['PlaneSizeRange'])

    # Feature Importance Plot
    importances = model.feature_importances_
    plt.figure()
    plt.title(f"Feature Importances for {segment_name}")
    plt.bar(range(len(features)), importances, color="r", align="center")
    plt.xticks(range(len(features)), features, rotation=90)
    plt.xlim([-1, len(features)])
    feature_importance_path = os.path.join(output_dir, f'feature_importances_{segment_name}.png')
    plt.savefig(feature_importance_path)
    plt.close()

    # SHAP Analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle different numbers of classes
    if isinstance(shap_values, list) and len(shap_values) > 1:
        for i, shap_value in enumerate(shap_values):
            shap.summary_plot(shap_value, X, plot_type="bar")
            shap_summary_path = os.path.join(output_dir, f'shap_summary_{segment_name}_class_{i}.png')
            plt.title(f'SHAP Summary for {segment_name} - Class {i}')
            plt.savefig(shap_summary_path)
            plt.close()
    else:
        shap.summary_plot(shap_values, X, plot_type="bar")
        shap_summary_path = os.path.join(output_dir, f'shap_summary_{segment_name}.png')
        plt.title(f'SHAP Summary for {segment_name}')
        plt.savefig(shap_summary_path)
        plt.close()
