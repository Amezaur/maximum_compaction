import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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
data['NumberOfNiches'] = data.groupby(['Group', 'Category'])['X'].transform('count')

# Aggregate data to have one row per ovary
agg_data = data.groupby(['Group', 'Category', 'PlaneSize', 'AverageSize', 'AverageDeviation']).agg({
    'X': 'first',
    'Y': 'first',
    'NumberOfNiches': 'first',
}).reset_index()

# Add 'Interaction' column as the product of features
agg_data['Interaction'] = (agg_data['NumberOfNiches'] *
                           agg_data['AverageSize'] *
                           agg_data['AverageDeviation'] *
                           agg_data['X'] *
                           agg_data['Y'])

# Apply bins and labels to the aggregated data
agg_data['PlaneSizeRange'] = pd.cut(agg_data['PlaneSize'], bins=bins, labels=labels)

# Define the segments
segments = {
    'Less_than_18': agg_data[agg_data['NumberOfNiches'] < 18],
    'Control_18_20': agg_data[(agg_data['NumberOfNiches'] >= 18) & (agg_data['NumberOfNiches'] <= 20)],
    'More_than_20': agg_data[agg_data['NumberOfNiches'] > 20]
}

# Initialize results dictionary
results = {}

# Create a new output directory
output_dir = 'rf_model_results'
os.makedirs(output_dir, exist_ok=True)

# Process each segment
for segment_name, segment_data in segments.items():
    if segment_data.empty:
        print(f"No data for segment {segment_name}")
        continue

    # Define features and target
    features = ['AverageSize', 'AverageDeviation', 'NumberOfNiches', 'Interaction', 'X', 'Y']
    target = 'PlaneSizeRange'

    X = segment_data[features]
    y = segment_data[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest with default hyperparameters
    print(f"Training RandomForest for {segment_name}")
    rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, bootstrap=True)
    rf.fit(X_train, y_train)
    best_model = rf

    # Evaluate model
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = np.mean(y_test == y_pred)

    results[f'{segment_name}_RandomForest'] = {
        'model': best_model,
        'report': report,
        'confusion_matrix': cm,
        'accuracy': accuracy
    }

    # Print results
    print(f"Results for {segment_name} using RandomForest:")
    print(f"  Classification Report: {report}")
    print(f"  Confusion Matrix: {cm}")
    print(f"  Accuracy: {accuracy}")

    # Get the unique labels for the confusion matrix
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))

    # Save classification report
    report_path = os.path.join(output_dir, f'classification_report_{segment_name}_RandomForest.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(y_test, y_pred, zero_division=1))

    # Save confusion matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    cm_display.plot(cmap='Blues')
    plt.title(f'Confusion Matrix for {segment_name} using RandomForest')
    cm_path = os.path.join(output_dir, f'confusion_matrix_{segment_name}_RandomForest.png')
    plt.savefig(cm_path)
    plt.close()

    # Feature Importance Plot
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances for {segment_name}")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    feature_importance_path = os.path.join(output_dir, f'feature_importances_{segment_name}_RandomForest.png')
    plt.savefig(feature_importance_path)
    plt.close()

    # SHAP analysis
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train, plot_type="bar")
    shap_summary_path = os.path.join(output_dir, f'shap_summary_{segment_name}_RandomForest.png')
    plt.title(f'SHAP Summary for {segment_name} using RandomForest')
    plt.savefig(shap_summary_path)
    plt.show()  # Add plt.show() to display the SHAP plot in the window before saving
