import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib  # To load the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime

# Create output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Function to generate a unique filename
def generate_filename(base_name, extension):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{base_name}_{timestamp}.{extension}'

# Load the new data
new_data_path = 'new_screen_coordinates.xlsx'
new_data = pd.read_excel(new_data_path)

# Drop any NaN values in the key columns if necessary
new_data.dropna(subset=['AverageSize', 'AverageDeviation', 'PlaneSize'], inplace=True)

# Add 'NumberOfNiches' column by counting the occurrences per 'Group' and 'Category'
new_data['NumberOfNiches'] = new_data.groupby(['Group', 'Category'])['AverageSize'].transform('count')

# Aggregate new data similarly as before
agg_new_data = new_data.groupby(['Group', 'Category']).agg({
    'NumberOfNiches': 'first',
    'AverageSize': 'first',
    'AverageDeviation': 'first',
    'PlaneSize': 'first'
}).reset_index()

# Add the interaction feature and compaction index
agg_new_data['Interaction'] = (agg_new_data['NumberOfNiches'] *
                               agg_new_data['AverageSize'] *
                               agg_new_data['AverageDeviation'])

agg_new_data['CompactionIndex'] = (agg_new_data['NumberOfNiches'] * agg_new_data['AverageSize']) / agg_new_data['PlaneSize']

# Apply the bins and labels to the new data
bins = [0, 1600, 2400, np.inf]
labels = ['Small', 'Medium', 'Large']
agg_new_data['True_PlaneSizeRange'] = pd.cut(agg_new_data['PlaneSize'], bins=bins, labels=labels)

# Define the new segments based on the models created
segments_new = {
    'Less_than_18': agg_new_data[agg_new_data['NumberOfNiches'] < 18],
    'Control_18_20': agg_new_data[(agg_new_data['NumberOfNiches'] >= 18) & (agg_new_data['NumberOfNiches'] <= 20)],
    'More_than_20': agg_new_data[agg_new_data['NumberOfNiches'] > 20]
}

# Extract all unique categories from your data
unique_categories = agg_new_data['Category'].unique()

# Dynamically assign colors based on the categories present
colors = {
    'control': 'green',
    'mutant_myc': 'purple',
    'mutant_stg': 'orange',
    'mutant_cycb': 'turquoise',
    'mutant_sax': 'hotpink'
}

# Assign a default color for any categories not in the predefined palette
for category in unique_categories:
    if category not in colors:
        colors[category] = sns.color_palette("husl", len(unique_categories)).as_hex()[len(colors) % len(unique_categories)]

# Define the markers for each size
markers = {
    'Small': 'o',
    'Medium': 'X',
    'Large': 's'
}

# List to hold all segment data for combined analysis
all_segments_data = []

# Process each segment
for segment_name, segment_data in segments_new.items():
    if segment_data.empty:
        print(f"No new data for segment {segment_name}")
        continue

    # Define features based on the segment
    if segment_name == 'Less_than_18':
        features = ['Interaction', 'AverageSize']
    else:
        features = ['Interaction', 'AverageSize', 'AverageDeviation']

    X_new = segment_data[features]
    y_true = segment_data['True_PlaneSizeRange']

    # Load the corresponding model
    model_filename = f'random_forest_{segment_name}.pkl'
    model = joblib.load(model_filename)

    if not isinstance(model, RandomForestClassifier):
        print(f"Error: Loaded model for segment {segment_name} is not a valid model. It is of type {type(model)}.")
        continue

    # Predict using the model
    y_pred_new = model.predict(X_new)
    segment_data = segment_data.copy()  # Ensure we work on a copy to avoid SettingWithCopyWarning

    # Use .loc to ensure setting values properly
    segment_data.loc[:, 'Predicted_PlaneSizeRange'] = y_pred_new
    segment_data.loc[:, 'Predicted'] = segment_data['Predicted_PlaneSizeRange'].apply(lambda x: f'Predicted: {x}')

    # Add to combined data for all segments
    all_segments_data.append(segment_data)

    # Calculate and print classification report and confusion matrix
    report = classification_report(y_true, y_pred_new, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_new, labels=labels)

    print(f"\nClassification Report for {segment_name}:\n{report}")
    print(f"Confusion Matrix for {segment_name}:\n{cm}")

    # Save the comparison and the confusion matrix
    comparison_csv_filename = generate_filename(f'{output_dir}/predictions_vs_actual_{segment_name}', 'csv')
    segment_data.to_csv(comparison_csv_filename, index=False)

    # Plot and save the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {segment_name}')
    cm_filename = generate_filename(f'{output_dir}/confusion_matrix_{segment_name}', 'png')
    plt.savefig(cm_filename)
    plt.close()  # Close the plot to avoid backend errors

    # Scatter plots with vibrant colors and trend lines
    sns.set(style="whitegrid")

    # Plot with Compaction Index and Average Size
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=segment_data,
        x='CompactionIndex',
        y='AverageSize',
        hue='Category',
        style='True_PlaneSizeRange',
        markers=markers,
        palette=colors,
        size=segment_data['Predicted_PlaneSizeRange'] == segment_data['True_PlaneSizeRange'],
        sizes={True: 300, False: 100}  # Larger size for correct matches, smaller for mismatches
    )
    sns.regplot(data=segment_data, x='CompactionIndex', y='AverageSize', color='black', scatter=False)
    plt.title(f'Compaction Index vs Average Size: Actual vs Predicted for {segment_name}')
    plt.legend(loc='best')
    plt.savefig(generate_filename(f'{output_dir}/compaction_scatter_{segment_name}_new_data', 'png'))
    plt.close()

    # Plot with Interaction and Compaction Index
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=segment_data,
        x='CompactionIndex',
        y='Interaction',
        hue='Category',
        style='True_PlaneSizeRange',
        markers=markers,
        palette=colors,
        size=segment_data['Predicted_PlaneSizeRange'] == segment_data['True_PlaneSizeRange'],
        sizes={True: 300, False: 100}  # Larger size for correct matches, smaller for mismatches
    )
    sns.regplot(data=segment_data, x='CompactionIndex', y='Interaction', color='black', scatter=False)
    plt.title(f'Compaction Index vs Interaction: Actual vs Predicted for {segment_name}')
    plt.legend(loc='best')
    plt.savefig(generate_filename(f'{output_dir}/ci_interaction_scatter_{segment_name}_new_data', 'png'))
    plt.close()

    # Original Plot with Interaction and Average Size
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=segment_data,
        x='Interaction',
        y='AverageSize',
        hue='Category',
        style='True_PlaneSizeRange',
        markers=markers,
        palette=colors,
        size=segment_data['Predicted_PlaneSizeRange'] == segment_data['True_PlaneSizeRange'],
        sizes={True: 300, False: 100}  # Larger size for correct matches, smaller for mismatches
    )
    sns.regplot(data=segment_data, x='Interaction', y='AverageSize', color='black', scatter=False)
    plt.title(f'Interaction vs Average Size: Actual vs Predicted for {segment_name}')
    plt.legend(loc='best')
    plt.savefig(generate_filename(f'{output_dir}/interaction_scatter_{segment_name}_new_data', 'png'))
    plt.close()

# Combine all segments data for the final plot
combined_data = pd.concat(all_segments_data, ignore_index=True)

# Plot real plane area vs predicted plane size range for all segments
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=combined_data,
    x='PlaneSize',
    y='Predicted_PlaneSizeRange',
    hue='Category',
    style='True_PlaneSizeRange',
    markers=markers,
    palette=colors,
    size=combined_data['Predicted_PlaneSizeRange'] == combined_data['True_PlaneSizeRange'],
    sizes={True: 300, False: 100}  # Larger size for correct matches, smaller for mismatches
)

# Adding vertical lines to indicate the boundaries between Small, Medium, and Large
plt.axvline(x=1600, color='gray', linestyle='--', label='Small/Medium Boundary')
plt.axvline(x=2400, color='gray', linestyle='--', label='Medium/Large Boundary')

plt.title('Real Plane Area vs Predicted Plane Size Range for All Segments')
plt.xlabel('Plane Size')
plt.ylabel('Predicted Plane Size Range')
plt.legend(loc='best')
plt.savefig(generate_filename(f'{output_dir}/plane_size_vs_predicted_all_segments', 'png'))
plt.close()