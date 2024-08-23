import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

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

# Train and save models for each segment
for segment_name, segment_data in segments.items():
    if segment_data.empty:
        print(f"No data for segment {segment_name}")
        continue

    # Define features and target based on the segment
    if segment_name == 'Less_than_18':
        # Use Interaction and AverageSize for the Less_than_18 segment
        features = ['Interaction', 'AverageSize']
    else:
        # Use Interaction, AverageSize, and AverageDeviation for the Control_18_20 and More_than_20 segments
        features = ['Interaction', 'AverageSize', 'AverageDeviation']

    target = 'PlaneSizeRange'
    X = segment_data[features]
    y = segment_data[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    model_filename = f'random_forest_{segment_name}.pkl'
    joblib.dump(model, model_filename)

    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Results for {segment_name}:")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # Plot and save the confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {segment_name}')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Add labels inside the matrix
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{segment_name}.png')
    plt.show()
