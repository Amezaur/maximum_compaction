import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
import lime
import lime.lime_tabular

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
    'X': 'first',  # Just to keep one of the X coordinates
    'Y': 'first',  # Just to keep one of the Y coordinates
    'NumberOfNiches': 'first',  # Keep the number of niches
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

# Model definitions
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'KNeighbors': KNeighborsClassifier()
}

# Hyperparameter grid for Gradient Boosting to avoid overfitting
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Create output directory
output_dir = 'model_outputs'
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

    # Apply SMOTE to balance the dataset
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, len(y_train) - 1))
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    except ValueError as e:
        print(f"SMOTE error for segment {segment_name}: {e}")
        X_train_smote, y_train_smote = X_train, y_train

    # Ensure X_test has valid feature names
    X_test = pd.DataFrame(X_test, columns=features)

    # K-Fold Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        print(f"Training {model_name} for {segment_name}")
        if model_name == 'GradientBoosting':
            grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring='accuracy')
            grid_search.fit(X_train_smote, y_train_smote)
            best_model = grid_search.best_estimator_
        else:
            # Apply cross-validation
            cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=kfold, scoring='accuracy')
            model.fit(X_train_smote, y_train_smote)
            best_model = model
            print(f"Cross-validation scores for {model_name} in {segment_name}: {cv_scores}")

        # Evaluate model
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = np.mean(y_test == y_pred)

        results[f'{segment_name}_{model_name}'] = {
            'model': best_model,
            'report': report,
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'cv_scores': cv_scores
        }

        # Print results
        print(f"Results for {segment_name} using {model_name}:")
        print(f"  Classification Report: {report}")
        print(f"  Confusion Matrix: {cm}")
        print(f"  Accuracy: {accuracy}")
        print(f"  Cross-Validation Scores: {cv_scores}")
        print(f"  Mean CV Accuracy: {np.mean(cv_scores)}")

        # Save classification report
        report_path = os.path.join(output_dir, f'classification_report_{segment_name}_{model_name}.txt')
        with open(report_path, 'w') as f:
            f.write(classification_report(y_test, y_pred, zero_division=1))

        # Save confusion matrix
        cm_path = os.path.join(output_dir, f'confusion_matrix_{segment_name}_{model_name}.png')
        plt.figure()
        plt.matshow(cm, cmap='coolwarm', alpha=0.8)
        plt.title(f'Confusion Matrix for {segment_name} using {model_name}')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(cm_path)
        plt.close()

        # SHAP analysis for tree-based models if binary classification
        if model_name in ['RandomForest', 'GradientBoosting'] and len(y_train.unique()) == 2:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_train_smote)

            # Handle SHAP for binary classification
            shap.summary_plot(shap_values, X_train_smote, plot_type="bar", show=False)
            shap_summary_path = os.path.join(output_dir, f'shap_summary_{segment_name}_{model_name}.png')
            plt.title(f'SHAP Summary for {segment_name} using {model_name}')
            plt.savefig(shap_summary_path)
            plt.close()

            # SHAP Interaction Values
            shap_interaction_values = explainer.shap_interaction_values(X_train_smote)

            # Check if the SHAP interaction values are correctly shaped
            if len(shap_interaction_values) > 0 and shap_interaction_values[0].shape == (X_train_smote.shape[0], X_train_smote.shape[1], X_train_smote.shape[1]):
                # SHAP Interaction Plot for Top Features
                for i in range(len(features)):
                    for j in range(i + 1, len(features)):
                        try:
                            shap.dependence_plot((i, j), shap_interaction_values, X_train_smote, show=False)
                            shap_interaction_path = os.path.join(output_dir, f'shap_interaction_{segment_name}_{model_name}_{features[i]}_{features[j]}.png')
                            plt.title(f'SHAP Interaction: {features[i]} and {features[j]} for {segment_name} using {model_name}')
                            plt.savefig(shap_interaction_path)
                            plt.close()
                        except IndexError as e:
                            print(f"Error plotting SHAP dependence for features {features[i]} and {features[j]} in segment {segment_name}: {e}")
                        except ValueError as e:
                            print(f"ValueError plotting SHAP dependence for features {features[i]} and {features[j]} in segment {segment_name}: {e}")

        # LIME analysis
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train_smote.values, feature_names=features, class_names=list(y_train_smote.unique()), discretize_continuous=True)
        lime_exp = lime_explainer.explain_instance(X_test.values[0], best_model.predict_proba, num_features=len(features))
        lime_path = os.path.join(output_dir, f'lime_{segment_name}_{model_name}.html')
        lime_exp.save_to_file(lime_path)

# Plot accuracies for models
segment_names = list(results.keys())
accuracies = [results[segment]['accuracy'] for segment in segment_names]
f1_scores = [results[segment]['report']['weighted avg']['f1-score'] for segment in segment_names]

plt.figure(figsize=(14, 8))
plt.bar(segment_names, accuracies, color='blue', alpha=0.6, label='Accuracy')
plt.bar(segment_names, f1_scores, color='red', alpha=0.6, label='F1 Score', bottom=accuracies)
plt.xlabel('Segment and Model')
plt.ylabel('Score')
plt.title('Model Performance for Each Segment and Model')
plt.legend()
plt.xticks(rotation=90)
model_performance_path = os.path.join(output_dir, 'model_performance.png')
plt.savefig(model_performance_path)
plt.close()

