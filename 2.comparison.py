import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Load real data
real_data_path = 'C:/Users/Zamfi/Dropbox/PC (2)/Desktop/screen_coordinates.xlsx'
real_data = pd.read_excel(real_data_path)

# Load theoretical model data
theoretical_model_path = 'C:/Users/Zamfi/Dropbox/PC (2)/Desktop/Model.csv'
theoretical_data = pd.read_csv(theoretical_model_path)

# Use provided area values directly
real_data['Niche_Area'] = real_data['AverageSize']
real_data['Plane_Area'] = real_data['PlaneSize']

# Calculate the total niche area and plane area for each group in each category
real_data_grouped = real_data.groupby(['Category', 'Group']).agg(
    total_niche_area=('Niche_Area', 'sum'),
    total_plane_area=('Plane_Area', 'mean')
).reset_index()

# Calculate the real packing efficiency
real_data_grouped['Real_Packing_Efficiency'] = real_data_grouped['total_niche_area'] / real_data_grouped['total_plane_area']

# Prepare comparison DataFrame
comparison = []

# For each unique category and group, compare real packing efficiency with theoretical density
for (category, group), group_data in real_data_grouped.groupby(['Category', 'Group']):
    num_niches = len(real_data[(real_data['Category'] == category) & (real_data['Group'] == group)])
    real_efficiency = group_data['Real_Packing_Efficiency'].values[0]
    theoretical_density = theoretical_data[theoretical_data['N'] == num_niches]['density'].values[0] if num_niches in theoretical_data['N'].values else np.nan
    comparison.append({
        'Genotype': category,
        'Niches': num_niches,
        'Real_Packing_Efficiency': real_efficiency,
        'Theoretical_Density': theoretical_density
    })

comparison_df = pd.DataFrame(comparison)

# Save the comparison dataframe to CSV
comparison_csv_path = 'C:/Users/Zamfi/Dropbox/PC (2)/Desktop/Packing_Efficiency_Comparison.csv'
comparison_df.to_csv(comparison_csv_path, index=False)

# Display the comparison dataframe
print(comparison_df)

# Normalize only the Theoretical_Density
scaler = StandardScaler()
comparison_df['Theoretical_Density'] = scaler.fit_transform(comparison_df[['Theoretical_Density']])

# Use polynomial regression
X = comparison_df['Theoretical_Density'].values.reshape(-1, 1)
y = comparison_df['Real_Packing_Efficiency'].values

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Predict and calculate metrics
y_pred = model.predict(X_poly)
mse = mean_squared_error(y, y_pred)
r_squared = r2_score(y, y_pred)

# Cross-validation
cv_scores = cross_val_score(model, X_poly, y, cv=5)

# Plot comparison with color coding for real data
colors = {
    'control': 'green',
    'mutant_arm': 'purple',
    'mutant_hpo': 'blue',
    'mutant_inr': 'hotpink',
    'mutant_n': 'turquoise'
}

fig, ax = plt.subplots(2, 2, figsize=(18, 12))

# Plot real vs theoretical efficiencies
for genotype in comparison_df['Genotype'].unique():
    genotype_data = comparison_df[comparison_df['Genotype'] == genotype]
    ax[0, 0].scatter(genotype_data['Theoretical_Density'], genotype_data['Real_Packing_Efficiency'], label=f'Real {genotype}', color=colors[genotype])

# Plot the polynomial regression line
X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_fit_poly = poly.transform(X_fit)
y_fit = model.predict(X_fit_poly)
ax[0, 0].plot(X_fit, y_fit, color='red', linestyle='--')

ax[0, 0].set_xlabel('Theoretical Density')
ax[0, 0].set_ylabel('Real Packing Efficiency')
ax[0, 0].set_ylim(0, 1)  # Set y-axis limits to [0, 1]
ax[0, 0].set_title(f'Comparison of Real Packing Efficiency and Theoretical Density\nMSE: {mse:.2f}, R-squared: {r_squared:.2f}\nCross-Validation Scores: {cv_scores}')
ax[0, 0].legend()
ax[0, 0].grid(True)

# Plot the real and theoretical packing efficiencies
for genotype in comparison_df['Genotype'].unique():
    genotype_data = comparison_df[comparison_df['Genotype'] == genotype]
    ax[0, 1].scatter(genotype_data['Niches'], genotype_data['Real_Packing_Efficiency'], label=f'Real {genotype}', color=colors[genotype])

# Plot the theoretical model as single data points
ax[0, 1].scatter(theoretical_data['N'], theoretical_data['density'], label='Theoretical Model', color='red', alpha=0.5, marker='o')

ax[0, 1].set_xlim(0, 40)
ax[0, 1].set_xlabel('Number of Niches')
ax[0, 1].set_ylabel('Packing Efficiency')
ax[0, 1].set_ylim(0, 1)  # Set y-axis limits to [0, 1]
ax[0, 1].set_title('Real vs Theoretical Packing Efficiency')
ax[0, 1].legend()
ax[0, 1].grid(True)

# Box plot to show the distribution of real packing efficiencies
sns.boxplot(x='Genotype', y='Real_Packing_Efficiency', data=comparison_df, ax=ax[1, 0], palette=colors)
ax[1, 0].set_ylim(0, 1)  # Set y-axis limits to [0, 1]
ax[1, 0].set_title('Distribution of Real Packing Efficiencies by Genotype')
ax[1, 0].grid(True)

# Histogram to show the distribution of real packing efficiencies
for genotype in comparison_df['Genotype'].unique():
    genotype_data = comparison_df[comparison_df['Genotype'] == genotype]
    ax[1, 1].hist(genotype_data['Real_Packing_Efficiency'], bins=10, alpha=0.5, label=f'{genotype}', color=colors[genotype])

ax[1, 1].set_xlabel('Real Packing Efficiency')
ax[1, 1].set_ylabel('Frequency')
ax[1, 1].set_xlim(0, 1)  # Set x-axis limits to [0, 1]
ax[1, 1].set_title('Histogram of Real Packing Efficiencies by Genotype')
ax[1, 1].legend()
ax[1, 1].grid(True)

plt.tight_layout(pad=3.0)
plt.show()
