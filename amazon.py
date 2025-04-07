import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Read the Excel file
file_path = "Amazon 2_Raw.xlsx"
df = pd.read_excel(file_path)
print("Dataset:")
print(df)

# Calculate central tendency measures for numeric columns
# print("\nCentral Tendency Measures:")
# print("-" * 50)

# For each numeric column
numeric_columns = df.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    print(f"\nColumn: {column}")
    print(f"Mean: {df[column].mean():.2f}")
    print(f"Median: {df[column].median():.2f}")
    # Mode can have multiple values
    mode_result = df[column].mode()
    print(f"Mode: {', '.join(map(str, mode_result.values))}")

# Optional: Create a box plot to visualize the distribution
plt.figure(figsize=(10, 6))
sns.histplot(df[numeric_columns], bins=20, alpha=0.7, color='skyblue')
plt.title('Distribution of Numeric Variables')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')

# Calculate skewness and kurtosis
numSales = df['Sales'].skew()
numkSales = df['Sales'].kurtosis()

print(f"Skewness: {numSales:.2f}")
print(f"Kurtosis: {numkSales:.2f}")

# Create visualizations
plt.figure(figsize=(15, 5))

# Histogram with KDE
sns.histplot(df['Sales'], kde=True, bins=30)
plt.title(f'Sales Distribution\nSkewness: {numSales:.2f}, Kurtosis: {numkSales:.2f}')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')
plt.show
plt.tight_layout()
plt.show()


("\nMeasures of Dispersion:")
print(f"Variance: {df[column].var():.2f}")
print(f"Standard Deviation: {df[column].std():.2f}")
q1 = df[column].quantile(0.25)
q3 = df[column].quantile(0.75)
iqr = q3 - q1
print(f"Interquartile Range (IQR): {iqr:.2f}")
print(f"Quartiles: Q1={q1:.2f}, Q3={q3:.2f}")

plt.figure(figsize=(12,8))
# plt.subplot(len(numeric_columns), 1, i)
sns.histplot(data=df, bins=20,x=column, kde=True)
plt.title(f'Distribution of {column}')
plt.show()

