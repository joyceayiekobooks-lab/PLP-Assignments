# -----------------------------------------
# Data Analysis and Visualization Assignment
# Using Pandas, Matplotlib, and Seaborn
# -----------------------------------------

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ---------------------------------------------------
# Task 1: Load and Explore the Dataset
# ---------------------------------------------------

try:
    # Load the Iris dataset from sklearn
    iris_data = load_iris()

    # Convert to a pandas DataFrame
    df = pd.DataFrame(
        data=iris_data.data,
        columns=iris_data.feature_names
    )
    df['species'] = [iris_data.target_names[i] for i in iris_data.target]

    # Display first few rows
    print("First five rows of the dataset:")
    print(df.head())

    # Check dataset information
    print("\nDataset info:")
    print(df.info())

    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Clean the dataset (if any missing values)
    df = df.dropna()

except FileNotFoundError:
    print("Error: The dataset file was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# ---------------------------------------------------
# Task 2: Basic Data Analysis
# ---------------------------------------------------

# Statistical summary
print("\nStatistical summary of numerical columns:")
print(df.describe())

# Group by species and compute mean of numerical columns
species_means = df.groupby('species').mean(numeric_only=True)
print("\nMean values for each species:")
print(species_means)

# Identify simple patterns
print("\nObservations:")
print("1. Iris-virginica generally has the highest petal and sepal dimensions.")
print("2. Iris-setosa tends to have the smallest petal measurements.")
print("3. There are clear differences among the species that can be visualized.")

# ---------------------------------------------------
# Task 3: Data Visualization
# ---------------------------------------------------

# Set a consistent visual style
sns.set(style="whitegrid")

# 1. Line Chart (simulating trend)
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title("Line Chart: Sepal Length Trend Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart (average petal length per species)
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df, estimator='mean')
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot (sepal length vs. petal length)
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x='sepal length (cm)',
    y='petal length (cm)',
    hue='species',
    data=df,
    palette='Set2'
)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()

# ---------------------------------------------------
# Findings and Conclusion
# ---------------------------------------------------

print("\nFindings:")
print("- Iris-setosa has smaller petals and sepals compared to the other species.")
print("- Iris-virginica is the largest in both sepal and petal dimensions.")
print("- The scatter plot clearly shows clustering by species, indicating potential for classification.")
