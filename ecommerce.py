import pandas as pn
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# structured_dt = pn.DataFrame({
#     "ID": [1, 2, 3, 4, 5],
#     "Name": ["Game", "Chair", "Rocker", "Maggie", "Bonus"],
#     "Price": [111.02, 122.3, 34.23, 56.0, 90.9]
# })

# structured_dt.to_csv("structured_dt.csv",index=False)
# # structured_dt.to_excel("structured_dt.excel",index=False)
# print(f" {structured_dt} \n")

# # Option 1: Simple Histogram of Prices
# plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
# sns.histplot(x="Price", data=structured_dt, kde=True)  # kde for density curve
# # Annotate each bar with the corresponding product name (approximate)
# for i, row in structured_dt.iterrows():
#     plt.text(row['Price'], 0.5, row['Name'], ha='center', va='bottom')  # Adjust vertical offset as needed

# plt.xlabel("Price")
# plt.ylabel("Frequency")
# plt.title("Distribution of Prices")
# plt.show()

# Scenario 1
#E-Commerce 
#Distribute Structure and Usnstructured Data

# structured_dt = pn.DataFrame({
#     "ID": [1, 2, 3, 4],
    
#     "Customer Name":["Hamza","Shugufta","Tania","Asfaq"],

#     "Email":["h@123.com","e@323.com","d@312.com","f@321.com"],


#     "Purchase History":[100.0,250.0,50.0,10.0],

#     "Transaction Amount":[100,80,45,22]
# })

# print(f"{structured_dt} \n")
# structured_dt.to_csv("structured_dt.csv",index=False)

# product_review = """
# Hamza : lorem ipsum dolor sit amet consectetur adipisicing elit. Quae, voluptatum.
# Shugufta : lorem ipsum dolor sit amet consectetur adipisicing elit. Quae, voluptatum.
# Tania : lorem ipsum dolor sit amet consectetur adipisicing elit. Quae, voluptatum.
# Asfaq : lorem ipsum dolor sit amet consectetur adipisicing elit. Quae, voluptatum.
# """
# with open("unstructured_data.txt", "w") as file:
#     file.write(product_review)

# Scenario 2

# hieght = np.random.normal(loc=170, scale=10, size=50)
# hospital_visits = np.random.randint(0, 10, size=50)
# # print(f"{hieght} \n")
# # print(f"{hospital_visits} \n")

# plt.figure(figsize=(10, 6))
# sns.histplot(x=hieght, bins=10 ,kde=True,color='blue', label='Height Distribution')
# plt.xlabel("Height (cm)")
# plt.ylabel("Frequency")
# plt.title("Height Distribution")
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.histplot(x=hospital_visits, bins=10 ,kde=True,color='blue', label='Height Distribution')
# plt.xlabel("Patient Visits")
# plt.ylabel("Counts")
# plt.title("Patient Visit Distribution")
# plt.show()

# # Scenario 3

# airport_dataset =({
#     "Gender":random.choices(["Female","Male","Other"],k=50),
#     "Class":random.choices(["Economy","Buisness","First Class"],k=50),
#     "Satisfaction":random.choices(["Excellent","Good","Average","Poor","Very Poor"],k=50),
#     "Would Recommend":random.choices(["Yes","No"],k=50),
# })

# df = pn.DataFrame(airport_dataset)
# satisfaction_count = df["Satisfaction"].value_counts()

# plt.figure(figsize=(8, 6))
# sns.countplot(x="Satisfaction", data=df)
# sns.barplot(x=satisfaction_count.index, y=satisfaction_count.values, palette="viridis")
# plt.xlabel("Satisfaction Level")
# plt.ylabel("Count")
# plt.title("Ordinal Data: Airline Passenger Satisfaction")
# plt.show()

# Scenario 4
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# Sample dataset
# car_data = pn.DataFrame({
#     "Engine Size (L)": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
#     "Price ($)": [15000, 20000, 25000, 30000, 35000, 40000],
#     "Sales Count": [500, 450, 400, 350, 300, 250]
# })

# # Features (X) and Target (y)
# X = car_data[["Engine Size (L)", "Price ($)"]]
# y = car_data["Sales Count"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Linear Regression Model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict Sales Count
# predictions = model.predict(X_test)
# print("Predicted Sales Count:", predictions)

# # Plot regression line
# plt.figure(figsize=(6, 4))
# sns.scatterplot(x=car_data["Engine Size (L)"], y=car_data["Sales Count"], color="blue", label="Actual Sales")
# sns.lineplot(x=car_data["Engine Size (L)"], y=model.predict(X), color="red", label="Predicted Sales")
# plt.xlabel("Engine Size (L)")
# plt.ylabel("Sales Count")
# plt.title("Car Sales Prediction")
# plt.legend()
# plt.show()


file_path = "6th class result 25.xlsx"  # Replace with your file path
df = pn.read_excel(file_path)

# Show the first few rows
# Set header manually (e.g., header is in second row)
df = pn.read_excel(file_path, header=2)
df = df[1:]
# print(df.head())
df['Maths'] = pn.to_numeric(df['Maths'], errors='coerce')
bins = [0,49,59,69,79,89,100]
labels = ['F','E','D','C','B','A']
df['Grade'] = pn.cut(df['Maths'], bins = bins ,labels = labels , include_lowest = True)
print(df[['Maths', 'Grade']].head())

# Plot a histogram (bar chart) of Maths grades
df['Grade'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Distribution of Maths Grades")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

# Select features and target
features = ['Urdu', 'English', 'Maths', 'Science', 'Islamiat']
df[features] = df[features].apply(pn.to_numeric, errors='coerce')
target = 'Total'
df[target] = pn.to_numeric(df[target], errors='coerce')

# Drop rows with missing data
df_model = df.dropna(subset=features + [target])

# Split data
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

plt.figure(figsize=(10, 6))
sns.histplot(x="Total", data=df_model, kde=True)
plt.xlabel("Total")
plt.ylabel("Frequency")
plt.title("Distribution of Total Scores")
plt.show()

# plt.figure(figsize=(10, 6))
# sns.barplot(x="Total", y="Grade",data=df_model, color="skyblue")
# plt.xlabel("Total")
# plt.ylabel("Grade")
# plt.title("Distribution of Total Scores")
# plt.show()

# Remove any rows with missing values
df_clean = df.dropna(subset=['Maths', 'Total', 'Grade'])


plt.figure(figsize=(12, 6))
sns.boxplot(x="Grade", y="Total", data=df_clean, order=['A', 'B', 'C', 'D', 'E', 'F'])
plt.xlabel("Grade")
plt.ylabel("Total Score")
plt.title("Distribution of Total Scores by Grade")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary of Grades and Total Scores:")
print(df_clean.groupby('Grade')['Total'].describe())
