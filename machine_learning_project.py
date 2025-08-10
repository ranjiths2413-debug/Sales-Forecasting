import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""Data Preprocessing"""
# Load dataset
df = pd.read_csv("Project Data set.csv")
print(df)

# Check missing values
finding_null_data = df.isnull().sum()
print(finding_null_data)

total_null_count = df.isnull().sum().sum()
print("Total null count:", total_null_count)

# Fill missing values with 1000
df.fillna(1000, inplace=True)
print(df.head())

total_null_count = df.isnull().sum().sum()
print("Total null count after filling:", total_null_count)

finding_null_data = df.isnull().sum()
print(finding_null_data)

# Drop rows with missing values (separate clean dataset)
data = pd.read_csv("Project Data set.csv")
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
print(data.head())

"""Data Visualization"""
# Bar Chart
plt.bar(data['PROD_GROUP'], data['SAL_COM_NEW'], color="red")
plt.title("Bar Chart: PROD_GROUP vs SAL_COM_NEW")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie Chart
grouped_data = df.groupby("INV_MONTH")["SAL_COM_PER"].sum()
plt.figure(figsize=(8, 8))
plt.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Revenue Distribution by INV_MONTH')
plt.show()

# """Cluster Chart"""
grouped_cluster = df.groupby(['SAL_COM_PER', 'ORDER_TYPE'])['INV_MONTH'].sum().reset_index()

grouped_cluster.plot(kind='bar', figsize=(10, 6))

plt.title("Clustered Column Chart: SAL_COM_PER by ORDER_TYPE and INV_MONTH")
plt.xlabel("SAL_COM_PER")
plt.ylabel("ORDER_TYPE")
plt.legend(title='INV_MONTH')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Line Chart
plt.figure(figsize=(10, 6))
monthly_units = df.groupby("PROD_GROUP")["SAL_COM_PER"].sum().reset_index()
plt.plot(monthly_units["PROD_GROUP"], monthly_units["SAL_COM_PER"], marker='o', linestyle='-', color='green')
plt.title("Line Chart: SAL_COM_PER Over PROD_GROUP")
plt.xlabel("PROD_GROUP")
plt.ylabel("Total SAL_COM_PER")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

"""Prediction"""
prediction_data = {
    "Year": [2024, 2025],
    "UNIT_CD": [18590, 43346]
}
df_pred = pd.DataFrame(prediction_data)
print(df_pred)

x = df_pred[['Year']]
y = df_pred['UNIT_CD']

# Creating subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# First subplot: line plot
axes[0].plot(x, y, color="red", marker=".", alpha=0.4, label="UNIT_CD")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Unit")
axes[0].legend()
axes[0].set_title("UNIT_CD Over Year")

# Second subplot: Boxplot
axes[1].boxplot(y)
axes[1].set_title("Boxplot of UNIT_CD")

plt.tight_layout()
plt.show()

# Train Linear Regression Model
model = LinearRegression()
model.fit(x, y)

# Predict for 2026
predicting_year_df = pd.DataFrame({"Year": [2026]})
output = model.predict(predicting_year_df)
print("Predicted UNIT_CD for 2026:", output[0])

