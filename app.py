## importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("All libraries installed")

df = pd.read_csv("data/retail_sales.csv")
print(df.shape)
print(df.dtypes)
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())
df= df.dropna()
print("\nDataset Summary:")
df.info()
print("Remove Missing:", df.isnull().sum())


print("==Data Cleaning and type conversion: ")

## Convert Date to detetime format
df['Date'] = pd.to_datetime(df['Date'])

## Extract additional time features

df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['DayOfweek'] = df['Date'].dt.day_name()
df['MonthName'] = df['Date'].dt.month_name()

## Ensure numeric columns are proper 
numeric_columns = ['Sales', 'Quantity', 'Profit']
for col in numeric_columns:
    df[col] =pd.to_numeric(df[col], errors='coerce')

print("Date type after conversion: ")
print(df.dtypes)
print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")


print("==Stataistical Summary==")
print("Descriptive Sataistics for Numerical Columns:")
print(df[['Sales', 'Quantity', 'Profit']].describe())

print("\nRegion-wise Summary: ")
region_summary =df.groupby('Region').agg({
    'Sales': ['sum', 'mean'],
    'Profit': ['sum', 'mean']
}).round(2)
print(region_summary)

##Monthly Sales Analysis
print("==Monthly Sales Analysis==")

monthly_sales =df.groupby(['Month', 'MonthName']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).reset_index()

monthly_sales = monthly_sales.sort_values('Month')
print("Monthly Sales Summary:")
print(monthly_sales)

## Visualization: Monthly sales Trand

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['MonthName'], monthly_sales['Sales'],
         marker='o', linewidth=2, markersize=8, label= 'Total Sales')
plt.plot(monthly_sales['MonthName'], monthly_sales['Profit'],
         marker='s', linewidth=2, markersize=8, label='Total Profit')
plt.title('Monthly Sales and Profit Trend', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Amount($)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

##Regional Performance

print("==Regional performance Analysis==")
regional_performance =df.groupby('Region').agg({
    'Sales': ['sum', 'mean', 'count'],
    'Profit': ['sum', 'mean']
}).round(2)

print("Reginal Performance:")
print(regional_performance)

## Visualization 3 Reginal Sales 

plt.figure(figsize=(10,6))
reginal_data = df.groupby('Region')['Sales'].sum()
plt.pie(reginal_data, labels=reginal_data.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette("pastel"))
plt.title('Sales Distribution by region', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()

## Day of Week Analysis
print("==Day fo week analysis")

day_order = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['DayOfweek'] = pd.Categorical(df['DayOfweek'], categories=day_order, ordered=True)

daily_performance = df.groupby('DayOfweek').agg({
    'Sales': 'mean',
    'Profit': 'mean',
    'Quantity': 'mean'
}).round(2)

print("Average Performance by Day of Week:")
print(daily_performance)

## Visualization
plt.figure(figsize=(10, 6))
plt.plot(daily_performance.index, daily_performance['Sales'],
         marker='o', linewidth=2, markersize=8, label='Average Sales')
plt.plot(daily_performance.index, daily_performance['Profit'],
         marker='s', linewidth=2, markersize=8, label='Average Profit')
plt.title ('Average Sales and profit by Day of Week',fontsize=16, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Amount ($)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

##Correlation Analysis

print("==Correlation analysis=")

numerical_df =df[['Sales', 'Quantity', 'Profit', 'Month']]
correlation_matrix =numerical_df.corr()

print ("Correlation Matrix")
print(correlation_matrix)

##Heatmap 

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
