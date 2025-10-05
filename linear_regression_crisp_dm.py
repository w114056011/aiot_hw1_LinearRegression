# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 15:23:41 2025

@author: wesley
"""

# CRISP-DM in Python: Linear Regression Example

# To run this script, you need to install the following libraries:
# pip install scikit-learn pandas numpy matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Business Understanding
# Goal: Predict house prices based on their size.
# This is a simple linear regression problem where Price is the dependent variable
# and Size is the independent variable.

# 2. Data Understanding & 3. Data Preparation
def load_and_prepare_data():
    """
    Generates sample data, and splits it into training and testing sets.
    """
    # Generate synthetic data for house sizes (in square feet) and prices (in thousands of dollars)
    np.random.seed(0)
    size = np.random.randint(500, 3500, 50)
    price = size * 0.15 + np.random.normal(0, 50, 50)

    # Create a pandas DataFrame
    data = pd.DataFrame({'Size': size, 'Price': price})

    print("---" + " Data Understanding ---")
    print("First 5 rows of the dataset:")
    print(data.head())
    print("\n---" + " Data Preparation ---")
    # Define features (X) and target (y)
    X = data[['Size']]
    y = data['Price']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, data

# 4. Modeling
def train_model(X_train, y_train):
    """
    Trains a linear regression model.
    """
    print("\n---" + " Modeling ---")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained.")
    return model

# 5. Evaluation
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model.
    """
    print("\n---" + " Evaluation ---")
    y_pred = model.predict(X_test)

    # The coefficients
    print(f"Coefficients: {model.coef_[0]:.2f}")
    # The mean squared error
    print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
    # The coefficient of determination (R^2)
    print(f"R^2 score: {r2_score(y_test, y_pred):.2f}")
    
    return y_pred

# Visualization of the results
def visualize_results(X_test, y_test, y_pred, model, full_data):
    """
    Visualizes the data and the regression line.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(full_data['Size'], full_data['Price'], color='blue', label='Actual Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.title('House Price vs. Size')
    plt.xlabel('Size (sq. ft.)')
    plt.ylabel('Price ($ thousands)')
    plt.legend()
    plt.grid(True)
    plt.savefig('linear_regression_plot.png')
    print("\nA plot of the regression has been saved as linear_regression_plot.png")

# 6. Deployment
def deploy_model(model):
    """
    Shows how to use the model for new predictions.
    """
    print("\n---" + " Deployment ---")
    # Predict the price of a new house with a size of 2000 sq. ft.
    new_size = np.array([[2000]])
    predicted_price = model.predict(new_size)
    print(f"Predicted price for a 2000 sq. ft. house: ${predicted_price[0]:.2f}k")

if __name__ == '__main__':
    # Execute the CRISP-DM steps
    X_train, X_test, y_train, y_test, full_data = load_and_prepare_data()
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    visualize_results(X_test, y_test, y_pred, model, full_data)
    deploy_model(model)
