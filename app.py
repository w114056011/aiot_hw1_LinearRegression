import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Streamlit Sidebar for User Inputs ---
st.sidebar.header('Linear Regression Parameters')

# Sliders for user to modify parameters
num_points = st.sidebar.slider('Number of Points', min_value=100, max_value=1000, value=500, step=10)
a = st.sidebar.slider('Slope (a)', min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
variance = st.sidebar.slider('Noise Variance', min_value=0, max_value=1000, value=100, step=10)
b = 10 # Intercept

# --- Data Generation ---
def generate_data(a, b, variance, num_points):
    """
    Generates synthetic data based on user-defined parameters.
    """
    np.random.seed(42)
    X = np.random.rand(num_points, 1) * 100
    noise_values = np.random.normal(0, np.sqrt(variance), num_points)
    y = a * X.flatten() + b + noise_values
    return pd.DataFrame({'X': X.flatten(), 'y': y})

data = generate_data(a, b, variance, num_points)

# --- Main App ---
st.set_page_config(page_title="Interactive Linear Regression Visualizer", layout="wide")

st.title('Interactive Linear Regression')

st.header('Generated Data')
st.write(data.head())

# --- Data Preparation ---
X = data[['X']]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Model Evaluation ---
y_pred = model.predict(X_test)

st.header('Model Evaluation')
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, y_pred):.2f}")
col2.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

# --- Outlier Display ---
st.header('Top 5 Outliers')
# Ensure y_pred is a flat array for subtraction
y_pred_flat = y_pred.flatten()
# Align y_test and y_pred_flat by index
y_test_aligned, y_pred_aligned = y_test.align(pd.Series(y_pred_flat, index=y_test.index), join='inner')
residuals = np.abs(y_test_aligned - y_pred_aligned)
outlier_indices = residuals.nlargest(5).index
st.write("Data points from the test set with the largest prediction error:")
st.write(data.loc[outlier_indices])

# --- Visualization ---
st.header('Regression Plot')
fig, ax = plt.subplots()
ax.scatter(data['X'], data['y'], label='Actual Data')
# Ensure X_test and y_pred have compatible shapes for plotting
X_test_sorted = X_test.sort_index()
y_pred_sorted = pd.Series(y_pred.flatten(), index=X_test.index).sort_index()
ax.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Line')
# Highlight outliers
ax.scatter(data.loc[outlier_indices]['X'], data.loc[outlier_indices]['y'], color='orange', s=100, label='Top 5 Outliers', edgecolors='black')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Linear Regression')
ax.legend()
st.pyplot(fig)