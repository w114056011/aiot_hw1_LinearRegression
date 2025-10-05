# Project Plan: Implementing Linear Regression with Streamlit

This plan outlines the steps to implement an interactive linear regression application.

## 1. Setup Environment
- **Create a virtual environment:**
  - Run `python -m venv venv`
  - Activate the environment:
    - Windows: `venv\Scripts\activate`
    - macOS/Linux: `source venv/bin/activate`
- **Install necessary Python libraries:**
  - Run `pip install pandas numpy scikit-learn matplotlib streamlit`
- **Create a Python script file:** `app.py` (renaming from `linear_regression_crisp_dm.py` to reflect its new nature as a Streamlit app).

## 2. Code Implementation (in `app.py`)

### 2.1. Import Libraries
- Import all required libraries: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

### 2.2. User Inputs in Streamlit Sidebar
- Create sidebar controls using `st.sidebar` for user to modify parameters:
  - **Slope `a`:** `st.sidebar.slider` for the 'a' in `y = ax + b`.
  - **Noise:** `st.sidebar.slider` to control the amount of random noise.
  - **Number of Points:** `st.sidebar.slider` to set the number of data points.

### 2.3. Data Handling
- **`generate_data(a, noise, num_points)` function:**
    - Generate synthetic data based on the user-defined parameters.
    - Create a pandas DataFrame.
    - Separate features (X) and target (y).
    - Use `train_test_split` to create training and testing sets.

### 2.4. Model Training
- **`train_model()` function:**
    - Instantiate `LinearRegression`.
    - Call the `.fit()` method with the training data.

### 2.5. Model Evaluation & Visualization
- **Main part of the Streamlit app:**
    - Display the generated data in a table using `st.write`.
    - Train the model on the generated data.
    - Make predictions.
    - Display evaluation metrics (MSE, R²) using `st.metric`.
    - Use `matplotlib` to create a scatter plot of the data and the regression line.
    - Display the plot in the Streamlit app using `st.pyplot`.

## 3. Execution and Verification
- Run the Streamlit app from the command line: `streamlit run app.py`.
- Verify the output in the web browser:
    - Interact with the sidebar controls to change `a`, `noise`, and the number of points.
    - Observe the plot and evaluation metrics update in real-time.
- Review the code for clarity and correctness.