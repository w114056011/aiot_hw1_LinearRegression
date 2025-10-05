# Interactive Linear Regression Explorer

This project is a Streamlit web application that allows users to interactively explore the concepts of linear regression. You can dynamically adjust the parameters of a synthetic dataset and see how the regression model adapts in real-time.

## Features

- **Interactive Parameter Tuning:** Adjust the following parameters using sliders in the sidebar:
  - **Slope (a):** Control the slope of the linear relationship.
  - **Noise Variance:** Modify the variance of the random noise added to the data.
  - **Number of Points:** Change the size of the dataset.
- **Real-time Visualization:** The application instantly updates a scatter plot of the data points and the corresponding regression line as you change the parameters.
- **Model Evaluation:** Key performance metrics for the regression model, such as Mean Squared Error (MSE) and R² Score, are calculated and displayed.
- **Outlier Detection:** The top 5 data points with the largest prediction errors (outliers) are identified, listed, and highlighted on the plot.

## Project Structure

```
C:\Users\wesle\OneDrive\桌面\nchu\1141\AIOT\hw1
├── venv/
├── app.py
├── devLog.md
├── linear_regression_crisp_dm.py
├── project_plan.md
└── Todo.md
```

- **`app.py`**: The main Python script containing the Streamlit application.
- **`venv/`**: The Python virtual environment directory.
- **`devLog.md`**: A development log tracking the conversation and steps taken to build the project.
- **`project_plan.md`**: The detailed plan for implementing the project.
- **`Todo.md`**: A checklist of the design steps based on the CRISP-DM framework.
- **`linear_regression_crisp_dm.py`**: The initial script for the linear regression model.

## Setup and Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the environment:**
    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

3.  **Install the required packages:**
    ```bash
    pip install streamlit pandas numpy scikit-learn matplotlib
    ```

## How to Run

Once the setup is complete, run the following command in your terminal:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

## Code Description

The `app.py` script is structured as follows:

1.  **Sidebar for User Inputs:** The Streamlit sidebar (`st.sidebar`) is used to create sliders for the user-adjustable parameters (`a`, `variance`, `num_points`).
2.  **Data Generation:** A function `generate_data` creates a synthetic dataset based on the user's input. The data follows the linear equation `y = ax + b` with added random noise from a normal distribution `N(0, var)`.
3.  **Model Training:** A `LinearRegression` model from `scikit-learn` is trained on a portion of the generated data.
4.  **Model Evaluation:** The trained model is evaluated on the test set, and the MSE and R² score are displayed using `st.metric`.
5.  **Outlier Display:** The absolute difference between the actual and predicted values (residuals) is calculated to find the top 5 outliers, which are then displayed in a table.
6.  **Visualization:** A scatter plot of the data, the regression line, and the highlighted outliers is rendered using `matplotlib` and displayed with `st.pyplot`.

## CRISP-DM Process

This project loosely follows the Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology. The initial planning and design steps, as documented in `Todo.md` and `project_plan.md`, cover the phases from Business Understanding to Deployment planning.

```