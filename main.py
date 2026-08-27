import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. Load Dataset ----------
def load_dataset(file_path: str):
    df = pd.read_csv(file_path)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# ---------- 2. Feature Selection ----------
def select_features(df):
    df_numeric = df.select_dtypes(include=[np.number])
    
    X = df_numeric.drop(columns=["mpg"]).to_numpy()
    Y = df_numeric["mpg"].to_numpy()
    
    return X, Y

# ---------- 3. Polynomial Feature Expansion ----------
def apply_polynomial_features(X, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_full = poly.fit_transform(X)
    
    n_original_features = X.shape[1]
    
    if np.allclose(X_poly_full[:, :n_original_features], X):
        X_poly = X_poly_full[:, n_original_features:]
    else:
      X_poly = X_poly_full

    return X_poly


# ---------- 4. Train/Test Split ----------
def split_data(X, Y, test_size=0.2, random_state=42):
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

# ---------- 5. Train Linear Regression ----------
def train_linear_model(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

# ---------- 6. Train Polynomial Regression ----------
def train_polynomial_model(X_train, Y_train, degree=2):
    X_poly = apply_polynomial_features(X_train, degree)
    model = LinearRegression()
    model.fit(X_poly, Y_train)
    return model

# ---------- 7. Prediction ----------
def predict_linear(model, X_test):
    return model.predict(X_test)

def predict_polynomial(model, X_test, degree=2):
    X_poly = apply_polynomial_features(X_test, degree)
    return model.predict(X_poly)

# ---------- 8. Evaluation ----------
def evaluate_model(Y_test, Y_pred, model_name):
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    print(model_name)
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)
    print()
    
    return mse, r2

# ---------- 9. Plot Results ----------
def plot_results(Y_test, Y_pred_linear, Y_pred_poly, save_path):
    plt.figure()
    
    plt.scatter(Y_test, Y_pred_linear, label="Linear Regression", alpha=0.6)
    plt.scatter(Y_test, Y_pred_poly, label="Polynomial Regression", alpha=0.6)
    
    plt.plot(
        [Y_test.min(), Y_test.max()],
        [Y_test.min(), Y_test.max()],
        label="Ideal Fit"
    )
    
    plt.xlabel("Actual MPG")
    plt.ylabel("Predicted MPG")
    plt.title("Linear vs Polynomial Regression Results")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# ---------- MAIN LOGIC ----------
def main():
    dataset_path = os.path.join(BASE_DIR, "dataset.csv")
    plot_path = os.path.join(BASE_DIR, "plot.png")
    
    # Check if dataset exists before proceeding (prevents errors if path is wrong)
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    df = load_dataset(dataset_path)
    X, Y = select_features(df)
    
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    
    # Linear Regression
    linear_model = train_linear_model(X_train, Y_train)
    Y_pred_linear = predict_linear(linear_model, X_test)
    evaluate_model(Y_test, Y_pred_linear, "Linear Regression")
    
    # Polynomial Regression
    poly_model = train_polynomial_model(X_train, Y_train, degree=2)
    Y_pred_poly = predict_polynomial(poly_model, X_test, degree=2)
    evaluate_model(Y_test, Y_pred_poly, "Polynomial Regression")
    
    plot_results(Y_test, Y_pred_linear, Y_pred_poly, plot_path)
    print(f"Plot saved to: {plot_path}")

main()

__all__ = ["LinearRegression]

           






































