import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Load the IoT energy dataset
# ----------------------------------------------------------------------
def load_data(excel_path: str = "Energy consumption.xlsx") -> pd.DataFrame:
    """
    Load the energy consumption dataset from an Excel file.

    Parameters
    ----------
    excel_path : str
        Path to the Excel file (default: "Energy consumption.xlsx").

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the columns:
        - Datetime
        - Current
        - Load
        - Energy Consumption
    """
    df = pd.read_excel(excel_path)
    # Basic sanity check
    required_cols = {"Current", "Load", "Energy Consumption"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Input file must contain columns: {required_cols}, "
            f"but got: {df.columns.tolist()}"
        )
    return df


# ----------------------------------------------------------------------
# 2. Fit multiple linear regression model
# ----------------------------------------------------------------------
def fit_regression(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit OLS regression model: Energy Consumption ~ Current + Load.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing "Current", "Load", and "Energy Consumption".

    Returns
    -------
    model : statsmodels RegressionResultsWrapper
        Fitted OLS model.
    """
    X = df[["Current", "Load"]]
    y = df["Energy Consumption"]

    # Add intercept term
    X_ols = sm.add_constant(X)

    # Fit OLS model
    model = sm.OLS(y, X_ols).fit()
    return model


# ----------------------------------------------------------------------
# 3. Prediction for a given operating point
# ----------------------------------------------------------------------
def predict_energy(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    current: float,
    load: float,
) -> float:
    """
    Predict energy consumption for a given Current and Load.

    Parameters
    ----------
    model : RegressionResultsWrapper
        Fitted statsmodels OLS model.
    current : float
        Current value (A).
    load : float
        Load value.

    Returns
    -------
    float
        Predicted energy consumption.
    """
    x_new = pd.DataFrame(
        {
            "const": [1.0],
            "Current": [current],
            "Load": [load],
        }
    )
    y_pred = model.predict(x_new)[0]
    return float(y_pred)


# ----------------------------------------------------------------------
# 4. Visualization functions
# ----------------------------------------------------------------------
def plot_actual_vs_predicted(
    df: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    output_path: str = "fig_energy_actual_vs_pred.png",
) -> None:
    """
    Plot actual vs. predicted energy consumption.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with "Energy Consumption".
    model : RegressionResultsWrapper
        Fitted model.
    output_path : str
        File name to save the figure.
    """
    y = df["Energy Consumption"]
    X = df[["Current", "Load"]]
    X_ols = sm.add_constant(X)
    y_pred = model.predict(X_ols)

    plt.figure()
    plt.scatter(y, y_pred, label="Predicted vs Actual")

    # 45-degree reference line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle="--", label="Perfect fit line")

    plt.xlabel("Actual Energy Consumption")
    plt.ylabel("Predicted Energy Consumption")
    plt.title("Actual vs Predicted Energy Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_current_vs_energy(
    df: pd.DataFrame,
    output_path: str = "fig_current_vs_energy.png",
) -> None:
    """
    Plot Current vs. Energy Consumption with a simple linear fit.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with "Current" and "Energy Consumption".
    output_path : str
        File name to save the figure.
    """
    x = df["Current"]
    y = df["Energy Consumption"]

    plt.figure()
    plt.scatter(x, y, label="Observed points")

    # Simple linear fit for visualization (Current only)
    coef_current = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = np.polyval(coef_current, x_line)
    plt.plot(x_line, y_line, linestyle="--", label="Linear fit (Current only)")

    plt.xlabel("Current (A)")
    plt.ylabel("Energy Consumption")
    plt.title("Current vs Energy Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_load_vs_energy(
    df: pd.DataFrame,
    output_path: str = "fig_load_vs_energy.png",
) -> None:
    """
    Plot Load vs. Energy Consumption with a simple linear fit.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with "Load" and "Energy Consumption".
    output_path : str
        File name to save the figure.
    """
    x = df["Load"]
    y = df["Energy Consumption"]

    plt.figure()
    plt.scatter(x, y, label="Observed points")

    # Simple linear fit for visualization (Load only)
    coef_load = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = np.polyval(coef_load, x_line)
    plt.plot(x_line, y_line, linestyle="--", label="Linear fit (Load only)")

    plt.xlabel("Load")
    plt.ylabel("Energy Consumption")
    plt.title("Load vs Energy Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ----------------------------------------------------------------------
# 5. Main entry point
# ----------------------------------------------------------------------
def main():
    # 1) Load data
    df = load_data("Energy consumption.xlsx")

    # 2) Fit regression model
    model = fit_regression(df)

    # 3) Print regression summary
    print(model.summary())

    # 4) Prediction for the given operating point
    current_value = 906.53
    load_value = 533.55
    pred_energy = predict_energy(model, current_value, load_value)
    print(f"\nPredicted energy consumption for "
          f"Current = {current_value}, Load = {load_value}: "
          f"{pred_energy:.4f}")

    # 5) Generate figures
    plot_actual_vs_predicted(df, model,
                             output_path="fig_energy_actual_vs_pred.png")
    plot_current_vs_energy(df,
                           output_path="fig_current_vs_energy.png")
    plot_load_vs_energy(df,
                        output_path="fig_load_vs_energy.png")

    print("\nFigures saved as:")
    print("  - fig_energy_actual_vs_pred.png")
    print("  - fig_current_vs_energy.png")
    print("  - fig_load_vs_energy.png")


if __name__ == "__main__":
    main()
