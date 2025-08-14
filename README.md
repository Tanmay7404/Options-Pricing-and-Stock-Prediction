# Financial Engineering and Time Series Prediction in Python

📈 This repository contains a collection of Jupyter Notebooks that implement and analyze various models for financial option pricing and stock price prediction. Developed as a project for a course in Computational Finance, this collection serves as a practical guide to the numerical methods used in quantitative finance and machine learning for time series forecasting.

## Key Concepts Covered

This project is divided into two main areas: **Financial Option Pricing** and **Stock Price Prediction**.

### Financial Option Pricing

This section covers the valuation of various options using established numerical methods.

-   **Option Types Explored**:
    -   **European Options**: Standard call and put options.
    -   **American Options**: Options with early exercise features.
    -   **Asian Options**: Path-dependent options based on the average asset price.
    -   **Lookback Options**: Path-dependent options with payoffs tied to the maximum asset price.

-   **Core Pricing Models**:
    -   **Binomial Tree Model**: A discrete-time lattice model for valuing options via backward induction.
    -   **Monte Carlo Simulation**: A stochastic method for pricing complex and path-dependent options by simulating asset paths.

-   **Key Numerical Techniques**:
    -   **Dynamic Programming**: An efficient algorithm for solving computationally intensive problems, applied here to Lookback options.
    -   **Variance Reduction**: Methods to improve the accuracy and efficiency of Monte Carlo simulations, including **Antithetic Variates** and **Control Variates**.

-   **Analysis & Insights**:
    -   **Sensitivity Analysis**: Examining the impact of parameters like volatility and interest rates on option prices.
    -   **Model Convergence**: Demonstrating how the binomial model's accuracy improves as the number of time steps increases.
    -   **Optimal Exercise Strategy**: Identifying the best time to exercise an American option.

### Stock Price Prediction

This section applies machine learning and deep learning models to forecast time series data.

-   **Forecasting Models**:
    -   **Linear Regression**: A simple baseline model to capture the underlying trend.
    -   **Long Short-Term Memory (LSTM) Networks**: A sophisticated recurrent neural network (RNN) designed for sequential data like stock prices.

-   **Key Methodologies & Features**:
    -   **Feature Engineering**: Creation of technical indicators to enhance model accuracy, including:
        -   Simple Moving Average (SMA)
        -   Exponential Moving Average (EMA)
        -   Moving Average Convergence Divergence (MACD)
        -   Relative Strength Index (RSI)
    -   **Data Preprocessing**: Techniques such as log normalization and Min-Max scaling to prepare time series data for neural networks.
    -   **Sequence Generation**: Structuring data into overlapping windows (unrolling) to create input sequences for the LSTM model.

-   **Performance Evaluation**:
    -   **Root Mean Squared Error (RMSE)**: A standard metric to measure the accuracy of the model's price predictions.
    -   **Visualization**: Plotting training and validation loss curves to diagnose model fit and prevent overfitting.

---

## Notebook Descriptions

### Financial Option Pricing

This suite of notebooks covers the valuation of various exotic and standard options using numerical methods.

#### 1. `01_EuropeanOptions.ipynb`

A clear and focused implementation of the Binomial Model for pricing standard **European call and put options**.

-   **Focus**: Demonstrates the convergence of the binomial model toward the theoretical Black-Scholes price as the number of time steps (`M`) increases.

#### 2. `02_AmericanOptions.ipynb`

This notebook extends the binomial model to price **American options**, which have the key feature of early exercise.

-   **Key Feature**: Implements the logic to check for optimal early exercise at each node of the binomial tree.
-   **Analysis**: Includes a full sensitivity analysis and visualization of the price lattice to determine the optimal exercise strategy.

#### 3. `03_AsianOptions.ipynb`

This notebook prices **Asian options**, where the payoff is determined by the average price of the underlying asset.

-   **Method**: Uses **Monte Carlo Simulation** to generate thousands of potential asset price paths under a risk-neutral framework using Geometric Brownian Motion (GBM).
-   **Features**: Includes implementations of **Antithetic Variates** and **Control Variates** to showcase powerful variance reduction techniques that improve simulation accuracy.

---

#### 4. `04_LookbackOptionPricing.ipynb`

This notebook contrasts standard option pricing with the valuation of a highly path-dependent **Lookback option**.

-   **Part 1**: Prices European options using two variations of the binomial model and performs a full sensitivity analysis.
-   **Part 2**: Focuses on pricing a Lookback option, whose payoff depends on the *maximum* asset price achieved. It demonstrates:
    -   **Brute-Force Method**: A naive O(2^M) algorithm that is computationally infeasible for a moderate number of steps.
    -   **Dynamic Programming**: An efficient O(M⁴) algorithm that uses memoization to make the problem solvable.
      

### Stock Price Prediction

This notebook focuses on forecasting future stock prices using both a simple baseline and advanced deep learning models.

#### 5. `05_StockPrediction.ipynb`

This notebook uses historical data for Google (`GOOG`) to predict its future stock price. It builds and evaluates several models of increasing complexity.

-   **Part 1: Simple Linear Regression**: A baseline model is established to predict stock price based solely on time, demonstrating a simple trend-following approach.

-   **Part 2: LSTM Model with Basic Features**:
    -   An **LSTM (Long Short-Term Memory)** network is built to predict future prices.
    -   **Features**: Uses the log-normalized values of the previous day's closing price and trading volume.
    -   **Methodology**: Data is split into training, validation, and test sets. The model's performance is visualized, and its RMSE is calculated.

-   **Part 3: Improved LSTM Model with Technical Indicators**:
    -   A more sophisticated LSTM model is developed by enriching the feature set.
    -   **Feature Engineering**: The following technical indicators are calculated and used as model inputs:
        -   Simple Moving Average (SMA)
        -   Exponential Moving Average (EMA)
        -   Cumulative Moving Average (CMA)
        -   Moving Average Convergence Divergence (MACD)
        -   Relative Strength Index (RSI)
    -   **Outcome**: This demonstrates how feature engineering can significantly improve the predictive power of a time series model. The final model is trained and its predictions are plotted against actual prices.

---

## 🚀 Setup and Installation

To run these notebooks, you will need Python 3 and several libraries. Using a virtual environment is highly recommended.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required libraries:**
    You can install the packages directly via pip:
    ```bash
    pip install numpy pandas matplotlib scipy yfinance statsmodels xgboost tensorflow jupyter
    ```

## 💻 How to Run

1.  **Launch Jupyter Notebook or JupyterLab** from your terminal:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```

2.  **Open a Notebook**: Navigate to one of the `.ipynb` files in the file browser and click to open it.

3.  **Run the Cells**: You can run the cells individually or all at once to see the calculations, analysis, and generated plots.

---

## 📖 Theoretical Background

-   **Binomial Model**: A discrete-time model that constructs a lattice of possible future asset prices. The option price is found via backward induction, discounting the expected value under risk-neutral probabilities.
-   **Monte Carlo Simulation**: A method that uses random sampling to model complex systems. For options, it involves simulating thousands of random asset price paths to find the average discounted payoff.
-   **Long Short-Term Memory (LSTM)**: A type of Recurrent Neural Network (RNN) well-suited for time series data. LSTMs have internal memory mechanisms that allow them to learn and remember patterns over long sequences, making them effective for tasks like stock price prediction.
---

