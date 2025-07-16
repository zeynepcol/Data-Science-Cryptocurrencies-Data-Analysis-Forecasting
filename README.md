<h1 align="center"> CRYPTOCURRENCIES FINANCIAL DATA ANALYSIS AND FORECASTING </h1>

This project aims to analyze and forecast the future prices of **10 different cryptocurrencies** by utilizing historical financial data collected from the **Binance API**. A web-based interface is developed using **Python** and **Flask** to visualize and interact with the predictions.

![Image](https://github.com/user-attachments/assets/b5a62fc2-5b3a-4531-85a3-3e2fdfbb2d4b)


## 🔄 Project Workflow

1. **Data Collection**  
   Historical price data for 10 cryptocurrencies were pulled directly from the **Binance API** and saved as `.csv` files.

2. **Data Preprocessing**  
   - Handled missing values carefully to ensure data quality  
   - Calculated and integrated various **technical analysis indicators**  
   - Scaled data using `StandardScaler`  
   - Prepared time-series input windows for the LSTM model (window size: 21 days)  
   - Configured forecasting horizon to predict prices 7 days ahead

3. **Data Analysis**  
   - Explored price trends, volatility, and correlations  
   - Visualized data patterns to support model selection  

4. **Price Forecasting**  
   Applied three regression-based machine learning models to predict future prices:  
   - **Linear Regression**  
   - **XGBoost Regressor**  
   - **LSTM (Long Short-Term Memory)** deep learning model  

   Models were evaluated with metrics including **MAE (Mean Absolute Error)**, **RMSE (Root Mean Square Error)**, and **R² Score**. The LSTM model showed strong performance in capturing temporal patterns in cryptocurrency prices.



## 📊 Dataset Features

Each dataset contains **6 main features**:

| Feature | Description                          |
|---------|------------------------------------|
| Date    | Timestamp of the price record       |
| Open    | Opening price at the start of period|
| High    | Highest price during the period     |
| Low     | Lowest price during the period      |
| Close   | Closing price at the end of period  |
| Volume  | Trading volume in that period       |

Datasets are stored in `.csv` format and enriched with technical indicators during preprocessing.



## 📁 Included Datasets

- `ADA-USD.csv` (Cardano)  
- `BNB-USD.csv` (Binance Coin)  
- `BTC-USD.csv` (Bitcoin)  
- `DOGE-USD.csv` (Dogecoin)  
- `DOT-USD.csv` (Polkadot)  
- `ETH-USD.csv` (Ethereum)  
- `LINK-USD.csv` (Chainlink)  
- `LTC-USD.csv` (Litecoin)  
- `XMR-USD.csv` (Monero)  
- `SOL-USD.csv` (Solana)  



## 🌐 Web Interface

A user-friendly web interface is developed with **Flask**, providing:

- Interactive visualization of actual vs. predicted cryptocurrency prices  
- Upload and selection of cryptocurrency datasets  
- Easy navigation to explore different cryptocurrencies  

The main web page (`index.html`) is located in the `templates/` folder.



## 🧠 Machine Learning Models

| Model              | Purpose                                |
|--------------------|---------------------------------------|
| Linear Regression   | Baseline regression model              |
| XGBoost Regressor   | Handles non-linear relationships       |
| LSTM Neural Network | Captures time-series dependencies and trends |

The LSTM model uses a window size of 21 days (`LSTM_WINDOW = 21`) to look back on historical data and predicts prices 7 days into the future (`FUTURE_STEPS = 7`).



## 🛠️ Technologies & Libraries

- **Python**  
- **Flask** (Web framework)  
- **Binance API** (Data collection)  
- **Pandas**, **NumPy** (Data manipulation)  
- **Matplotlib**, **Seaborn** (Visualization)  
- **Scikit-learn**, **XGBoost** (Machine learning)  
- **TensorFlow/Keras** (Deep learning with LSTM)  



## 👥 Team Members 

- **Burak SAYAR** – [GitHub Profile](https://github.com/BurakSayar)
- **Mert BUYUKNISAN** – [GitHub Profile](https://github.com/MertBuyuknisan)
- **Rahime GEDİK** – [GitHub Profile](https://github.com/rahimegedik)
- **Zeynep COL** – [GitHub Profile](https://github.com/zeynepcol)
