# Dogecoin Cryptocurrency Analysis

📖 **Project Overview**
This project provides a comprehensive analysis of Dogecoin (DOGE), focusing on historical price trends, volatility patterns, and predictive modeling using both traditional machine learning and deep learning approaches.  

🎯 **Project Objectives**
Collect and preprocess historical Dogecoin price data

Perform exploratory data analysis to identify trends and patterns

Develop predictive models for short-term price forecasting

Compare Dogecoin's performance against major cryptocurrencies (Bitcoin and Ethereum)

Evaluate model performance using appropriate metrics

📊 **Data Sources**
Price Data: Historical OHLCV (Open, High, Low, Close, Volume) data for Dogecoin, Bitcoin, and Ethereum from Yahoo Finance 

Technical Indicators: Calculated from price data (moving averages,trading volumes,volatility and daily returns etc.)

🛠️ **Technical Stack**
Programming Language: Python 

Data Processing: Pandas, NumPy

Visualization: Matplotlib

Machine Learning: Scikit-learn, XGBoost, Random forest

Deep Learning: TensorFlow, Keras (LSTM)

Data Collection: yfinance

📈 **Project Workflow**
Data Collection
Price Data: Yahoo Finance for DOGE, BTC, ETH historical OHLCV

Compilation: Combined price, volume, and market cap.

2. Data Preprocessing

Calculations: Daily returns and volatility metrics

3. Exploratory Data Analysis (EDA)
Visualization: Price movements and trading volumes

Trend Analysis: 7-day and 30-day moving averages

Volatility Analysis: Daily returns and risk metrics

Comparative Analysis: DOGE vs. BTC and ETH performance

Correlation Analysis: Inter-cryptocurrency relationships

5. Predictive Modeling
Traditional ML
Random Forest Regressor

XGBoost Regressor

Deep Learning
LSTM (Long Short-Term Memory)

6. Model Evaluation
Validation: Time-series cross-validation

Metrics: RMSE, MAE, MAPE, R² score

Performance Comparison: ML vs. DL models
