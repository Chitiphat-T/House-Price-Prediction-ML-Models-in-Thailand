# Thailand Housing Price Prediction

A modular machine learning pipeline built to estimate residential property prices across Thailand. This project compares **Linear Regression**, **Decision Trees**, and **XGBoost** to identify the most accurate valuation model for the local market.

---

## Project Performance
By leveraging gradient boosting, the pipeline achieved high predictive accuracy, specifically excelling in the high-volume 2M–12M THB segment.

| Model | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **XGBoost** | 3,081,056 | 1,753,957 | **0.8009** |
| **Decision Tree** | 3,514,376 | 2,071,383 | 0.7410 |
| **Linear Regression** | 4,750,043 | 3,057,311 | 0.5269 |

---

## Project Structure

```
├── data/               # Automated data storage (managed by gdown)
├── src/                # Modular logic
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── model_training.py
├── results/            # Performance plots and saved metrics
├── main.py             # Main entry point
├── requirements.txt    # Dependency list
└── README.md           # README file
```

---

## Getting Started

### File Location
* https://drive.google.com/file/d/1C0dxQBvT92qE1dVLtnhtbzvM4JHu3sF6/view?usp=drive_link

### **Requirements**
* Python 3.9+
* Libraries listed in `requirements.txt`

### **Installation & Execution**
1. **Clone the repository**
     ```
   git clone [https://github.com/Chitiphat-T/House-Price-Prediction-ML-Models-in-Thailand.git](https://github.com/Chitiphat-T/House-Price-Prediction-ML-Models-in-Thailand.git)
   cd House-Price-Prediction-ML-Models-in-Thailand

2. **Install Dependencies**
     ```
    pip install - r requirements.txt

3. **Run 'main.py'**
     ```
    python main.py

## Conclusion
This project successfully met its goal of creating a reliable prediction tool for Thai real estate. XGBoost proved to be the superior choice for handling the non-linear nature of property pricing.
