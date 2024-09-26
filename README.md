# Project-2-Group-3

# Project 1 of 2:
# Walmart Sales Forecasting Project

## Overview
This project analyzes historical sales data from Walmart stores to forecast future weekly sales. Using machine learning techniques, specifically a Random Forest Regressor, we explore the relationships between various factors and weekly sales figures.

## Dataset
The dataset contains information on sales from multiple Walmart stores, including:
- Store number
- Date
- Weekly Sales
- Holiday Flag
- Temperature
- Fuel Price
- Consumer Price Index (CPI)
- Unemployment Rate

## Analysis Steps
1. Data Loading and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering (e.g., extracting date components)
4. Model Training (Random Forest Regressor)
5. Model Evaluation
6. Feature Importance Analysis

## Key Findings
- The model achieved an R-squared score of approximately 0.96, indicating strong predictive performance.
- Top predictors of weekly sales are:
  1. Store
  2. Size
  3. Type
  4. Dept
  5. Temperature

## Requirements
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Usage
1. Ensure all required libraries are installed.
2. Place the Walmart dataset CSV file in the 'Resources' folder.
3. Run the Jupyter Notebook cells in order.
4. The notebook includes data preprocessing, model training, and result visualization.

## Future Work
- Implement time series analysis techniques for more accurate forecasting.
- Source department-level or item-level sales information by store for indepth analysis.
- Investigate store-specific trends and patterns, like foot traffic.
- Explore the impact of promotional events on sales and price elasticity modeling.
- Consider incorporating external data sources (e.g., proximity to competitors) to improve predictions.

## Conclusion
This analysis provides valuable insights into the factors influencing Walmart's weekly sales. The results can be useful for inventory management, staffing decisions, and overall business strategy. The high predictive accuracy of the model demonstrates the potential for data-driven decision making in retail operations.


# Project 2 of 2:
# Cardiovascular Disease Analysis Project

## Overview
This project analyzes a dataset of cardiovascular disease risk factors to build a predictive model for cardiovascular disease. Using machine learning techniques, we explore the relationships between various health indicators and the presence of cardiovascular disease.

## Dataset
The dataset contains information on 70,000 patients and includes the following features:
- Age
- Gender
- Height and Weight
- Blood Pressure (Systolic and Diastolic)
- Cholesterol Levels
- Glucose Levels
- Smoking Status
- Alcohol Intake
- Physical Activity
- Presence of Cardiovascular Disease (Target Variable)

## Analysis Steps
1. Data Loading and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training (Random Forest Classifier)
5. Model Evaluation
6. Feature Importance Analysis

## Key Findings
- The model achieved an accuracy of 71% in predicting cardiovascular disease.
- The top predictors of cardiovascular disease risk are:
  1. Age
  2. BMI (Body Mass Index)
  3. Systolic Blood Pressure
  4. Weight
  5. Height

## Requirements
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Usage
1. Ensure all required libraries are installed.
2. Run the Jupyter Notebook cells in order.
3. The notebook includes data preprocessing, model training, and result visualization.

## Future Work
- Explore other machine learning algorithms for comparison.
- Investigate feature interactions and their impact on predictions.
- Consider collecting additional relevant features to improve model performance.

## Conclusion
This analysis provides insights into the key factors associated with cardiovascular disease risk. The results can be useful for healthcare professionals in identifying high-risk individuals and developing targeted prevention strategies.
