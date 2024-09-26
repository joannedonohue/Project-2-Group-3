import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

# Function to load cleaned data
@st.cache
def load_cleaned_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data['Month'] = data['Date'].dt.month
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week
    return data

# Function to add lag and rolling features
def add_features(data):
    # Add lag and rolling mean features
    data['Lag_Weekly_Sales'] = data['Weekly_Sales'].shift(1)
    data['Rolling_Mean_Sales_Monthly'] = data['Weekly_Sales'].rolling(window=4).mean()

    # Add holiday season flag and interaction feature
    data['Holiday_Season'] = data['Month'].apply(lambda x: 1 if x in [11, 12] else 0)  # Holiday season flag
    data['Temp_Fuel_Interaction'] = data['Temperature'] * data['Fuel_Price']  # Interaction feature
    
    # Drop rows with missing values (from lag/rolling features)
    data = data.dropna()
    return data

# Load and process data
cleaned_file_path = 'Resources/cleaned_Walmart_data.csv'
data = load_cleaned_data(cleaned_file_path)
data = add_features(data)

# Features and target
X = data[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag', 'Month', 'WeekOfYear', 'Lag_Weekly_Sales', 'Rolling_Mean_Sales_Monthly', 'Holiday_Season', 'Temp_Fuel_Interaction']]
y = data['Weekly_Sales']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar for model selection
st.sidebar.header("Choose a Regression Model")
model_choice = st.sidebar.selectbox("Select a Model", 
                                    ("Random Forest", "Gradient Boosting", "XGBoost", "Stacking Regressor", "Ensemble (Voting Regressor)"))

# Button to train the model
if st.sidebar.button("Train Model"):
    
    # Model selection based on user choice
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    
    elif model_choice == "XGBoost":
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        xgb_model = XGBRegressor(random_state=42)
        random_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=10, cv=5, verbose=2, n_jobs=-1)
        random_search.fit(X_train_scaled, y_train)
        model = random_search.best_estimator_

    elif model_choice == "Stacking Regressor":
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]
        meta_model = Ridge()
        model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    
    elif model_choice == "Ensemble (Voting Regressor)":
        model1 = RandomForestRegressor(n_estimators=100, random_state=42)
        model2 = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model3 = XGBRegressor(n_estimators=100, random_state=42)
        model = VotingRegressor([('rf', model1), ('gb', model2), ('xgb', model3)])

    # Train the selected model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model's performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display the results
    st.subheader(f"Model Performance: {model_choice}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-Squared: {r2:.2f}")

    # Perform cross-validation predictions (if applicable)
    if model_choice != "XGBoost":  # XGBoost already does hyperparameter tuning
        cv_scores = cross_val_predict(model, X_train_scaled, y_train, cv=5)
        r2_cv = r2_score(y_train, cv_scores)
        st.subheader("Cross-Validation Performance")
        st.write(f"Cross-Validation R-Squared: {r2_cv:.2f}")
