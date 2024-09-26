import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to load cleaned data
@st.cache
def load_cleaned_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    return data

# Title of the app
st.title("Holiday Week Prediction using Ensemble Methods")

# Load the cleaned data directly from the path
cleaned_file_path = 'Resources/cleaned_Walmart_data.csv'
data = load_cleaned_data(cleaned_file_path)

if data is not None:
    st.write("Cleaned Data Loaded Successfully!")
    st.dataframe(data.head())

    # Select features and target variable
    X = data[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
    y = data['Holiday_Flag']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sidebar for model selection
    st.sidebar.header("Choose an Ensemble Learning Method")
    ensemble_choice = st.sidebar.selectbox("Select a Method", 
                                           ("Random Forest", "AdaBoost", "Gradient Boosting", "Voting Classifier"))

    # Hyperparameters for Random Forest
    if ensemble_choice == "Random Forest":
        st.sidebar.header("Random Forest Hyperparameters")
        n_estimators = st.sidebar.selectbox("Number of Estimators (Trees)", [50, 100, 200], index=1)
        max_depth = st.sidebar.selectbox("Max Depth", [None, 10, 20, 30], index=0)
        min_samples_split = st.sidebar.selectbox("Min Samples Split", [2, 5, 10], index=0)
        min_samples_leaf = st.sidebar.selectbox("Min Samples Leaf", [1, 2, 4], index=0)

    # Button to train the model
    if st.sidebar.button("Train Model"):
        if ensemble_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators, 
                                           max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf, 
                                           random_state=42)
        elif ensemble_choice == "AdaBoost":
            model = AdaBoostClassifier(n_estimators=100, random_state=42)
        elif ensemble_choice == "Gradient Boosting":
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        elif ensemble_choice == "Voting Classifier":
            # Use Logistic Regression, SVC, and Random Forest in the Voting Classifier
            model1 = LogisticRegression()
            model2 = SVC(probability=True)
            model3 = RandomForestClassifier(n_estimators=100, random_state=42)
            model = VotingClassifier(estimators=[('lr', model1), ('svc', model2), ('rf', model3)], voting='soft')

        # Train the selected model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Display the results
        st.subheader(f"Model Performance: {ensemble_choice}")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(report)

        # Hyperparameter grid if Random Forest is selected
        if ensemble_choice == "Random Forest":
            st.subheader("Random Forest Hyperparameters Selected:")
            st.write(f"Number of Estimators: {n_estimators}")
            st.write(f"Max Depth: {max_depth}")
            st.write(f"Min Samples Split: {min_samples_split}")
            st.write(f"Min Samples Leaf: {min_samples_leaf}")