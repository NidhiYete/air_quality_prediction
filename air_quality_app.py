import streamlit as st
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import aqi
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "plotly_white"

# Read Excel file
air_quality = pd.read_excel('target/air_quality.xlsx')


# Title
st.title('Air Quality Analysis')

# Subheader
st.subheader('Ensemble Model and Evaluation')

# Make container
select_features = st.container()

# convert date into datetime type
air_quality['date_utc'] = pd.to_datetime(air_quality['date_utc'])

# Assuming 'date' column is in datetime format
air_quality['year'] = air_quality['date_utc'].dt.year

# copy dataset
aq_aqi = air_quality.copy()

aq_aqi.isna().sum()
aq_aqi = aq_aqi.drop('city', axis=1)

aq_df = aq_aqi.dropna()

numeric_features = (['co', 'o3', 'no2', 'so2', 'pm25', 'pm10'])

# outliers
for pollutants in numeric_features:
    aq_df[pollutants] = winsorize(aq_df[pollutants], limits=[0.01, 0.01])

# Define AQI breakpoints and corresponding AQI values
aqi_breakpoints = [
    (0, 12.0, 50), (12.1, 35.4, 100), (35.5, 55.4, 150),
    (55.5, 150.4, 200), (150.5, 250.4, 300), (250.5, 350.4, 400)
    ]

def calculate_aqi(pollutant_name, concentration):
    for low, high, aqi in aqi_breakpoints:
        if low <= concentration <= high:
            return aqi
    return None

def calculate_overall_aqi(row):
    aqi_values = []
    pollutants = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    for pollutant in pollutants:
        aqi = calculate_aqi(pollutant, row[pollutant])
        if aqi is not None:
            aqi_values.append(aqi)
    return max(aqi_values)

# Calculate AQI for each row
aq_df['AQI'] = aq_df.apply(calculate_overall_aqi, axis=1)


# Define AQI categories
aqi_categories = [
    (0, 50, 'Good'), (51, 100, 'Moderate'), (101, 150, 'Unhealthy for Sensitive Groups'),
    (151, 200, 'Unhealthy'), (201, 300, 'Very Unhealthy'), (301, 500, 'Hazardous')
]


def categorize_aqi(aqi_value):
    for low, high, category in aqi_categories:
        if low <= aqi_value <= high:
            return category
    return None

# Categorize AQI
aq_df['AQI_Category'] = aq_df['AQI'].apply(categorize_aqi)

air_new = aq_df.copy()

#st.write(air_new)
# drop columns
air_new.drop(['locationId', 'location', 'date_utc', 'coordinates', 'country', 'year'], inplace=True, axis=1)



# creating instances
ode = OrdinalEncoder()
scaler = PowerTransformer(method='yeo-johnson')

# column transformer
ct = make_column_transformer(
    (ode, ['AQI_Category']),
    (scaler, ['co', 'o3', 'no2', 'so2', 'pm25', 'pm10', 'AQI']),
    remainder='passthrough')


ct.set_output(transform="pandas")

# fit transform
air_new_df = ct.fit_transform(air_new)
#st.write(air_new_df)
# Get original column names
original_columns = ['AQI_Category', 'co', 'o3', 'no2', 'so2', 'pm25', 'pm10', 'AQI']


# Rename the columns in the resulting DataFrame
air_new_df.columns = original_columns


# Dataset split

# feature matrix and target variable
X = air_new.drop(columns=['AQI_Category'])
y = air_new['AQI_Category']



# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# rf model

# Use 'balanced' class weights in the classifier
rf_model = RandomForestClassifier(n_estimators=100,

                                  random_state=123,
                                  max_depth=10)

# Train the classifier on the training set
rf_model.fit(X_train, y_train)


def user_input_features():
    with select_features:
        st.subheader("Please Choose Hyperparameters of the Model.")

        # select sliders
        pol_co = st.slider("Choose the level of CO", 300, 10000, 100)
        st.write("Value is:", pol_co, "microgram per cubic meter")
        pol_o3 = st.slider("Choose the level of O3", 0, 200, 1000)
        st.write("Value is:", pol_o3, "microgram per cubic meter")
        pol_no2 = st.slider("Choose the level of NO2", 0, 200, 100)
        st.write("Value is:", pol_no2, "microgram per cubic meter")
        pol_so2 = st.slider("Choose the level of SO2", 0, 200)
        st.write("Value is:", pol_so2, "microgram per cubic meter")
        pol_pm25 = st.slider("Choose the level of PM25", 0, 200)
        st.write("Value is:", pol_pm25, "microgram per cubic meter")
        pol_pm10 = st.slider("Choose the level of PM10", 0, 200)
        st.write("Value is:", pol_pm10, "microgram per cubic meter")


        data = {
                'CO': pol_co,
                'O3': pol_o3,
                'NO2': pol_no2,
                'SO2': pol_so2,
                'PM25': pol_pm25,
                'PM10': pol_pm10}

        features = pd.DataFrame(data, index=[0])
        return features

df = user_input_features()
st.write(df)
st.subheader('Data Modeling and Evaluation')
st.sidebar.header("Choose Hyperparameters of the Models")

if st.button("Calculate AQI"):
    # Define AQI breakpoints and corresponding AQI values
    aqi_breakpoints = [
        (0, 12.0, 50), (12.1, 35.4, 100), (35.5, 55.4, 150),
        (55.5, 150.4, 200), (150.5, 250.4, 300), (250.5, 350.4, 400)
    ]


    def calculate_aqi(pollutant_name, concentration):
        for low, high, aqi in aqi_breakpoints:
            if low <= concentration <= high:
                return aqi
        return None


    def calculate_overall_aqi(row):
        aqi_values = []
        pollutants = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
        for pollutant in pollutants:
            aqi = calculate_aqi(pollutant, row[pollutant])
            if aqi is not None:
                aqi_values.append(aqi)
        return max(aqi_values)


    # Calculate AQI for each row
    df['AQI'] = df.apply(calculate_overall_aqi, axis=1)

    # Define AQI categories
    aqi_categories = [
        (0, 50, 'Good'), (51, 100, 'Moderate'), (101, 150, 'Unhealthy for Sensitive Groups'),
        (151, 200, 'Unhealthy'), (201, 300, 'Very Unhealthy'), (301, 500, 'Hazardous')
    ]


    def categorize_aqi(aqi_value):
        for low, high, category in aqi_categories:
            if low <= aqi_value <= high:
                return category
        return None


    # Categorize AQI
    df['AQI_Category'] = df['AQI'].apply(categorize_aqi)



    # drop columns
    df.drop(['pm10'], inplace=True, axis=1)


    # creating instances
    ode = OrdinalEncoder()
    scaler = PowerTransformer(method='yeo-johnson')

    # column transformer
    ct = make_column_transformer(
        (ode, ['AQI_Category']),
        (scaler, ['co', 'o3', 'no2', 'so2', 'pm25', 'pm10', 'AQI']),
        remainder='passthrough')

    ct.set_output(transform="pandas")

    # fit transform
    air_new_df = ct.fit_transform(df)

    # Get original column names
    original_columns = ['AQI_Category', 'co', 'o3', 'no2', 'so2', 'pm25', 'pm10', 'AQI']

    # Rename the columns in the resulting DataFrame
    air_new_df.columns = original_columns

    # predict
    rf_predictions_df = rf_model.predict(air_new_df)

    st.subheader("Air Quality Index")
    st.write(rf_predictions_df)
    st.write("Air Quality Index ", rf_predictions_df, "production per unit area")
    st.balloons()



