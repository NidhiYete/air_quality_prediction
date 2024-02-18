import streamlit as st
import datetime
import os
import ast
from dotenv import load_dotenv
import pandas as pd
import requests
import json
import csv
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.stats.mstats import winsorize
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import aqi
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "plotly_white"

st.set_page_config(
    page_title='Air Quality Index Prediction',
    page_icon=":seedling:",
)

load_dotenv()
api_key = os.getenv("API_KEY")

# Read Excel file
air_quality = pd.read_csv('data_2/air_data.csv')



# Make container
select_features = st.container()

# 'coordinates' column contains strings like "[latitude, longitude]"
air_quality['coordinates'] = air_quality['coordinates'].apply(ast.literal_eval)

# Separate 'coordinates' column into 'latitude' and 'longitude'
air_quality[['latitude', 'longitude']] = pd.DataFrame(air_quality['coordinates'].tolist(), index=air_quality.index)

# Drop the original 'coordinates' column
air_quality = air_quality.drop('coordinates', axis=1)

# convert numeric columns to the appropriate data type
numeric_columns = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'latitude', 'longitude']

for col in numeric_columns:
    air_quality[col] = pd.to_numeric(air_quality[col], errors='coerce')

# Convert 'date' column to datetime format
air_quality['date'] = pd.to_datetime(air_quality['date'])

# Display the data types after conversion
print(air_quality.dtypes)


st.map(data=air_quality, latitude="latitude", longitude="longitude")

# Assuming 'date' column is in datetime format
air_quality['year'] = air_quality['date'].dt.year

# copy dataset
aq_aqi = air_quality.copy()


aq_df = aq_aqi.dropna()

numeric_features = (['pm25', 'pm10', 'o3', 'no2', 'so2', 'co'])
for col in numeric_features:
    aq_df[col] = winsorize(aq_df[col], limits=[0.001, 0.001])


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
    pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
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
air_new.drop(['city', 'year', 'date', 'pm10', 'AQI', 'latitude', 'longitude'], inplace=True, axis=1)


# creating instances
ode = OrdinalEncoder()
scaler = MinMaxScaler()
#scaler = PowerTransformer(method='box-cox', standardize=False)

# column transformer
ct = make_column_transformer(
    #(ode, ['AQI_Category']),
    (scaler, ['pm25', 'o3', 'no2', 'so2', 'co']),
    remainder='passthrough')


ct.set_output(transform="pandas")

# fit transform
air_new_df = ct.fit_transform(air_new)
#st.write(air_new_df)
# Get original column names
original_columns = ['pm25', 'o3', 'no2', 'so2', 'co', 'AQI_Category']


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
rf_model = RandomForestClassifier()

# Train the classifier on the training set
rf_model.fit(X_train, y_train)

# Random Oversampling
rf_pipeline = make_pipeline(SMOTE(), rf_model)

# Step 3: Train your model using the training set with oversampling or undersampling
rf_pipeline.fit(X_train, y_train)



def extract_data(json_response):
    # Load the JSON response
    response_dict = json.loads(json_response)

    # Extract specific fields
    iaqi_keys = ["co", "so2", "pm25", "no2", "pm10", "o3"]
    iaqi_data = {key: response_dict['data']['iaqi'][key]['v'] for key in iaqi_keys}
    city_geo_location = response_dict['data']['city']['geo']
    time_s_field = response_dict['data']['time']['s']
    aqi_data = response_dict['data']['aqi']

    # Return the extracted data
    return {
        "date": time_s_field,
        "co": iaqi_data["co"],
        "so2": iaqi_data["so2"],
        "pm25": iaqi_data["pm25"],
        "no2": iaqi_data["no2"],
        "pm10": iaqi_data["pm10"],
        "o3": iaqi_data["o3"],
        "AQI": aqi_data,
        "city": city_geo_location

    }

def create_csv_file(data, file_path):
    # Append data to CSV file
    with open(file_path, mode='a', newline='') as csv_file:
        fieldnames = data.keys()
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Check if the CSV file is empty, if so, write header
        if csv_file.tell() == 0:
            csv_writer.writeheader()

        csv_writer.writerow(data)




#  air quality widget
def air_quality_widget(aqi_value):
    # Define AQI categories and colors
    aqi_categories = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    colors = ['green', 'yellow', 'orange', 'red', 'purple', 'maroon']

    # Determine AQI category index
    category_index = min(int(aqi_value / 100), 5)

    # Create gauge chart
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=aqi_value,
        mode="gauge+number",
        title={'text': "Air Quality Index"},
        gauge={
            'axis': {'range': [None, 500]},
            'bar': {'color': colors[category_index]},
            'steps': [
                {'range': [0, 50], 'color': colors[0]},
                {'range': [51, 100], 'color': colors[1]},
                {'range': [101, 150], 'color': colors[2]},
                {'range': [151, 200], 'color': colors[3]},
                {'range': [201, 300], 'color': colors[4]},
                {'range': [301, 500], 'color': colors[5]},
            ],
        }
    ))

    # Display the gauge chart
    st.plotly_chart(fig)


st.title('Air Quality Index Prediction')
st.subheader('Ensemble Model and Evaluation')

categories = ('MM-Delhi', 'RKP-Delhi', 'Pusa-Delhi', 'DCNS-Delhi', 'Greater-Noida')

# Create select boxes for category and visualization
category = st.selectbox('Choose a Location', list(categories))



if category == 'MM-Delhi':
    if os.path.exists('real_data.csv'):
        os.remove('real_data.csv')
        print(f"Deleted existing CSV file: {'real_data.csv'}")


    url = f"https://api.waqi.info/feed/@2554/?token={api_key}"

    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    response = requests.get(url, headers=headers)

    # Extract data
    extracted_data = extract_data(response.text)
    csv_file_path = 'real_data.csv'
    # Create CSV file
    create_csv_file(extracted_data, csv_file_path)

    df = pd.read_csv('real_data.csv')


    time_1 = st.date_input("Date", datetime.date.today())

    location_name = st.text_input('Location name', 'Mandir Marg, Delhi')

    if st.button("Calculate Air Quality Index"):
        df.drop(['pm10', 'AQI', 'date', 'city'], inplace=True, axis=1)

        # column transformer
        ct = make_column_transformer(
            # (ode, ['AQI_Category']),
            (scaler, ['pm25', 'o3', 'no2', 'so2', 'co']),
            remainder='passthrough')

        ct.set_output(transform="pandas")

        # fit transform
        new_df = ct.fit_transform(df)
        # st.write(air_new_df)
        # Get original column names
        original_col = ['pm25', 'o3', 'no2', 'so2', 'co']

        # Rename the columns in the resulting DataFrame
        new_df.columns = original_col
        # predict
        rf_predictions_df = rf_pipeline.predict(new_df)

        st.subheader("Air Quality Index")
        st.write(rf_predictions_df)
        st.balloons()




elif category == 'Pusa-Delhi':

    if os.path.exists('real_data.csv'):
        os.remove('real_data.csv')
        print(f"Deleted existing CSV file: {'real_data.csv'}")



    url = f"https://api.waqi.info/feed/@10124/?token={api_key}"

    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    response = requests.get(url, headers=headers)

    # Extract data
    extracted_data = extract_data(response.text)
    csv_file_path = 'real_data.csv'
    # Create CSV file
    create_csv_file(extracted_data, csv_file_path)

    df = pd.read_csv('real_data.csv')
    #st.write(df)

    time_1 = st.date_input("Date", datetime.date.today())


    location_name = st.text_input('Location name', 'Pusa, Delhi')

    if st.button("Calculate Air Quality Index"):
        df.drop(['pm10', 'AQI', 'date', 'city'], inplace=True, axis=1)

        # column transformer
        ct = make_column_transformer(
            # (ode, ['AQI_Category']),
            (scaler, ['pm25', 'o3', 'no2', 'so2', 'co']),
            remainder='passthrough')

        ct.set_output(transform="pandas")

        # fit transform
        new_df = ct.fit_transform(df)
        # st.write(air_new_df)
        # Get original column names
        original_col = ['pm25', 'o3', 'no2', 'so2', 'co']

        # Rename the columns in the resulting DataFrame
        new_df.columns = original_col
        # predict
        rf_predictions_df = rf_pipeline.predict(new_df)

        st.subheader("Air Quality Index")
        st.write(rf_predictions_df)
        st.balloons()




elif category == 'DCNS-Delhi':
    if os.path.exists('real_data.csv'):
        os.remove('real_data.csv')
        print(f"Deleted existing CSV file: {'real_data.csv'}")

    url = f"https://api.waqi.info/feed/@10111/?token={api_key}"

    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    response = requests.get(url, headers=headers)

    # Extract data
    extracted_data = extract_data(response.text)
    csv_file_path = 'real_data.csv'
    # Create CSV file
    create_csv_file(extracted_data, csv_file_path)

    df = pd.read_csv('real_data.csv')


    time_1 = st.date_input("Date", datetime.date.today())

    location_name = st.text_input('Location name', 'Major Dhyan Chand National Stadium, Delhi')

    if st.button("Calculate Air Quality Index"):
        df.drop(['pm10', 'AQI', 'date', 'city'], inplace=True, axis=1)

        # column transformer
        ct = make_column_transformer(
            # (ode, ['AQI_Category']),
            (scaler, ['pm25', 'o3', 'no2', 'so2', 'co']),
            remainder='passthrough')

        ct.set_output(transform="pandas")

        # fit transform
        new_df = ct.fit_transform(df)
        # st.write(air_new_df)
        # Get original column names
        original_col = ['pm25', 'o3', 'no2', 'so2', 'co']

        # Rename the columns in the resulting DataFrame
        new_df.columns = original_col
        # predict
        rf_predictions_df = rf_pipeline.predict(new_df)

        st.subheader("Air Quality Index")
        st.write(rf_predictions_df)
        st.balloons()



elif category == 'Greater-Noida':
    if os.path.exists('real_data.csv'):
        os.remove('real_data.csv')
        print(f"Deleted existing CSV file: {'real_data.csv'}")

    url = f"https://api.waqi.info/feed/@12463/?token={api_key}"

    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    response = requests.get(url, headers=headers)

    # Extract data
    extracted_data = extract_data(response.text)
    csv_file_path = 'real_data.csv'
    # Create CSV file
    create_csv_file(extracted_data, csv_file_path)

    df = pd.read_csv('real_data.csv')


    time_1 = st.date_input("Date", datetime.date.today())

    location_name = st.text_input('Location name', 'Knowledge Park- V, Greater Noida')

    if st.button("Calculate Air Quality Index"):
        df.drop(['pm10', 'AQI', 'date', 'city'], inplace=True, axis=1)

        # column transformer
        ct = make_column_transformer(
            # (ode, ['AQI_Category']),
            (scaler, ['pm25', 'o3', 'no2', 'so2', 'co']),
            remainder='passthrough')

        ct.set_output(transform="pandas")

        # fit transform
        new_df = ct.fit_transform(df)
        # st.write(air_new_df)
        # Get original column names
        original_col = ['pm25', 'o3', 'no2', 'so2', 'co']

        # Rename the columns in the resulting DataFrame
        new_df.columns = original_col
        # predict
        rf_predictions_df = rf_pipeline.predict(new_df)

        st.subheader("Air Quality Index")
        st.write(rf_predictions_df)
        st.balloons()



elif category == 'RKP-Delhi':
    if os.path.exists('real_data.csv'):
        os.remove('real_data.csv')
        print(f"Deleted existing CSV file: {'real_data.csv'}")

    url = f"https://api.waqi.info/feed/@2556/?token={api_key}"

    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    response = requests.get(url, headers=headers)

    # Extract data
    extracted_data = extract_data(response.text)
    csv_file_path = 'real_data.csv'
    # Create CSV file
    create_csv_file(extracted_data, csv_file_path)

    df = pd.read_csv('real_data.csv')


    time_1 = st.date_input("Date", datetime.date.today())

    location_name = st.text_input('Location name', 'R.K Puram, Delhi')

    if st.button("Calculate Air Quality Index"):
        st.subheader("Air Quality Index")
        #sample_aqi = df['AQI'].iloc[0]
        #air_quality_widget(sample_aqi)

        df.drop(['pm10', 'AQI', 'date', 'city'], inplace=True, axis=1)

        # column transformer
        ct = make_column_transformer(
            # (ode, ['AQI_Category']),
            (scaler, ['pm25', 'o3', 'no2', 'so2', 'co']),
            remainder='passthrough')

        ct.set_output(transform="pandas")

        # fit transform
        new_df = ct.fit_transform(df)
        # st.write(air_new_df)
        # Get original column names
        original_col = ['pm25', 'o3', 'no2', 'so2', 'co']

        # Rename the columns in the resulting DataFrame
        new_df.columns = original_col
        # predict
        rf_predictions_df = rf_pipeline.predict(new_df)
        st.write(rf_predictions_df)
        st.balloons()










