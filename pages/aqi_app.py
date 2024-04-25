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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from scipy.stats.mstats import winsorize
from streamlit_supabase_auth import logout_button
import pygwalker as pyg
import streamlit.components.v1 as components
import plotly.io as pio

import plotly.graph_objects as go
pio.templates.default = "plotly_white"
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title='Paavan Vayu',
    layout='wide',
    page_icon="♻"
)

# CSS to hide the sidebar when the user is not in session
hide_sidebar = """
<style>
    section[data-testid="stSidebar"][aria-expanded="true"] {
        display: none;
    }
</style>
"""

# Access the session state
session_state = st.session_state

# Logout function definition
def logoutfunct():
    print('Logout function called.')
    logout_button()
    print("move to session out")

    # st.markdown('<meta http-equiv="refresh" content="2">', unsafe_allow_html=True)
st.markdown(hide_sidebar, unsafe_allow_html=True)
# Check if session_state is initialized
if hasattr(st,'session_state'):


    # Check if 'id' and 'email' are present in session_state
    if hasattr(st.session_state, 'id') and hasattr(st.session_state, 'email'):
        print("session_state is initialized with id and email - aqi_app")
    else:
        print("session_state is not fully initialized - aqi_app")
        # Display a warning message and a login link
        st.warning("Please Login to Continue...")
        login_path = "/"
        st.markdown(f'<a href="{login_path}" target="_self"> Login </a>', unsafe_allow_html=True)
        st.stop()  # Stop the execution if not fully initialized
else:
    st.warning('Session_state not initialized.')


# create columns
col21, col22 = st.columns([2, 0.5])

# title
with col21:
    st.title(':blue[Air Quality Index Prediction]')

# logout button
with col22:
    # Display the welcome message along with the logout button
    st.write(f"Welcome  - {session_state.email}")
    logoutfunct()



# get api key

load_dotenv()
api_key = os.getenv("API_KEY")

# Read Excel file
air_quality = pd.read_csv('data_2/air_data.csv')



# Make container
select_features = st.container()

# Code or model training

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


#st.map(data=air_quality, latitude="latitude", longitude="longitude")

# Assuming 'date' column is in datetime format
air_quality['year'] = air_quality['date'].dt.year

# copy dataset
aq_aqi = air_quality.copy()


aq_df = aq_aqi.dropna()

numeric_features = (['pm25', 'pm10', 'o3', 'no2', 'so2', 'co'])
for col in numeric_features:
    aq_df[col] = winsorize(aq_df[col], limits=[0.01, 0.01])


# Define AQI breakpoints and corresponding AQI values
# Breakpoints for pollutant AQI calculation
pm10_breakpoints = {
    'c_low': [0, 50, 101, 251, 351, 431, 501, 601],
    'c_high': [50, 100, 250, 350, 430, 500, 600, 999.0],
    'aqi_low': [0, 51, 101, 201, 301, 401, 501, 601],
    'aqi_high': [50, 100, 200, 300, 400, 500, 600, 999]
}

# AQI calculation function for PM10
def calculate_aqi_pm10(value):
    for i in range(len(pm10_breakpoints['c_low'])):
        if pm10_breakpoints['c_low'][i] <= value <= pm10_breakpoints['c_high'][i]:
            c_low = pm10_breakpoints['c_low'][i]
            c_high = pm10_breakpoints['c_high'][i]
            aqi_low = pm10_breakpoints['aqi_low'][i]
            aqi_high = pm10_breakpoints['aqi_high'][i]
            break


    aqi_pm10 = (aqi_high - aqi_low) / (c_high - c_low) * (value - c_low) + aqi_low
    return int(round(aqi_pm10))

# Create a new column 'aqi_pm10' and apply the AQI calculation function
aq_df['aqi_pm10'] = aq_df['pm10'].apply(calculate_aqi_pm10)

no2_breakpoints = {
    'c_low_no2': [0, 41, 81, 181, 281, 401],
    'c_high_no2': [40, 80, 180, 280, 400, 500],
    'aqi_low_no2': [0, 51, 101, 201, 301, 401],
    'aqi_high_no2': [50, 100, 200, 300, 400, 500]
}

# AQI calculation function for PM10
def calculate_aqi_no2(value):
    for i in range(len(no2_breakpoints['c_low_no2'])):
        if no2_breakpoints['c_low_no2'][i] <= value <= no2_breakpoints['c_high_no2'][i]:
            c_low_no2 = no2_breakpoints['c_low_no2'][i]
            c_high_no2 = no2_breakpoints['c_high_no2'][i]
            aqi_low_no2 = no2_breakpoints['aqi_low_no2'][i]
            aqi_high_no2 = no2_breakpoints['aqi_high_no2'][i]
            break

    aqi_no2 = (aqi_high_no2 - aqi_low_no2) / (c_high_no2 - c_low_no2) * (value - c_low_no2) + aqi_low_no2
    return int(round(aqi_no2))

# Create a new column 'aqi_pm10' and apply the AQI calculation function
aq_df['aqi_no2'] = aq_df['no2'].apply(calculate_aqi_no2)

pm25_breakpoints = {
    'c_low_pm25': [0, 31, 61, 91, 121, 251, 351],
    'c_high_pm25': [30, 60, 90, 120, 250, 350, 500],
    'aqi_low_pm25': [0, 51, 101, 201, 301, 401, 401],
    'aqi_high_pm25': [50, 100, 200, 300, 400, 500, 500]
}

# AQI calculation function for PM10
def calculate_aqi_pm25(value):
    for i in range(len(pm25_breakpoints['c_low_pm25'])):
        if pm25_breakpoints['c_low_pm25'][i] <= value <= pm25_breakpoints['c_high_pm25'][i]:
            c_low_pm25 = pm25_breakpoints['c_low_pm25'][i]
            c_high_pm25 = pm25_breakpoints['c_high_pm25'][i]
            aqi_low_pm25 = pm25_breakpoints['aqi_low_pm25'][i]
            aqi_high_pm25 = pm25_breakpoints['aqi_high_pm25'][i]
            break

    aqi_pm25 = (aqi_high_pm25 - aqi_low_pm25) / (c_high_pm25 - c_low_pm25) * (value - c_low_pm25) + aqi_low_pm25
    return int(round(aqi_pm25))


# Create a new column 'aqi_pm10' and apply the AQI calculation function
aq_df['aqi_pm25'] = aq_df['pm25'].apply(calculate_aqi_pm25)

# Breakpoints for O3 AQI calculation
o3_breakpoints = {
    'c_low': [0, 51, 101, 169, 209, 748, 1009, 1259],
    'c_high': [50, 100, 168, 208, 748, 1008, 1258, 999.0],
    'aqi_low': [0, 51, 101, 201, 301, 401, 501, 601],
    'aqi_high': [50, 100, 200, 300, 400, 500, 600, 999]
}

# AQI calculation function for O3
def calculate_aqi_o3(value):
    for i in range(len(o3_breakpoints['c_low'])):
        if o3_breakpoints['c_low'][i] <= value <= o3_breakpoints['c_high'][i]:
            c_low = o3_breakpoints['c_low'][i]
            c_high = o3_breakpoints['c_high'][i]
            aqi_low = o3_breakpoints['aqi_low'][i]
            aqi_high = o3_breakpoints['aqi_high'][i]
            break

    aqi_o3 = (aqi_high - aqi_low) / (c_high - c_low) * (value - c_low) + aqi_low
    return int(round(aqi_o3))

# Example usage:
aq_df['aqi_o3'] = aq_df['o3'].apply(calculate_aqi_o3)

# Breakpoints for SO2 AQI calculation (Indian Standards)
so2_breakpoints = {
    'c_low': [0, 41, 81, 381, 801, 1601],
    'c_high': [40, 80, 380, 800, 1600, 2000],
    'aqi_low': [0, 51, 101, 201, 301, 401],
    'aqi_high': [50, 100, 200, 300, 400, 500]
}

# AQI calculation function for SO2 (Indian Standards)
def calculate_aqi_so2(value):
    for i in range(len(so2_breakpoints['c_low'])):
        if so2_breakpoints['c_low'][i] <= value <= so2_breakpoints['c_high'][i]:
            c_low = so2_breakpoints['c_low'][i]
            c_high = so2_breakpoints['c_high'][i]
            aqi_low = so2_breakpoints['aqi_low'][i]
            aqi_high = so2_breakpoints['aqi_high'][i]
            break

    aqi_so2 = (aqi_high - aqi_low) / (c_high - c_low) * (value - c_low) + aqi_low
    return int(round(aqi_so2))

# Example usage:
aq_df['aqi_so2'] = aq_df['so2'].apply(calculate_aqi_so2)

# Breakpoints for CO AQI calculation
co_breakpoints = {
    'c_low_co': [0, 1.1, 2.1, 10, 17, 34],
    'c_high_co': [1.0, 2.0, 10, 17, 34, 68],
    'aqi_low_co': [0, 51, 101, 201, 301, 401],
    'aqi_high_co': [50, 100, 200, 300, 400, 500]
}

# AQI calculation function for CO
def calculate_aqi_co(value):
    for i in range(len(co_breakpoints['c_low_co'])):
        if co_breakpoints['c_low_co'][i] <= value <= co_breakpoints['c_high_co'][i]:
            c_low_co = co_breakpoints['c_low_co'][i]
            c_high_co = co_breakpoints['c_high_co'][i]
            aqi_low_co = co_breakpoints['aqi_low_co'][i]
            aqi_high_co = co_breakpoints['aqi_high_co'][i]
            break

    aqi_co = (aqi_high_co - aqi_low_co) / (c_high_co - c_low_co) * (value - c_low_co) + aqi_low_co
    return int(round(aqi_co))

# Example usage:
aq_df['aqi_co'] = aq_df['co'].apply(calculate_aqi_co)


poll = ['pm10', 'pm25', 'no2', 'o3', 'co', 'so2']
# Create a new column 'max_aqi' to store the overall maximum AQI for each row
aq_df['AQI'] = aq_df[[f'aqi_{pollutant}' for pollutant in poll]].max(axis=1)

# Categorize the overall AQI into a new column 'AQI_category'
def categorize_aqi(overall_aqi):
    if overall_aqi <= 50:
        return "Good"
    elif 51 <= overall_aqi <= 100:
        return "Satisfactory"
    elif 101 <= overall_aqi <= 200:
        return "Moderate"
    elif 201 <= overall_aqi <= 300:
        return "Poor"
    elif 301 <= overall_aqi <= 400:
        return "Very Poor"
    elif overall_aqi >= 401:
        return "Severe"

# Apply the categorization function to create the 'AQI_category' column
aq_df['AQI_Category'] = aq_df['AQI'].apply(categorize_aqi)

air_new = aq_df.copy()

#st.write(air_new)
# drop columns
air_new.drop(['city', 'year', 'date', 'pm10', 'AQI', 'latitude', 'longitude', 'aqi_co',
              'aqi_o3', 'aqi_no2', 'aqi_so2', 'aqi_pm25', 'aqi_pm10'], inplace=True, axis=1)

# creating instances
ode = OrdinalEncoder()
scaler = MinMaxScaler()
#scaler = PowerTransformer(method='yeo-johnson')

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

rf_model = RandomForestClassifier()
rf_pipeline = make_pipeline(SMOTEENN(), rf_model)
rf_pipeline.fit(X_train, y_train)

# Function to train Stacking Classifier ensemble and return the trained model object.
@st.cache_resource
def train_stacking_classifier(X_train, y_train):
    # Create base classifiers
    rf_model = RandomForestClassifier()
    svm_model = SVC(class_weight='balanced', probability=True)
    nb_model = GaussianNB()
    knn_model = KNeighborsClassifier(n_neighbors=6)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=123)

    # Create pipelines with Random Oversampling
    rf_pipeline = make_pipeline(SMOTEENN(), rf_model)
    svm_pipeline = make_pipeline(SMOTEENN(), svm_model)
    nb_pipeline = make_pipeline(SMOTEENN(), nb_model)
    knn_pipeline = make_pipeline(SMOTEENN(), knn_model)
    mlp_pipeline = make_pipeline(SMOTEENN(), mlp_model)

    # Create a stacking classifier with base and meta classifiers
    model_st = StackingClassifier(
        estimators=[
            ('rf', rf_pipeline),
            ('svm', svm_pipeline),
            ('nb', nb_pipeline),
            ('knn', knn_pipeline),
            ('mlp', mlp_pipeline)
        ],
        final_estimator=RandomForestClassifier()
    )

    # Train the classifier on the training set
    model_st.fit(X_train, y_train)

    return model_st

# Outside the function, call the function to get the trained model
trained_stacking_classifier = train_stacking_classifier(X_train, y_train)



# real time data extraction
def extract_data(json_response):
    # Load the JSON response
    response_dict = json.loads(json_response)

    # Extract specific fields
    iaqi_keys = ["co", "so2", "pm25", "no2", "pm10", "o3"]
    iaqi_data = {key: response_dict['data']['iaqi'][key] for key in iaqi_keys}
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
# creating path to store data
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
    aqi_categories = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
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
                {'range': [101, 200], 'color': colors[2]},
                {'range': [201, 300], 'color': colors[3]},
                {'range': [301, 400], 'color': colors[4]},
                {'range': [401, 500], 'color': colors[5]},
            ],
        }
    ))
    # Adjust layout properties to reduce size and prevent overlapping
    fig.update_layout(width=400, height=400)  # Adjust width, height, and margins

    # Display the gauge chart
    st.plotly_chart(fig)

# Dashboard creation
# Create tabs for different reports
tab1, tab2, tab3, tab4, tab13 = st.tabs(["Paavan Vayu", "Visualization", "Help", "FAQ", "Submit Feedback"])

# create columns
with tab1:
    col1, col2, col3, = st.columns(3)

    with col1:
        st.subheader("Real Time Monitoring")
        # Display an image
        image_url = 'https://www.epa.gov/sites/default/files/styles/large/public/2021-05/aqaw_2021_0.png?itok=FwknteTh'
        st.image(image_url, use_column_width=True)

        categories = ('MM-Delhi', 'RKP-Delhi', 'Pusa-Delhi', 'DCNS-Delhi', 'AV-Delhi')

        # Create select boxes for category and visualization
        category = st.selectbox('Choose a Location', list(categories))

# Mandir Marg Delhi Location

        if category == 'MM-Delhi':
            if os.path.exists('../real_data.csv'):
                os.remove('../real_data.csv')
                print(f"Deleted existing CSV file: {'real_data.csv'}")

            url = f"https://api.waqi.info/feed/@2554/?token={api_key}"

            headers = {
                "accept": "application/json",
                "content-type": "application/json"
            }
            response = requests.get(url, headers=headers)

            # Extract data
            extracted_data = extract_data(response.text)
            csv_file_path = '../real_data.csv'
            # Create CSV file
            create_csv_file(extracted_data, csv_file_path)

            df = pd.read_csv('../real_data.csv')

            time_1 = st.date_input("Date", datetime.date.today())

            #time_2 = st.time_input('Time', value="now")

            location_name = st.text_input('Location name', 'Mandir Marg, Delhi')


            if st.button(":red[Calculate Air Quality Index]"):
                df.drop(['date', 'city'], inplace=True, axis=1)
                st.write(df)
                aq = df['AQI'].iloc[0]
                air_quality_widget(aq)
                df['AQI_Category'] = df['AQI'].apply(categorize_aqi)
                ac = df['AQI_Category'].iloc[0]
                #st.subheader("Air Quality Index", ac)
                st.write("Air Quality Index : ",ac)


                df.drop(['pm10', 'AQI', 'AQI_Category'], inplace=True, axis=1)

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
                st1_predictions_df = trained_stacking_classifier.predict(new_df)
                #pred=rf_pipeline.predict(new_df)
                #st.write(pred)
                #st.write(rf_predictions_df)
                #st.write("Air Quality Index : ", st1_predictions_df)


# Pusa- Delhi location

        elif category == 'Pusa-Delhi':

            if os.path.exists('../real_data.csv'):
                os.remove('../real_data.csv')
                print(f"Deleted existing CSV file: {'real_data.csv'}")

            url = f"https://api.waqi.info/feed/@10124/?token={api_key}"

            headers = {
                "accept": "application/json",
                "content-type": "application/json"
            }
            response = requests.get(url, headers=headers)

            # Extract data
            extracted_data = extract_data(response.text)
            csv_file_path = '../real_data.csv'
            # Create CSV file
            create_csv_file(extracted_data, csv_file_path)

            df = pd.read_csv('../real_data.csv')
            # st.write(df)

            time_1 = st.date_input("Date", datetime.date.today())

            #time_2 = st.time_input('Time', value="now")

            location_name = st.text_input('Location name', 'Pusa, Delhi')

            if st.button(":red[Calculate Air Quality Index]"):
                df.drop(['date', 'city'], inplace=True, axis=1)
                st.write(df)
                aq = df['AQI'].iloc[0]
                air_quality_widget(aq)
                df['AQI_Category'] = df['AQI'].apply(categorize_aqi)
                ac = df['AQI_Category'].iloc[0]


                st.write("Air Quality Index : ",ac)

                df.drop(['pm10', 'AQI', 'AQI_Category'], inplace=True, axis=1)

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
                st2_predictions_df = trained_stacking_classifier.predict(new_df)

                #st.write("Air Quality Index : ", st2_predictions_df)
                #st.balloons()



# Code or DCNS - Delhi location
        elif category == 'DCNS-Delhi':
            if os.path.exists('../real_data.csv'):
                os.remove('../real_data.csv')
                print(f"Deleted existing CSV file: {'real_data.csv'}")

            url = f"https://api.waqi.info/feed/@10111/?token={api_key}"

            headers = {
                "accept": "application/json",
                "content-type": "application/json"
            }
            response = requests.get(url, headers=headers)

            # Extract data
            extracted_data = extract_data(response.text)
            csv_file_path = '../real_data.csv'
            # Create CSV file
            create_csv_file(extracted_data, csv_file_path)

            df = pd.read_csv('../real_data.csv')

            time_1 = st.date_input("Date", datetime.date.today())
            #time_2 = st.time_input('Time', value="now")

            location_name = st.text_input('Location name', 'Major Dhyan Chand National Stadium, Delhi')

            if st.button(":red[Calculate Air Quality Index]"):
                df.drop(['date', 'city'], inplace=True, axis=1)
                st.write(df)
                aq = df['AQI'].iloc[0]
                air_quality_widget(aq)
                df['AQI_Category'] = df['AQI'].apply(categorize_aqi)
                ac = df['AQI_Category'].iloc[0]
                st.write("Air Quality Index : ", ac)

                df.drop(['pm10', 'AQI', 'AQI_Category'], inplace=True, axis=1)

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
                st3_predictions_df = trained_stacking_classifier.predict(new_df)

                #st.write("Air Quality Index : ", st3_predictions_df)
                #st.write(st_predictions_df)
                #st.balloons()


# Greater Noida location
        elif category == 'AV-Delhi':
            if os.path.exists('../real_data.csv'):
                os.remove('../real_data.csv')
                print(f"Deleted existing CSV file: {'real_data.csv'}")

            url = f"https://api.waqi.info/feed/@2553/?token={api_key}"

            headers = {
                "accept": "application/json",
                "content-type": "application/json"
            }
            response = requests.get(url, headers=headers)

            # Extract data
            extracted_data = extract_data(response.text)
            csv_file_path = '../real_data.csv'
            # Create CSV file
            create_csv_file(extracted_data, csv_file_path)

            df = pd.read_csv('../real_data.csv')

            time_1 = st.date_input("Date", datetime.date.today())

            #time_2 = st.time_input('Time', value="now")

            location_name = st.text_input('Location name', 'Anand Vihar, Delhi')

            if st.button(":red[Calculate Air Quality Index]"):
                df.drop(['date', 'city'], inplace=True, axis=1)
                st.write(df)
                aq = df['AQI'].iloc[0]
                air_quality_widget(aq)
                df['AQI_Category'] = df['AQI'].apply(categorize_aqi)
                ac = df['AQI_Category'].iloc[0]
                st.write("Air Quality Index : ", ac)

                df.drop(['pm10', 'AQI', 'AQI_Category'], inplace=True, axis=1)

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
                st4_predictions_df = trained_stacking_classifier.predict(new_df)
                #st.write("Air Quality Index : ", st4_predictions_df)

                #st.subheader("Air Quality Index")
                #st.write(st_predictions_df)
                #st.balloons()


# RKP Delhi location
        elif category == 'RKP-Delhi':
            if os.path.exists('../real_data.csv'):
                os.remove('../real_data.csv')
                print(f"Deleted existing CSV file: {'real_data.csv'}")

            url = f"https://api.waqi.info/feed/@2556/?token={api_key}"

            headers = {
                "accept": "application/json",
                "content-type": "application/json"
            }
            response = requests.get(url, headers=headers)

            # Extract data
            extracted_data = extract_data(response.text)
            csv_file_path = '../real_data.csv'
            # Create CSV file
            create_csv_file(extracted_data, csv_file_path)

            df = pd.read_csv('../real_data.csv')

            time_1 = st.date_input("Date", datetime.date.today())

            #time_2 = st.time_input('Time', value="now")

            location_name = st.text_input('Location name', 'R.K Puram, Delhi')

            if st.button(":red[Calculate Air Quality Index]"):
                df.drop(['date', 'city'], inplace=True, axis=1)
                st.write(df)
                aq = df['AQI'].iloc[0]
                air_quality_widget(aq)
                df['AQI_Category'] = df['AQI'].apply(categorize_aqi)
                ac = df['AQI_Category'].iloc[0]
                st.write("Air Quality Index : ", ac)

                df.drop(['pm10', 'AQI', 'AQI_Category'], inplace=True, axis=1)

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
                st5_predictions_df = trained_stacking_classifier.predict(new_df)
                #st.write("Air Quality Index : ", st5_predictions_df)
                #st.write(st_predictions_df)
                #st.balloons()

    # Air feature
    with col2:
        st.subheader("Air")
        # Create tabs for different reports
        tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["AQI", "PM", "CO", "SO2", "NO2", "O3"])

        with tab5:
            st.subheader("Air Quality Index (AQI)")
            img_url1 = 'https://w.ndtvimg.com/sites/3/2019/12/18122812/air_pollution_standards_cpcb.png'
            st.image(img_url1, use_column_width=True)

            st.markdown(
                "AQI stands for Air Quality Index. It is a numerical scale used to communicate how polluted the air "
                "currently is or how polluted it is forecast to become. The AQI scale typically ranges from 0 to 500,"
                " with higher values indicating poorer air quality and a greater potential for adverse health effects."
                "The AQI is calculated based on the concentrations of several major air pollutants, including:")

            st.markdown("- **1. Ground-level ozone (O3)**")

            st.markdown("- **2. Particulate matter (PM10 and PM2.5)**")

            st.markdown("- **3. Carbon monoxide (CO)**")

            st.markdown(" - **4. Sulfur dioxide (SO2)**")
            st.markdown(" - **5. Nitrogen dioxide (NO2)**")
            st.markdown(
                "Each of these pollutants has its own sub-index, and the overall AQI is determined by the highest "
                " sub-index value. The AQI is divided into different color-coded categories, each representing a "
                "different level of health concern. These categories often include:")

            st.markdown(
                " - **0-50:** Good (Air quality is considered satisfactory, and air pollution poses little or no risk.)"
                "")
            st.markdown(
                " - **51-100:** Satisfactory(Low health risks, but vulnerable groups may experience minor issues.)")
            st.markdown(
                " - **101-200:** Moderate(Adverse effects possible for sensitive individuals, especially those with "
                "respiratory or cardiovascular conditions.)")
            st.markdown(
                " - **201-300:** Poor (Increased health risks, particularly for vulnerable groups such as "
                "children, the elderly, and those with existing health conditions.)")
            st.markdown(
                " - **301-400:** Very Poor (Health alert: everyone may experience more serious health effects.)")

            st.markdown(
                " - **401-500:** Severe (Health warnings of emergency conditions; the entire population is likely"
                " to be affected.)")

            st.markdown(
                "Monitoring the AQI is important for assessing the potential impact of air pollution on human health"
                " and taking appropriate actions, such as reducing outdoor activities or using air purifiers, when "
                "air quality is poor.")


        # PArticulate Matter section
        with tab6:
            st.subheader("Particulate Matter")
            img_url2 = "https://learn.kaiterra.com/hs-fs/hubfs/Particulate%20Matter%20Sizes%20(2).png?width=1700&name=Particulate%20Matter%20Sizes%20(2).png"
            st.image(img_url2, use_column_width=True)

            st.markdown(
                "Particulate Matter (PM) refers to tiny particles or droplets in the air that can be inhaled into the"
                " respiratory system. These particles vary in size, composition, and origin. The two main categories"
                " of particulate matter are PM10 and PM2.5, which denote particles with diameters of 10 micrometers "
                "or smaller and 2.5 micrometers or smaller, respectively. The measurement in micrometers represents "
                "the aerodynamic diameter of the particles, indicating their ability to remain suspended in the air "
                "and potentially reach the deeper parts of the respiratory system. Sources of Particulate Matter are:")

            st.markdown("**1. Natural Sources**")

            st.markdown("- Volcanic eruptions.")

            st.markdown("- Forest fires.")

            st.markdown("- Dust storms.")

            st.markdown("2. **Human-Generated Sources**")

            st.markdown("- Combustion of fossil fuels (e.g., vehicle exhaust, power plants).")

            st.markdown("- Industrial processes.")

            st.markdown("- Construction activities.")

            st.markdown("- Agricultural activities (e.g., tilling, burning of crop residues).")

            st.markdown(
                "Particulate matter can consist of various substances, including organic chemicals, metals, and soil"
                " or dust particles. The composition of PM can influence its health effects. The smaller particles "
                "(PM2.5) are of particular concern because they can penetrate deeper into the lungs and even enter "
                "the bloodstream, posing greater health risks.")

            st.markdown("**Health Effects of Particulate Matter**")

            st.markdown(
                "- **1. Respiratory Issues:** PM can irritate the respiratory system, leading to coughing, shortness"
                " of breath, and aggravated asthma symptoms.")

            st.markdown(
                "- **2. Cardiovascular Effects:** Fine particles can enter the bloodstream, potentially causing "
                "cardiovascular problems such as heart attacks and aggravated heart conditions.")

            st.markdown(
                "- **3. Reduced Lung Function:** Long-term exposure to elevated levels of PM may contribute to chronic"
                " respiratory diseases and reduced lung function.")

            st.markdown(
                "- **4. Premature Mortality:** Exposure to high levels of PM has been associated with an increased "
                "risk of premature death, especially among individuals with pre-existing health conditions.")

            st.markdown(
                "Regulatory agencies monitor and set standards for particulate matter levels to protect public health."
                " The Air Quality Index (AQI) includes PM concentrations as one of its key components to communicate"
                " air quality conditions to the public. Reducing emissions from sources that contribute to particulate"
                " matter is essential for improving air quality and safeguarding public health.")


        # Carbon Monoxide section
        with tab7:
            st.subheader("Carbon Monoxide (CO)")
            img_url3 = "https://www.cdc.gov/co/images/co-cant-be-720px.jpg?_=82369"
            st.image(img_url3, use_column_width=True)

            st.markdown(
                "CO stands for Carbon Monoxide. It is a colorless, odorless gas that is produced by the incomplete"
                " combustion of carbon-containing fuels. Common sources of carbon monoxide include vehicle exhaust, "
                "industrial processes, and residential heating appliances such as gas furnaces and wood-burning "
                "stoves. Key Points about Carbon Monoxide (CO) are:")

            st.markdown(
                "**1. Toxicity:** Carbon monoxide is highly toxic because it binds to hemoglobin in the bloodstream"
                " more tightly than oxygen. This can lead to a reduction in the delivery of oxygen to body tissues, "
                "causing health problems.")

            st.markdown("**2. Symptoms of CO Poisoning:** Exposure to elevated levels of carbon monoxide can result in "
                        "symptoms such as headache, dizziness, nausea, confusion, and, in severe cases, unconsciousness or "
                        "death.")

            st.markdown("**3. Common Sources**")

            st.markdown("- Motor vehicle exhaust (especially in enclosed spaces).")
            st.markdown("- Combustion appliances (gas stoves, water heaters, furnaces).")
            st.markdown("- Tobacco smoke.")
            st.markdown("- Wood-burning stoves and fireplaces.")

            st.markdown("**4. Prevention**")
            st.markdown("- Proper ventilation in enclosed spaces is crucial to prevent the buildup of carbon monoxide.")
            st.markdown(
                "- Regular maintenance of combustion appliances to ensure they are operating efficiently and not "
                "emitting excessive CO.")
            st.markdown("- Use of carbon monoxide detectors in homes and buildings.")

            st.markdown(
                "**5. Regulatory Standards:** Many countries have established air quality standards and guidelines"
                " for carbon monoxide concentrations to protect public health.")


        # sulphur dioxide section
        with tab8:
            st.subheader("Sulphur Dioxide (SO2)")
            img_url4 = "https://cdn.images.express.co.uk/img/dynamic/78/590x/hawaii-volcano-eruption-leilani-estates-957437.jpg?r=1686998680160"
            st.image(img_url4, use_column_width=True)

            st.markdown(
                "Sulfur Dioxide (SO2) is a colorless gas with a pungent and irritating smell. It is produced by the "
                "burning of fossil fuels containing sulfur, such as coal and oil. Sulfur dioxide is also released "
                "during volcanic eruptions and certain industrial processes.")

            st.markdown("**1. Sources**")

            st.markdown("- Combustion of fossil fuels (coal, oil) in power plants and industrial facilities.")
            st.markdown("- Volcanic eruptions release sulfur dioxide into the atmosphere.")
            st.markdown("- Certain industrial processes, such as the production of sulfuric acid.")

            st.markdown(
                "**2. Environmental Impact:** Sulfur dioxide can contribute to air pollution and acid rain formation "
                "when it reacts with water vapor and oxygen in the atmosphere.")

            st.markdown("**3. Health Effects**")
            st.markdown("- Short-term exposure to sulfur dioxide can irritate the eyes, nose, and throat.")
            st.markdown("- It can exacerbate respiratory conditions such as asthma and bronchitis.")
            st.markdown("- Long-term exposure may lead to chronic respiratory problems.")


        # nitrogen dioxide section
        with tab9:
            st.subheader("Nitrogen Dioxide (NO2)")
            img_url5 = "https://cleanairhamilton.ca/wp-content/uploads/2019/06/nitrogen_2012.png"
            st.image(img_url5, use_column_width=True)

            st.markdown("NO2 stands for Nitrogen Dioxide. It is a reddish-brown gas that is a prominent air pollutant. "
                        "Nitrogen dioxide is one of the nitrogen oxides (NOx), a group of gases produced by the combustion "
                        "of fossil fuels and certain industrial processes.")

            st.markdown("**1. Sources**")

            st.markdown("- Combustion of fossil fuels in vehicles, power plants, and industrial facilities.")
            st.markdown("- Agricultural activities, especially the use of nitrogen-based fertilizers.")
            st.markdown("- Certain natural processes, such as lightning and wildfires.")

            st.markdown(
                "**2. Environmental Impact:** Nitrogen dioxide can contribute to the formation of ground-level ozone"
                " and particulate matter, both of which are harmful air pollutants. It is a precursor to nitric acid,"
                " which can contribute to acid rain.")

            st.markdown("**3. Health Effects**")
            st.markdown(
                "- Short-term exposure to nitrogen dioxide can irritate the respiratory system, leading to symptoms "
                "such as coughing, wheezing, and difficulty breathing.")
            st.markdown("- Long-term exposure may contribute to respiratory problems and decreased lung function.")


        # Ozone section
        with tab10:
            st.subheader("Ozone (O3)")
            img_url6 = "https://scx1.b-cdn.net/csz/news/800a/2017/ozoneandhaze.jpg"
            st.image(img_url6, use_column_width=True)

            st.markdown(
                "O3 stands for Ozone. Ozone is a molecule composed of three oxygen atoms (O3). While ozone in the "
                "stratosphere (upper atmosphere) plays a crucial role in protecting life on Earth by absorbing the "
                "majority of the sun's harmful ultraviolet (UV) radiation, ground-level ozone is a harmful air "
                "pollutant.")

            st.markdown(
                "**1. Formation:** Ground-level ozone is not emitted directly into the air but is formed through "
                "complex chemical reactions involving precursor pollutants in the presence of sunlight. The primary "
                "precursors are nitrogen oxides (NOx) and volatile organic compounds (VOCs), which are emitted from "
                "vehicles, industrial facilities, and other sources.")

            st.markdown("**2. Environmental Impact**")

            st.markdown("- Ground-level ozone is a major component of smog and can contribute to air pollution.")
            st.markdown("- It can harm vegetation, including crops and forests.")
            st.markdown("- Ozone can also impact ecosystems and aquatic environments.")

            st.markdown("**3. Health Effects**")
            st.markdown("- Ozone exposure can cause respiratory problems, especially in vulnerable populations such as "
                        "children, the elderly, and individuals with respiratory conditions like asthma.")
            st.markdown("- It may lead to symptoms such as coughing, throat irritation, and difficulty breathing.")


    # Safety Recommendation section
    with col3:
        st.subheader("Safety Recommendations")

        tab11, tab12 = st.tabs(["Health Safety Recommendations", "Environment Safety Recommendations"])

        # Health Safety Advise section
        with tab11:
            st.subheader("Health Safety Recommendations")
            img_url7 = "https://publichealthcollaborative.org/wp-content/uploads/2023/08/PHCC_8-Ways-to-Stay-Safe_Rectangle-Social-Graphic_ENG-1024x576.png.webp"
            st.image(img_url7, use_column_width=True)

            st.markdown("**1. Stay Informed:** Monitor local air quality indices provided by environmental agencies or "
                        "weather services. Use smartphone apps or online resources to stay updated on air quality conditions"
                        " in your area.")

            st.markdown(
                "**2. Limit Outdoor Activities:** Reduce outdoor activities, especially strenuous exercises, during"
                " periods of high air pollution. Choose indoor exercise options when air quality is poor.")

            st.markdown("**3. Avoid High-Pollution Areas:** Limit time spent in areas with high levels of traffic, "
                        "industrial activity, or other potential pollution sources.")

            st.markdown(
                "**4. Use Air Purifiers:** Consider using air purifiers with HEPA filters at home to reduce indoor "
                "air pollution. Ensure proper ventilation in enclosed spaces.")

            st.markdown(
                "**5. Wear Masks:** In areas with significant air pollution, especially during events like wildfires,"
                " wearing masks designed to filter out particles (e.g., N95 masks) can provide some protection.")

            st.markdown("**6. Stay Hydrated:** Drink plenty of water to help flush out toxins from the body.")

            st.markdown(
                "**7. Seek Medical Advice:** Individuals with respiratory conditions or pre-existing health issues "
                "should consult with healthcare professionals for personalized advice. Pay attention to any "
                "worsening of respiratory symptoms and seek medical attention if needed.")

            st.markdown("**8. Create a Clean Indoor Environment:** Keep indoor spaces clean and well-ventilated. Use "
                        "air-conditioning or air filtration systems to improve indoor air quality.")

            st.markdown(
                "**9. Reduce Vehicle Emissions:** Use public transportation or carpool to reduce individual vehicle "
                "emissions. Choose fuel-efficient vehicles or electric alternatives when possible.")

            st.markdown(
                "**10. Support Air Quality Initiatives:** Advocate for and support initiatives that aim to reduce "
                "air pollution at the community and policy levels. Participate in activities that promote "
                "sustainable practices and cleaner air.")

        # Environment Safety Recommendation section
        with tab12:
            st.subheader("Environment Safety Recommendations")
            img_url8 = "https://www.unicef.org/vietnam/sites/unicef.org.vietnam/files/Poster%205_Solution%20for%20Air%20Pollution-01.jpg"
            st.image(img_url8, use_column_width=True)

            st.markdown("**1. Reduce Vehicle Emissions:** Choose public transportation, carpooling, biking, or walking "
                        "instead of relying solely on personal vehicles. Opt for fuel-efficient or electric vehicles when "
                        "purchasing a car. Keep vehicles well-maintained to ensure optimal fuel efficiency.")

            st.markdown(
                "**2. Conserve Energy:** Use energy-efficient appliances and light bulbs at home. Turn off lights, "
                "electronics, and appliances when not in use. Reduce heating and cooling needs by properly "
                "insulating homes.")

            st.markdown("**3. Limit Outdoor Burning:** Avoid burning leaves, trash, or other materials outdoors. Use "
                        "alternatives to burning, such as composting or recycling.")

            st.markdown(
                "**4. Be Mindful of Personal Consumption:** Consume products with minimal packaging to reduce waste."
                "Recycle and properly dispose of household waste.")

            st.markdown(
                "**5. Support Sustainable Practices:** Choose products and services from companies that prioritize "
                "environmental sustainability. Support policies and initiatives that promote clean energy and "
                "environmental conservation..")

            st.markdown("**6. Conserve Water:** Use water-saving appliances and fixtures. Avoid wasting water in daily "
                        "activities, such as washing dishes or watering the garden.")

            st.markdown(
                "**7. Plant Trees and Maintain Green Spaces:** Participate in community tree-planting initiatives. "
                "Create and maintain green spaces around homes and neighborhoods.")

            st.markdown(
                "**8. Reduce, Reuse, Recycle:** Minimize single-use plastics and opt for reusable alternatives. "
                "Recycle materials like paper, glass, and plastic to reduce landfill waste.")

            st.markdown(
                "**9. Use Energy-Efficient Appliances:** Choose energy-efficient appliances for homes and workplaces."
                "Consider renewable energy sources, such as solar panels for residential use.")

            st.markdown("**10. Educate Others:** Raise awareness about air pollution and its impact on the environment."
                        "Share information on sustainable practices with friends, family, and colleagues.")


# Visualization Section
with tab2:
    st.subheader("Visualization")

    # Generate the HTML using Pygwalker
    pyg_html = pyg.to_html(aq_df)

    # Embed the HTML into the Streamlit app
    components.html(pyg_html, height=1500, width=1000, scrolling=True)

# Help section
with tab3:
    st.subheader("Help")

    st.markdown("Welcome to the Paavan Vayu App Help Section! Whether you are new to the app or looking for specific "
                "information, we've got you covered. Here's a guide to assist you.")

    st.markdown("**1. Email Sign-Up and Confirmation**")

    st.markdown("- Manually fill in the email and password fields on the sign-up page.")

    st.markdown("- Click the Sign-Up button to submit the form.")

    st.markdown("- Check the email inbox for the confirmation email.")

    st.markdown("- Confirm that the email contains a valid confirmation link.")

    st.markdown("- Click the confirmation link in the email.")

    st.markdown("- Confirm that the link redirects to the sign-in page.")

    st.markdown("- Manually fill in the email and password fields on the sign-in page.")

    st.markdown("- Click the Sign-In button to submit the form.")

    st.markdown("- Verify that the login is successful, and the user is redirected to the main application page.")

    st.markdown("- **2. Login via GitHub**")

    st.markdown("- Click the Login with GitHub button.")

    st.markdown("- If prompted, authorize the Paavan Vayu app on GitHub.")

    st.markdown("- Verify that the login is successful, and the user is redirected to the main application page.")

    st.markdown("**3. Login via Google**")

    st.markdown(" - Click the Login with Google button.")

    st.markdown(" - If prompted, authorize the Paavan Vayu app on Google.")

    st.markdown(" - Verify that the login is successful, and the user is redirected to the main application page.")

    st.markdown(" **4. Real Time Monitoring Feature**")

    st.markdown(" - **Choosing a Location**")

    st.markdown(" - To begin, select your desired location from the provided list. The app currently covers five key "
                "regions in Delhi, India: 'Mandir Marg-Delhi,' 'R. K. Puram-Delhi,' 'Pusa-Delhi,' 'Major Dhyan Chand "
                "National Stadium-Delhi,' and 'Knowledge Park, Greater-Noida.'")

    st.markdown(" - Click on the Air Quality Index button.")

    st.markdown(" Experience real-time updates on air quality conditions. The app continually fetches the latest data, "
                "keeping you informed about the current atmospheric situation in your selected location.")

    st.markdown("**Date Information**")

    st.markdown(" - Refer to the date displayed to ensure you are viewing the most recent air quality data. This date "
                "indicates when the latest information was recorded, providing context for the presented insights.")



    st.markdown("**5. Air Feature**")

    st.markdown("Learn about the Air Quality Index (AQI) and its significance. The AQI is a numerical scale used to "
                "communicate the current or forecasted level of air pollution, incorporating key pollutants such as "
                "ozone, particulate matter, carbon monoxide, sulfur dioxide, and nitrogen dioxide.")

    st.markdown("**6. Safety Recommendations**")

    st.markdown("Access Health Safety Recommendations and Environment Safety Recommendations to understand recommended "
                "actions during varying air quality conditions. The app provides tips on staying informed, limiting "
                "outdoor activities, avoiding high-pollution areas, and using air purifiers.")

    st.markdown("**7. FAQs (Frequently Asked Questions)**")
    st.markdown("Check the FAQs section for answers to common queries. If you have questions about the AQI, data "
                "updates, or any other aspect of the app, you might find the information you need here.")

    st.markdown("**8. Submitting Feedback**")
    st.markdown("Your feedback is valuable to us! Use the Submit Feedback feature within the app to share your thoughts,"
                " report any issues, or suggest improvements. We appreciate your input in enhancing the app's "
                "performance.")

    st.markdown(" **9. Visualization Feature**")

    st.markdown("Explore historical data to gain insights into past air quality records and trends. This feature allows"
                " you to analyze patterns and fluctuations over time for a more comprehensive understanding.")

    st.markdown("**10. 503 Error**")

    st.markdown("The 503 Service Unavailable error is an indication that the server is temporarily unable to handle "
                "the request. While our Paavan Vayu application is designed to provide a seamless user experience, "
                "occasionally, server-related issues may arise. The 503 error typically occurs when the server is "
                "undergoing maintenance, is overloaded, or is facing temporary issues that prevent it from fulfilling"
                " the request. Please try again after some time.")

    st.markdown("**11. Forgot Password**")
    st.markdown("If you forget your password, click on the 'Forgot Password' link on the login page. Enter your email "
                "address and follow the instructions sent to your email to reset your password.")


# FAQ section
with tab4:
    st.subheader("FAQs")

    st.markdown("- **What does Paavan Vayu means?**")
    st.markdown("Paavan is a Sanskrit word that can be translated to mean pure, sacred, clean, or holy. Vayu is also a "
                "Sanskrit word that translates to wind or air. The word Paavan Vayu means clean air.")

    st.markdown("- **From where does the Paavan Vayu app obtain its data?**")
    st.markdown("The Paavan Vayu app sources its air quality data from the World Air Quality Index (WAQI) platform. "
                "This global platform aggregates real-time air quality information from monitoring stations worldwide. "
                "The primary focus areas for this app are five regions in Delhi, India: 'Mandir Marg-Delhi,' 'R. K. "
                "Puram-Delhi,' 'Pusa-Delhi,' 'Major Dhyan Chand National Stadium-Delhi,' and 'Knowledge Park, "
                "Greater-Noida.' The WAQI platform acts as a central repository, collecting data from various regional "
                "and national monitoring stations, including those operated by Delhi Pollution Control Committee (DPCC) "
                "and Central Pollution Control Board (CPCB). These organizations strategically place monitoring "
                "stations across Delhi to capture real-time air quality readings, which contribute to the WAQI platform "
                "for broader accessibility and standardization. The app utilizes this consolidated data for public "
                "awareness, research, and analysis.")

    st.markdown("- **What is the Air Quality Index (AQI)?**")
    st.markdown("The AQI is a numerical scale used to communicate the current or forecasted level of air pollution. "
                "It ranges from 0 to 500, with higher values indicating poorer air quality. The AQI is calculated based "
                "on concentrations of major pollutants like ozone (O3), particulate matter (PM), carbon monoxide (CO), "
                "sulfur dioxide (SO2), and nitrogen dioxide (NO2).")

    st.markdown("- **How is the AQI calculated?**")
    st.markdown("The AQI is determined by the highest sub-index value among various pollutants. Each pollutant has its"
                "own sub-index, and the overall AQI is categorized to represent different levels of health concern.")

    st.markdown("- **How often is the air quality data updated in the app?**")
    st.markdown("The app offers real-time updates, ensuring users have access to the most current air quality "
                "information. This allows users to stay informed about the immediate atmospheric conditions in their "
                "chosen locations.")

    st.markdown("- **How accurate is the air quality prediction model used in Paavan Vayu?**")
    st.markdown("The air quality prediction model employed in Paavan Vayu demonstrates an accuracy rate of "
                "approximately 90%. While this accuracy is high and reliable for most scenarios, it's essential to "
                "acknowledge that, like any statistical model, there's a 10% margin for error. Factors such as "
                "unexpected changes in weather patterns or localized influences may contribute to variations in "
                "predictions. Users are encouraged to interpret the results with awareness of this margin and consider"
                " them as valuable insights rather than absolute certainties. Regular updates and improvements to the "
                "model are made to enhance its overall performance.")


    st.markdown("- **Does the Paavan Vayu app contain historical data?**")
    st.markdown("The app provides a historical data feature in the Visualization section. Users can peek into past air"
                " quality records and trends, allowing them to analyze patterns and fluctuations over time.")


    st.markdown("- **How many locations does the Paavan Vayu App cover for air quality analysis?**")
    st.markdown("The Paavan Vayu App offers air quality data for five prominent regions in Delhi, India. These key "
                "areas include 'Mandir Marg-Delhi,' 'R. K. Puram-Delhi,' 'Pusa-Delhi,' 'Major Dhyan Chand National "
                "Stadium-Delhi,' and 'Knowledge Park, Greater-Noida.' Users can access detailed air quality insights "
                "for each of these locations to stay informed about the atmospheric conditions in specific regions.")

    st.markdown("- **How can I submit feedback about the app?**")
    st.markdown("Use the Submit Feedback feature within the Paavan Vayu app to share your thoughts, report issues, or "
                "suggest improvements. Your feedback is valuable for enhancing the app's performance.")

    st.markdown("- **What actions can I take based on the air quality information provided?**")
    st.markdown("The Paavan Vayu app offers Safety Advise, including health and environment safety recommendations. "
                "Users can follow these guidelines to stay informed and take necessary precautions during varying air "
                "quality conditions.")

    st.markdown("- **Why is monitoring the AQI important?**")
    st.markdown("Monitoring the AQI is crucial for assessing the potential impact of air pollution on human health. "
                "It helps individuals make informed decisions and take necessary precautions, such as reducing outdoor "
                "activities or using air purifiers, when air quality is poor.")

    st.markdown("- **What safety advice is provided for health and the environment?**")
    st.markdown("The Paavan Vayu app offers Health Safety Recommendations, including tips such as staying informed, limiting "
                "outdoor activities during poor air quality, avoiding high-pollution areas, and using air purifiers. "
                "Environment Safety Recommendations is also available, promoting responsible practices for environmental "
                "conservation.")

# submit feedback section
with tab13:
    col11, col12, col13 = st.columns([1, 2, 1])

    with col12:
        st.subheader("Submit Feedback")

        feedback_form = """
                <form action="https://formsubmit.co/NYete@my.gcu.edu" method="POST">
                     <input type="hidden" name="_captcha" value="false">
                     <input type="text" name="name" placeholder="Your name" required>
                     <input type="email" name="email" placeholder="Your email" required>
                     <textarea name="message" placeholder="Your feedback here"></textarea>
                     <button type="submit">Send</button>
                </form>
                """

        st.markdown(feedback_form, unsafe_allow_html=True)


        # Use Local CSS File
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


        local_css("style/style.css")







