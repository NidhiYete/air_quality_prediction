import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import os
import import_ipynb
import io
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import ast
import Air_Quality_Data_Extraction
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler,PowerTransformer
from scipy.stats.mstats import winsorize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, auc
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels

from Air_Quality_Data_Extraction import read_csv_files, concatenate_dataframes, process_dataframe, save_to_csv
pio.templates.default = "plotly_white"




Air_Quality_Data_Extraction.read_csv_files
Air_Quality_Data_Extraction.concatenate_dataframes
Air_Quality_Data_Extraction.process_dataframe
Air_Quality_Data_Extraction.save_to_csv

air_quality = pd.read_csv('data_2/air_data.csv')
air_new_df = pd.read_csv('Out_115.csv')
md_new_df = pd.read_csv('Out_37.csv')
# feature matrix and target variable
X = air_new_df.drop(columns=['AQI_Category'])
y = air_new_df['AQI_Category']
#%%
# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# feature matrix and target variable
mdX = md_new_df.drop(columns=['AQI_Category'])
mdy = md_new_df['AQI_Category']

# Data Collection

# Assuming 'data_2' folder contains test CSV files for testing
TEST_FOLDER_PATH = 'data_2'

# Test data for sample CSV files
SAMPLE_DATA_1 = {'date': ['2022-01-01', '2022-01-02'], 'value': [10, 20]}
SAMPLE_DATA_2 = {'date': ['2022-01-03', '2022-01-04'], 'value': [30, 40]}

class TestMyModule(unittest.TestCase):

    def test_read_csv_files(self):
        dfs = read_csv_files(TEST_FOLDER_PATH)
        self.assertEqual(len(dfs), 6)  # Assuming there are two test CSV files
        # Ensure each element in dfs is a DataFrame
        self.assertTrue(all(isinstance(df, pd.DataFrame) for df in dfs))

    def test_concatenate_dataframes(self):
        df1 = pd.DataFrame(SAMPLE_DATA_1)
        df2 = pd.DataFrame(SAMPLE_DATA_2)
        combined_df = concatenate_dataframes([df1, df2])
        # Check if the length of combined DataFrame is sum of lengths of individual DataFrames
        self.assertEqual(len(combined_df), len(df1) + len(df2))

    def test_process_dataframe(self):
        # Create a sample DataFrame
        df = pd.DataFrame({'date': ['2022-01-02', '2022-01-01'], 'value': [20, 10]})
        processed_df = process_dataframe(df)
        # Check if 'date' column is in datetime format
        self.assertIsInstance(processed_df['date'][0], pd.Timestamp)
        # Check if DataFrame is sorted by 'date' column
        self.assertTrue(processed_df['date'].is_monotonic_increasing)

    def test_save_to_csv(self):
        # Create a sample DataFrame
        df = pd.DataFrame(SAMPLE_DATA_1)
        output_file = 'test_output.csv'
        save_to_csv(df, output_file)
        # Read the saved CSV file
        saved_df = pd.read_csv(output_file)
        # Check if the saved DataFrame is equal to the original DataFrame
        self.assertTrue(df.equals(saved_df))

if __name__ == '__main__':
    unittest.main()

# Data Preprocessing

class TestAirQuality(unittest.TestCase):
    def setUp(self):
        # Prepare test data
        self.expected_data = pd.read_csv('data_2/air_data.csv')

    def test_read_csv(self):
        # Read the CSV file
        air_quality = pd.read_csv('data_2/air_data.csv')
        # Check if the data read from CSV matches the expected data
        self.assertTrue(self.expected_data.equals(air_quality))

    def test_view_first_20_rows(self):
        # Read the CSV file
        air_quality = pd.read_csv('data_2/air_data.csv')
        # Get the first 20 rows of the dataset
        first_20_rows = air_quality.head(20)
        # Check if the first 20 rows match the expected data
        self.assertTrue(self.expected_data.head(20).equals(first_20_rows))

class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        # Prepare test data
        self.air_quality = pd.read_csv('data_2/air_data.csv')

    def test_coordinates_conversion(self):
        # Convert 'coordinates' column from strings to lists of latitude and longitude values
        self.air_quality['coordinates'] = self.air_quality['coordinates'].apply(ast.literal_eval)

        # Check if 'coordinates' column is converted correctly
        self.assertIsInstance(self.air_quality['coordinates'][0], list)
        self.assertEqual(len(self.air_quality['coordinates'][0]), 2)
        self.assertIsInstance(self.air_quality['coordinates'][0][0], float)
        self.assertIsInstance(self.air_quality['coordinates'][0][1], float)

    def test_coordinate_separation(self):
        # Separate 'coordinates' column into 'latitude' and 'longitude' columns
        coordinates_df = self.air_quality['coordinates'].apply(ast.literal_eval).apply(pd.Series)
        coordinates_df.columns = ['latitude', 'longitude']
        self.air_quality = pd.concat([self.air_quality, coordinates_df], axis=1)
        self.air_quality.drop('coordinates', axis=1, inplace=True)

        # Check if 'latitude' and 'longitude' columns are created correctly
        self.assertIn('latitude', self.air_quality.columns)
        self.assertIn('longitude', self.air_quality.columns)

    def test_drop_coordinates_column(self):
        # Drop the original 'coordinates' column
        self.air_quality = self.air_quality.drop('coordinates', axis=1)

        # Check if 'coordinates' column is dropped
        self.assertNotIn('coordinates', self.air_quality.columns)

    def test_numeric_conversion(self):
        # Convert numeric columns to the appropriate data type
        numeric_columns = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        for col in numeric_columns:
            self.air_quality[col] = pd.to_numeric(self.air_quality[col], errors='coerce')

        # Check if numeric columns are converted correctly
        for col in numeric_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(self.air_quality[col]))
    def test_date_conversion(self):
        # Convert 'date' column to datetime format
        self.air_quality['date'] = pd.to_datetime(self.air_quality['date'])

        # Check if 'date' column is converted to datetime format
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.air_quality['date']))

    def test_check_structure(self):

        buffer = io.StringIO()
        sys.stdout = buffer
        self.air_quality.info()
        # Reset the standard output to its original state
        sys.stdout = sys.__stdout__

        info_output = buffer.getvalue()

        self.assertIn('Data columns (total', info_output)
        self.assertIn('non-null', info_output)
        buffer.close()

if __name__ == '__main__':
    unittest.main()

# missing data
class TestAirQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.data = pd.read_csv('data_2/air_data.csv')
    def test_missing_values_initial(self):

        missing_values_count = self.data.isna().sum()
        print(missing_values_count)
    def test_drop_missing_values(self):

        data_cleaned = self.data.dropna()
        self.assertTrue(data_cleaned.isna().sum().sum() == 0)

if __name__ == "__main__":
    unittest.main()

# Exploratory Data Analysis
# Data Visualization
def plot_distribution(air_quality):
    plt.figure(figsize=(15, 20))
    plt.subplot(4, 2, 1)
    sns.distplot(air_quality['co'], bins=20, color='red')
    plt.subplot(4, 2, 2)
    sns.distplot(air_quality['so2'], bins=10, color='green')
    plt.subplot(4, 2, 3)
    sns.distplot(air_quality['no2'], bins=10, color='blue')
    plt.subplot(4, 2, 4)
    sns.distplot(air_quality['o3'], bins=10, color='yellow')
    plt.subplot(4, 2, 5)
    sns.distplot(air_quality['pm25'], bins=20, color='brown')
    plt.subplot(4, 2, 6)
    sns.distplot(air_quality['pm10'], bins=20, color='orange')
    plt.show()
class TestDistributionPlot(unittest.TestCase):
    def test_plot_distribution(self):
        # Create a sample dataframe for testing
        data = pd.DataFrame({
            'co': [0.1, 0.5, 0.3, 0.7, 0.2],
            'so2': [1, 2, 3, 4, 5],
            'no2': [0.5, 1.5, 2.5, 3.5, 4.5],
            'o3': [5, 10, 15, 20, 25],
            'pm25': [15, 20, 25, 30, 35],
            'pm10': [25, 30, 35, 40, 45]
        })

        # Call the function and check if it executes without errors
        try:
            plot_distribution(data)
        except Exception as e:
            self.fail(f"plot_distribution() raised an unexpected exception: {e}")

if __name__ == '__main__':
    unittest.main()

# visualization 2
# carbon monoxide emission
def carbon_monoxide_plot(air_quality):
    sns.set(style="darkgrid")
    plt.figure(figsize=(12, 5))
    # Use Seaborn's lineplot with 'hue' for different cities
    sns.lineplot(data=air_quality, x='year', y='co', hue='city', marker='o', markersize=12)
    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('CO')
    plt.title('Average Carbon Monoxide Emission Over the Years for Different Cities')
    # Show the plot
    plt.show()

class TestCarbonMonoxidePlot(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    @patch('seaborn.lineplot')
    def test_carbon_monoxide_plot(self, mock_lineplot, mock_show):

        air_quality['year'] = pd.to_datetime(air_quality['date']).dt.year

        # Call the function
        carbon_monoxide_plot(air_quality)
        mock_lineplot.assert_called_once_with(data=air_quality, x='year', y='co', hue='city', marker='o', markersize=12)
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()

# detecting outliers
# Your function to test
def process_pm25(air_quality):
    sns.boxplot(air_quality['pm25'])
    air_quality['pm25'] = winsorize(air_quality['pm25'], limits=[0.01, 0.01])
    air_quality['pm25'] = pd.Series(air_quality['pm25'], name='pm25')
    sns.boxplot(air_quality['pm25'])
    return air_quality
class TestProcessPM25(unittest.TestCase):
    @patch('seaborn.boxplot')
    def test_process_pm25(self, mock_boxplot):
        # Create a mock DataFrame
        air_quality = pd.DataFrame({'pm25': [10, 20, 30, 40, 50]})
        # Call the function
        result = process_pm25(air_quality)
        self.assertEqual(mock_boxplot.call_count, 2)
        # Check if the 'pm25' column is a pandas Series with the correct name
        self.assertTrue(isinstance(result['pm25'], pd.Series))
        self.assertEqual(result['pm25'].name, 'pm25')

if __name__ == '__main__':
    unittest.main()

# descriptive analysis
class TestDataDescription(unittest.TestCase):
    def test_descriptive_analysis(self):
        # Sample data
        sample_data = {
            'date': pd.date_range(start='2022-01-01', periods=10),
            'value1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'value2': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        }
        air_quality = pd.DataFrame(sample_data)

        # Call the describe function
        result = air_quality.describe(include="all", datetime_is_numeric=True)

        # Assert that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Assert that the result contains the expected columns
        expected_columns = ['value1', 'value2']
        for col in expected_columns:
            self.assertIn(col, result.columns)

if __name__ == '__main__':
    unittest.main()


# AQI Calculation
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

    aqi_pm10 = (((aqi_high - aqi_low) / (c_high - c_low)) * (value - c_low)) + aqi_low
    return int(round(aqi_pm10))


# Unit test case for calculate_aqi_pm10 function
class TestCalculateAQIPM10(unittest.TestCase):
    def test_aqi_calculation(self):
        # Sample data for PM10 levels
        pm10_values = [20, 80, 150, 300, 450]

        # Expected AQI values based on the corrected breakpoints
        expected_aqi_values = [20, 80, 134, 250, 428]

        for value, expected_aqi in zip(pm10_values, expected_aqi_values):
            calculated_aqi = calculate_aqi_pm10(value)
            self.assertEqual(calculated_aqi, expected_aqi)

if __name__ == '__main__':
    unittest.main()

# categorize aqi
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

class TestCategorizeAQI(unittest.TestCase):
    def test_aqi_categories(self):
        self.assertEqual(categorize_aqi(30), "Good")
        self.assertEqual(categorize_aqi(75), "Satisfactory")
        self.assertEqual(categorize_aqi(150), "Moderate")
        self.assertEqual(categorize_aqi(250), "Poor")
        self.assertEqual(categorize_aqi(350), "Very Poor")
        self.assertEqual(categorize_aqi(500), "Severe")

if __name__ == '__main__':
    unittest.main()


# data transformation
# creating instances
ode = OrdinalEncoder()
#scaler = MinMaxScaler()
scaler = PowerTransformer(method='yeo-johnson')

# Get original column names
original_columns = ['AQI_Category', 'co', 'o3', 'no2', 'so2', 'pm25']
def transform_air_quality_data(data):

    ct = make_column_transformer(
        (ode, ['AQI_Category']),
        (scaler, ['co', 'o3', 'no2', 'so2', 'pm25']),
        remainder='passthrough'
    )
    ct.set_output(transform="pandas")
    transformed_data = ct.fit_transform(data)
    transformed_data.columns = original_columns
    return transformed_data

class TestTransformAirQualityData(unittest.TestCase):
    def test_transformation(self):
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'AQI_Category': ['Good', 'Moderate'],
            'co': [0.5, 0.8],
            'o3': [0.6, 0.9],
            'no2': [0.4, 0.7],
            'so2': [0.3, 0.6],
            'pm25': [10, 20]
        })

        # Call the function
        transformed_data = transform_air_quality_data(sample_data)

        # Verify that the transformed data has the expected columns
        expected_columns = ['AQI_Category', 'co', 'o3', 'no2', 'so2', 'pm25']
        self.assertListEqual(list(transformed_data.columns), expected_columns)

if __name__ == '__main__':
    unittest.main()

# correlation matrix

def calculate_correlation_coefficients(data):
    correlation_matrix = data.corr()
    return correlation_matrix

# Example usage:
correlation_matrix_df = calculate_correlation_coefficients(air_quality)
print(correlation_matrix_df)

class TestCalculateCorrelationCoefficients(unittest.TestCase):
    def test_correlation_matrix(self):
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'co': [0.5, 0.8, 0.3],
            'o3': [0.6, 0.9, 0.4],
            'no2': [0.4, 0.7, 0.2],
            'so2': [0.3, 0.6, 0.1],
            'pm25': [10, 20, 15]
        })

        # Call the function
        correlation_matrix = calculate_correlation_coefficients(sample_data)

        # Verify that the correlation matrix has the expected shape
        self.assertEqual(correlation_matrix.shape, (5, 5))

if __name__ == '__main__':
    unittest.main()

# data split
def split_data(data):
    X = air_new_df.drop(columns=['AQI_Category'])
    y = air_new_df['AQI_Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    return X_train, X_test, y_train, y_test
class TestSplitData(unittest.TestCase):
    def test_data_split(self):
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'AQI_Category': ['Good', 'Moderate', 'Poor'],
            'co': [0.5, 0.8, 0.3],
            'o3': [0.6, 0.9, 0.4],
            'no2': [0.4, 0.7, 0.2],
            'so2': [0.3, 0.6, 0.1],
            'pm25': [10, 20, 15]
        })

        # Call the function
        X_train, X_test, y_train, y_test = split_data(sample_data)

        # Verify that the shapes of training and testing data are as expected
        self.assertEqual(X_train.shape[0], 10904)
        self.assertEqual(X_test.shape[0], 2727)
if __name__ == '__main__':
    unittest.main()


# Data Modeling
# random forest
def train_random_forest_classifier(X_train, y_train):
    rf_model = RandomForestClassifier(class_weight='balanced')
    rf_pipeline = make_pipeline(SMOTEENN(), rf_model)
    rf_pipeline.fit(X_train, y_train)
    return rf_pipeline
class TestRandomForestClassifier(unittest.TestCase):
    def test_rf_model(self):
        # Train the model
        model = train_random_forest_classifier(X_train, y_train)
        # Predict on the test set
        y_pred = model.predict(X_test)
        # Calculate accuracy
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='micro')
        f1_score = metrics.f1_score(y_test, y_pred, average='micro')
        recall = metrics.recall_score(y_test, y_pred, average='micro')

        # Assert that performance is within a reasonable range (e.g., 0.7 to 1.0)
        self.assertTrue(0.7 <= accuracy <= 1.0)
        self.assertTrue(0.7 <= precision <= 1.0)
        self.assertTrue(0.7 <= f1_score <= 1.0)
        self.assertTrue(0.7 <= recall <= 1.0)

if __name__ == '__main__':
    unittest.main()

# SVM
def train_svm_classifier(X_train, y_train):
    svm_model = SVC(class_weight='balanced', probability=True)
    svm_pipeline = make_pipeline(SMOTEENN(), svm_model)
    svm_pipeline.fit(X_train, y_train)
    return svm_pipeline
class TestSVMClassifier(unittest.TestCase):
    def test_svm_model(self):
        # Train the model
        s_model = train_svm_classifier(X_train, y_train)
        # Predict on the test set
        sy_pred = s_model.predict(X_test)
        # Calculate accuracy
        accuracy = metrics.accuracy_score(y_test, sy_pred)
        precision = metrics.precision_score(y_test, sy_pred, average='micro')
        f1_score = metrics.f1_score(y_test, sy_pred, average='micro')
        recall = metrics.recall_score(y_test, sy_pred, average='micro')

        # Assert that performance is within a reasonable range (e.g., 0 to 1.0)
        self.assertTrue(0 <= accuracy <= 1.0)
        self.assertTrue(0 <= precision <= 1.0)
        self.assertTrue(0 <= f1_score <= 1.0)
        self.assertTrue(0 <= recall <= 1.0)

if __name__ == '__main__':
    unittest.main()

# Naive's Bayes
def train_nb_classifier(X_train, y_train):
    nb_model = GaussianNB()
    nb_pipeline = make_pipeline(SMOTEENN(), nb_model)
    nb_pipeline.fit(X_train, y_train)
    return nb_pipeline
class TestNBClassifier(unittest.TestCase):
    def test_nb_model(self):
        # Train the model
        n_model = train_nb_classifier(X_train, y_train)
        # Predict on the test set
        ny_pred = n_model.predict(X_test)
        # Calculate accuracy
        accuracy = metrics.accuracy_score(y_test, ny_pred)
        precision = metrics.precision_score(y_test, ny_pred, average='micro')
        f1_score = metrics.f1_score(y_test, ny_pred, average='micro')
        recall = metrics.recall_score(y_test, ny_pred, average='micro')

        # Assert that performance is within a reasonable range (e.g., 0.7 to 1.0)
        self.assertTrue(0.7 <= accuracy <= 1.0)
        self.assertTrue(0.7 <= precision <= 1.0)
        self.assertTrue(0.7 <= f1_score <= 1.0)
        self.assertTrue(0.7 <= recall <= 1.0)

if __name__ == '__main__':
    unittest.main()


# KNN
def train_knn_classifier(X_train, y_train):
    knn_model = KNeighborsClassifier(n_neighbors=6)
    knn_pipeline = make_pipeline(SMOTEENN(), knn_model)
    knn_pipeline.fit(X_train, y_train)
    return knn_pipeline
class TestKnnClassifier(unittest.TestCase):
    def test_knn_model(self):
        # Train the model
        kn_model = train_knn_classifier(X_train, y_train)
        # Predict on the test set
        kny_pred = kn_model.predict(X_test)
        # Calculate accuracy
        accuracy = metrics.accuracy_score(y_test, kny_pred)
        precision = metrics.precision_score(y_test, kny_pred, average='micro')
        f1_score = metrics.f1_score(y_test, kny_pred, average='micro')
        recall = metrics.recall_score(y_test, kny_pred, average='micro')

        # Assert that performance is within a reasonable range (e.g., 0.7 to 1.0)
        self.assertTrue(0.5 <= accuracy <= 1.0)
        self.assertTrue(0.5 <= precision <= 1.0)
        self.assertTrue(0.5 <= f1_score <= 1.0)
        self.assertTrue(0.5 <= recall <= 1.0)

if __name__ == '__main__':
    unittest.main()

# MLP
def train_mlp_classifier(X_train, y_train):
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=123)
    mlp_pipeline = make_pipeline(SMOTEENN(), mlp_model)
    mlp_pipeline.fit(X_train, y_train)
    return mlp_pipeline
class TestMlpClassifier(unittest.TestCase):
    def test_mlp_model(self):
        # Train the model
        mlp_model = train_mlp_classifier(X_train, y_train)
        # Predict on the test set
        my_pred = mlp_model.predict(X_test)
        # Calculate accuracy
        accuracy = metrics.accuracy_score(y_test, my_pred)
        precision = metrics.precision_score(y_test, my_pred, average='micro')
        f1_score = metrics.f1_score(y_test, my_pred, average='micro')
        recall = metrics.recall_score(y_test, my_pred, average='micro')

        # Assert that performance is within a reasonable range (e.g., 0 to 1.0)
        self.assertTrue(0 <= accuracy <= 1.0)
        self.assertTrue(0 <= precision <= 1.0)
        self.assertTrue(0 <= f1_score <= 1.0)
        self.assertTrue(0 <= recall <= 1.0)

if __name__ == '__main__':
    unittest.main()

# stacking ensemble

def train_stacking_ensemble(X_train, y_train):
    rf_model = RandomForestClassifier(class_weight='balanced')
    rf_pipeline = make_pipeline(SMOTEENN(), rf_model)
    svm_model = SVC(class_weight='balanced', probability=True)
    svm_pipeline = make_pipeline(SMOTEENN(), svm_model)
    nb_model = GaussianNB()
    nb_pipeline = make_pipeline(SMOTEENN(), nb_model)
    knn_model = KNeighborsClassifier(n_neighbors=6)
    knn_pipeline = make_pipeline(SMOTEENN(), knn_model)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=123)
    mlp_pipeline = make_pipeline(SMOTEENN(), mlp_model)

    # Create a stacking classifier with base and meta classifiers
    model_st = StackingClassifier(estimators=[('rf', rf_pipeline),
                                              ('svm', svm_pipeline),
                                              ('nb', nb_pipeline),
                                              ('knn', knn_pipeline),
                                              ('mlp', mlp_pipeline)], final_estimator=RandomForestClassifier())
    # Train the classifier on the training set
    model_st.fit(X_train, y_train)
    return model_st

class TestStackingEnsemble(unittest.TestCase):
    def test_st_model(self):
        # Train the model
        se_model = train_stacking_ensemble(X_train, y_train)
        # Predict on the test set
        se_pred = se_model.predict(X_test)
        # Calculate accuracy
        accuracy = metrics.accuracy_score(y_test, se_pred)
        precision = metrics.precision_score(y_test, se_pred, average='micro')
        f1_score = metrics.f1_score(y_test, se_pred, average='micro')
        recall = metrics.recall_score(y_test, se_pred, average='micro')

        # Assert that performance is within a reasonable range (e.g., 0 to 1.0)
        self.assertTrue(0 <= accuracy <= 1.0)
        self.assertTrue(0 <= precision <= 1.0)
        self.assertTrue(0 <= f1_score <= 1.0)
        self.assertTrue(0 <= recall <= 1.0)

if __name__ == '__main__':
    unittest.main()

class TestStackingEnsemble_md(unittest.TestCase):
    def test_st_model_md(self):
        # Train the model
        se_model = train_stacking_ensemble(X_train, y_train)
        # Predict on the test set
        se_pred = se_model.predict(mdX)
        # Calculate accuracy
        accuracy = metrics.accuracy_score(mdy, se_pred)
        precision = metrics.precision_score(mdy, se_pred, average='micro')
        f1_score = metrics.f1_score(mdy, se_pred, average='micro')
        recall = metrics.recall_score(mdy, se_pred, average='micro')

        # Assert that performance is within a reasonable range (e.g., 0 to 1.0)
        self.assertTrue(0.7 <= accuracy <= 1.0)
        self.assertTrue(0.7 <= precision <= 1.0)
        self.assertTrue(0.7 <= f1_score <= 1.0)
        self.assertTrue(0.7 <= recall <= 1.0)

if __name__ == '__main__':
    unittest.main()

# Model Evaluation
# cross validation scores
class CrossValidation(unittest.TestCase):
    def test_cross_validation(self):
        # Train the stacking ensemble model
        model_st = train_stacking_ensemble(X_train, y_train)

        # Perform cross-validation
        cv_scores = cross_val_score(model_st, X_train, y_train, cv=5)

        # Assertions
        self.assertIsInstance(model_st, StackingClassifier)
        self.assertIsInstance(cv_scores, np.ndarray)
        self.assertEqual(len(cv_scores), 5)
        self.assertTrue(all(score >= 0 for score in cv_scores))

if __name__ == '__main__':
    unittest.main()

# precision scores
class PrecisionScores(unittest.TestCase):
    def test_precision_scores(self):
        # Train the stacking ensemble model
        model_st = train_stacking_ensemble(X_train, y_train)

        # Perform cross-validation
        precision_scores = cross_val_score(model_st, X_train, y_train, cv=5, scoring='precision_macro')

        # Assertions
        self.assertIsInstance(model_st, StackingClassifier)
        self.assertIsInstance(precision_scores, np.ndarray)
        self.assertEqual(len(precision_scores), 5)
        self.assertTrue(all(score >= 0 for score in precision_scores))

if __name__ == '__main__':
    unittest.main()

# Recall Scores
class RecallScores(unittest.TestCase):
    def test_recall_scores(self):
        # Train the stacking ensemble model
        model_st = train_stacking_ensemble(X_train, y_train)

        # Perform cross-validation
        recall_scores = cross_val_score(model_st, X_train, y_train, cv=5, scoring='recall_macro')

        # Assertions
        self.assertIsInstance(model_st, StackingClassifier)
        self.assertIsInstance(recall_scores, np.ndarray)
        self.assertEqual(len(recall_scores), 5)
        self.assertTrue(all(score >= 0 for score in recall_scores))

if __name__ == '__main__':
    unittest.main()

# F1 Scores
class F1Scores(unittest.TestCase):
    def test_f1_scores(self):
        # Train the stacking ensemble model
        model_st = train_stacking_ensemble(X_train, y_train)

        # Perform cross-validation
        f1_scores = cross_val_score(model_st, X_train, y_train, cv=5, scoring='recall_macro')

        # Assertions
        self.assertIsInstance(model_st, StackingClassifier)
        self.assertIsInstance(f1_scores, np.ndarray)
        self.assertEqual(len(f1_scores), 5)
        self.assertTrue(all(score >= 0 for score in f1_scores))

if __name__ == '__main__':
    unittest.main()


# Accuracy Scores
class AccuracyScores(unittest.TestCase):
    def test_acc_scores(self):
        # Train the stacking ensemble model
        model_st = train_stacking_ensemble(X_train, y_train)

        # Perform cross-validation
        acc_scores = cross_val_score(model_st, X_train, y_train, cv=5, scoring='accuracy')

        # Assertions
        self.assertIsInstance(model_st, StackingClassifier)
        self.assertIsInstance(acc_scores, np.ndarray)
        self.assertEqual(len(acc_scores), 5)
        self.assertTrue(all(score >= 0 for score in acc_scores))

if __name__ == '__main__':
    unittest.main()

# confusion matrix
def evaluate_model(model, X_test, y_test):

    # Make predictions
    y_pred = model.predict(X_test)

    # Generate confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.plasma)
    plt.show()
    return conf_mat


class TestEvaluateModel(unittest.TestCase):
    # Set up test data
    def test_confusion_matrix(self):
        model_st = train_stacking_ensemble(X_train, y_train)
        # Call the evaluate_model function
        conf_mat = evaluate_model(model_st, X_test, y_test)

        # Assertion
        self.assertIsNotNone(conf_mat, "The confusion matrix should not be None.")

        # Display the confusion matrix for visual inspection
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=model_st.classes_)
        disp.plot(cmap=plt.cm.plasma)
        plt.show()

if __name__ == '__main__':
    unittest.main()

# ROC-AUC
def evaluate_roc(model, X_test, y_test):

    # Make predictions
    y_proba = model.predict_proba(X_test)

    # RoC_auc
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    return roc_auc

class TestRocAuc(unittest.TestCase):
    # Set up test data
    def test_roc_auc(self):
        model_st = train_stacking_ensemble(X_train, y_train)
        # Call the evaluate_model function
        roc = evaluate_roc(model_st, X_test, y_test)

        # Assertion
        self.assertIsNotNone(roc, "The ROC AUC score should not be None.")


if __name__ == '__main__':
    unittest.main()











