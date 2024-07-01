import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('output.csv')
#sampled_df = df.sample(frac=0.4)  # Get 50% of the data 
#len(sampled_df)


#st.dataframe(sampled_df)


import streamlit as st # import streamlit and assign it to the variable 'st'

st.title("Machine Learning Workflow with Streamlit")

# Step 1: Upload the dataset
st.header("Step 1: Upload the Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
else:
    st.stop()
    
    
import streamlit as st # import streamlit and assign it to the variable 'st'
import io # import the io module

st.title("Machine Learning Workflow with Streamlit")

# Step 2: Data Exploration
st.header("Step 2: Data Exploration")
st.write("Data Information:")
buffer = io.StringIO() # Create a StringIO object to capture df.info() output
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)
st.write("Summary Statistics:")
st.write(df.describe())

# Step 3: Data Cleaning
st.header("Step 3: Data Cleaning")

# Handle Missing Values
st.write("Handling Missing Values...")
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)
st.write("Missing values handled.")

# Remove Duplicates
st.write("Removing Duplicates...")
df.drop_duplicates(inplace=True)
st.write("Duplicates removed.")

# Encode Categorical Features

import pandas as pd
from sklearn.preprocessing import LabelEncoder
st.write("Encoding Categorical Features...")
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])
st.write("Categorical features encoded.")
st.write(df.head())

# Step 4: Train the Model
st.header("Step 4: Train the Model")


# Select the target variable from the dataset
# Author text
st.sidebar.markdown('<h5 style="color: black;"> Author : Mactar Sarr </h5>', unsafe_allow_html=True)

# Sidebar for user input selection
st.sidebar.markdown('<h1 style="color: blue;">Select One output and at least one input Variable</h1>', unsafe_allow_html=True)

# Assuming 'df' is your DataFrame, get the column names
column_names = df.columns.tolist()  # Get column names from your DataFrame

# Select output variable
output_variable_model = st.sidebar.selectbox('Select One output Variable', column_names)

# Get potential default values that are actually in column_names
potential_defaults = [col for col in ['R_450', 'R_550', 'R_650', 'R_720', 'R_750', 'R_800'] if col in column_names]

# Select input variables to predict the target variable (output)
# Use the safe defaults calculated above
input_variables_model = st.sidebar.multiselect('Select at least one input Variable', column_names, default=potential_defaults)

if not output_variable_model or not input_variables_model:
    st.warning('Select One output and at least one input Variable to start.')

# User option for setting the rate of test data
test_data_rate = st.sidebar.slider('Select the rate of test data (%)', 0, 100, 20, 1)
# Import necessary library
from sklearn.model_selection import train_test_split
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor 

# Define input features (X) and target variable (y) for model training
# Ensure input_variables_model is not empty before proceeding
if input_variables_model:
    X_model = df[input_variables_model]
    y_model = df[output_variable_model]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=test_data_rate / 100, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
else:
    st.warning('Please select at least one input variable.') # Provide feedback to the user
    
# Import necessary library
from sklearn.model_selection import train_test_split
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor 

# Define input features (X) and target variable (y) for model training
# Ensure input_variables_model is not empty before proceeding
if input_variables_model: # Check if input variables have been selected
    X_model = df[input_variables_model]
    y_model = df[output_variable_model]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=test_data_rate / 100, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train) 
else:
    st.warning('Please select at least one input variable.') # Provide feedback to the user if no input variables are selected


# Use the model for prediction
if input_variables_model: # Check if input variables have been selected before making predictions
    st.title('Use the Model for Prediction')
    st.markdown('<h4 style="color: black;"> Use Sidebar menu to select the values of input variables to predict the target variable. </h4>', unsafe_allow_html=True)

    # User input for feature values
    st.sidebar.markdown('<h2 style="color: blue;"> Select the values of input variables to predict the target variable</h2>', unsafe_allow_html=True)
    user_input_prediction = {}
    for column in input_variables_model:
        user_input_prediction[column] = st.sidebar.slider(f'Select {column}', float(df[column].min()), float(df[column].max()), float(df[column].mean())) # Use 'df' instead of 'data'

    # Predict and display result
    prediction = model.predict(pd.DataFrame([user_input_prediction]))
    st.subheader('Prediction')
    st.write(f'The predicted {output_variable_model} value is: {prediction[0]:.5f}')

    # Display a bar chart for the predicted output
    st.subheader('Predicted Output Chart')
    prediction_data = pd.DataFrame({output_variable_model: [prediction[0]]})
    fig, ax=plt.subplots(figsize=(8, 5))
    sns.barplot(data=prediction_data, palette=['orange'])
    ax.set_title(f'Predicted {output_variable_model} Value')
    ax.set_ylabel('Value')
    st.pyplot(fig)
else:
    st.warning("Please select input variables first.") # Inform the user to select input variables
    
    

