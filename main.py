import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()


st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html = True
)



@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename)

    return taxi_data


with header:
    st.title("Welcome to my awesome data science project.")
    st.text("In this project I look into the transaction of taxis in NYC...")

with dataset:
    st.header("NYC taxi dataset")
    st.text("I found this dataset on worldbank.com, ....")

    st.subheader('Pick-up location ID distribution on the NYC dataset')
    taxi_data = pd.read_csv('yellow_tripdata_2021-01.csv')
    st.write(taxi_data.head())

    st.subheader('Pick-up location ID distribution on the NYC dataset.')
    population_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts())
    st.bar_chart(population_dist)


with features:
    st.header("The features I created")

    st.markdown('* **first feature:** I created feature because of this..I calculated it using this logic..')
    st.markdown('* **second feature:** I created feature because of this..I calculated it using this logic..')

with model_training:
    st.header("Time to train the model")
    st.text("Here you get to choose the hyperparameters of the model and see how the performance change.")

    sel_col, disp_col = st.beta_columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100, 200, 300, 'No limit'], index = 0)

    sel_col.text('Here is list of feature in my data:')
    sel_col.write(taxi_data.columns.values)

    input_feature = sel_col.text_input('which feature should be used as the input feature?', 'PULocationID')

    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]

    regr.fit(X, y)
    prediction = regr.predict(X)
    disp_col.subheader("Mean absolute error of the model is: ")
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader("Mean squared error error of the model is:")
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader("R squared score of the model is:")
    disp_col.write(r2_score(y, prediction))



