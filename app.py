import streamlit as st
import joblib
import pandas as pd 

model = joblib.load('TicketClassifier.pkl') # Assuming this is your best model
scaler = joblib.load('Scaler.pkl')

st.title('Ticket Category Prediction Web App')
st.text('This is a simple web app to predict ticket categories using a pre-trained machine learning model.')


num_code = st.number_input('Flight Number (num_code):', min_value=100, max_value=9999, value=812)


dep_Time = st.number_input('Departure Time :', min_value=0.0, max_value=23.99, value=9.75)
arr_Time = st.number_input('Arrival Time :', min_value=0.0, max_value=23.99, value=19.91)


stop_options = {'non-stop': 0, '1-stop': 1, '2+-stop': 2}
selected_stop = st.selectbox('Number of Stops:', list(stop_options.keys()))
stop_encoded = stop_options[selected_stop]


typeofFlight_options = {'economy': 0, 'business': 1}
selected_typeofFlight = st.selectbox('Type of Flight:', list(typeofFlight_options.keys()))
typeofFlight_encoded = typeofFlight_options[selected_typeofFlight]


day = st.number_input('Day of month:', min_value=1, max_value=31, value=5)
month = st.number_input('Month:', min_value=1, max_value=12, value=3)


airline_options_full = ['Vistara', 'Air India', 'Indigo', 'GO FIRST', 'AirAsia', 'SpiceJet', 'StarAir', 'Trujet']
selected_airline = st.selectbox('Airline:', airline_options_full)


source_options_full = ['Bangalore', 'Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Hyderabad']
selected_source = st.selectbox('Source City:', source_options_full)


destination_options_full = ['Bangalore', 'Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Hyderabad']
selected_destination = st.selectbox('Destination City:', destination_options_full)


if st.button('Predict Ticket Category'):
    
    airline_ohe = {airline: 0 for airline in airline_options_full}
    source_ohe = {source: 0 for source in source_options_full}
    destination_ohe = {dest: 0 for dest in destination_options_full}

    
    if selected_airline in airline_ohe: airline_ohe[selected_airline] = 1
    if selected_source in source_ohe: source_ohe[selected_source] = 1
    if selected_destination in destination_ohe: destination_ohe[selected_destination] = 1

    
    input_data_dict = {
        'num_code': num_code,
        'dep_time': dep_Time,
        'stop': stop_encoded,
        'arr_time': arr_Time,
        'type': typeofFlight_encoded,
        'day': day,
        'month': month,
    }
    
  
    for airline_col in airline_options_full:
        input_data_dict[airline_col] = airline_ohe[airline_col]

    
    for source_col in source_options_full:
        input_data_dict[f'source_{source_col}'] = source_ohe[source_col]

    
    for dest_col in destination_options_full:
        input_data_dict[f'destination_{dest_col}'] = destination_ohe[dest_col]

    
    feature_columns_order = [
        'num_code', 'dep_time', 'stop', 'arr_time', 'type', 'day', 'month',
        'Vistara', 'Air India', 'Indigo', 'GO FIRST', 'AirAsia', 'SpiceJet', 'StarAir', 'Trujet',
        'source_Bangalore', 'source_Delhi', 'source_Mumbai', 'source_Kolkata', 'source_Chennai', 'source_Hyderabad',
        'destination_Bangalore', 'destination_Delhi', 'destination_Mumbai', 'destination_Kolkata', 'destination_Chennai', 'destination_Hyderabad'
    ]

    
    X_predict = pd.DataFrame([input_data_dict], columns=feature_columns_order)

    
    X_scaled = scaler.transform(X_predict)

    
    y_pred_numeric = model.predict(X_scaled)
    predicted_label_index = int(y_pred_numeric[0])

    
    labels = ['cheap', 'moderate', 'expensive', 'very expensive'] 
    predicted_category = labels[predicted_label_index]

    st.success(f"Predicted ticket category: {predicted_category}")