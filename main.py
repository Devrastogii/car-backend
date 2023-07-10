from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = os.getenv("BASE_URL")

app = Flask(__name__)
car = pd.read_csv('backend/cleaned_cars_2')

def log1p_transform(x):
    x_numeric = float(x.iloc[0])
    transformed = np.log1p(x_numeric)
    return pd.DataFrame([transformed])

def sq_transform(x):
    x_numeric = float(x.iloc[0])
    transformed = np.square(x_numeric)
    return pd.DataFrame([transformed])

model = pickle.load(open('backend/LinearRegressionModel-2.pkl', 'rb'))

CORS(app, origins=[BASE_URL], supports_credentials=True)

@app.route('/all_data', methods = ['GET'])
def home():
    name = sorted(car['Name'].unique().tolist())
    company = sorted(car['Company'].unique().tolist())
    fuel = sorted(car['Fuel_type'].unique().tolist())

    # When you are not getting unique values even using unique() then check dtype. Like in above case there was 'Petrol' and ' Petrol', these are different so we were getting two petrol so used strip()

    year = sorted(car['Year'].unique().tolist())
    location = sorted([str(item) for item in car['Location'].unique().tolist()])

    result_data = []    
    json_data =  {
    'name': name,
    'company': company,
    'fuel': fuel,
    'year': year,
    'location': location
    }

    result_data.append(json_data)
    return result_data
    

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('nameData')
        company = data.get('companyData')
        year = data.get('yearData')
        fuel = data.get('fuelData')
        location = data.get('locationData')
        km = data.get('kmData')        

        prediction = model.predict(pd.DataFrame([[name, location, company, year, km, fuel]], columns=['Name', 'Location', 'Company', 'Year', 'Kms_driven', 'Fuel_type']))
    
    return jsonify(np.round(prediction[0]))

app.run(debug=True)