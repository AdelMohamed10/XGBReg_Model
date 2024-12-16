import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request
from utils import preprocessing_new

# initialize
app = Flask(__name__)

# Load the trained model
model = joblib.load('xgbr_model.pkl')


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from the form
            longit = float(request.form['longitude'])
            latit = float(request.form['latitude'])
            housemedage = float(request.form['housingMedianAge'])
            totalrooms = float(request.form['totalRooms'])
            totalbed = float(request.form['totalBedrooms'])
            pop = float(request.form['population'])
            hholds = float(request.form['households'])
            medincome = float(request.form['medianIncome'])

            # Calculate new features
            roomsPerHousehold = totalrooms / hholds
            bedRoomsPerroom = totalbed / totalrooms
            PopulationPerHousehold = pop / hholds

            # Define feature names
            features = {
                'longitude':longit, 'latitude':latit, 'housingMedianAge':housemedage, 'totalRooms':totalrooms, 
                'totalBedrooms':totalbed, 'population':pop, 'households':hholds, 'medianIncome':medincome,
                'roomsPerHousehold':roomsPerHousehold, 'bedRoomsPerroom':bedRoomsPerroom, 'PopulationPerHousehold':PopulationPerHousehold
            }

            # Create DataFrame
            df = pd.DataFrame([features])

            # Function to classify households
            def households_class(row, medianHousingAge, medIncome):
                if ((row['housingMedianAge'] <= medianHousingAge) & (row['medianIncome'] >= medIncome * 2)) | (row['medianIncome'] >= medIncome * 2):
                    return 'A'
                elif ((row['housingMedianAge'] <= medianHousingAge * 2) & (row['medianIncome'] >= medIncome)) | (row['medianIncome'] >= medIncome):
                    return 'B'
                else:
                    return 'C'

            # Compute global intervals
            medianHousingAge = np.max(housemedage) // 3
            medIncome = np.max(medincome) / 3

            # Apply classification function to the DataFrame
            df['householdClass'] = df.apply(
                households_class, axis=1, 
                medianHousingAge=medianHousingAge, 
                medIncome=medIncome
            )
            print(df)

            # Preprocess data
            df_processed = preprocessing_new(df)

            # Predict the house value
            y_pred = model.predict(df_processed)[0]

            # Render the result in a new template
            return render_template('result.html', prediction=round(y_pred, 2))

        except Exception as e:
            # Handle errors
            return render_template('predict.html', error=str(e))
    else:
        return render_template('predict.html')


# terminal
if __name__ == '__main__':
    app.run(debug=True)
