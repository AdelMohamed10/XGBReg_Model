import numpy as np
import pandas as pd
import joblib
import gradio as gr
from utils import preprocessing_new, households_class

# Load the trained model
model = joblib.load('xgbr_model.pkl')

def predict_house_value(longitude, latitude, housingMedianAge, totalRooms, totalBedrooms, population, households, medianIncome):
    try:
        roomsPerHousehold = totalRooms / households
        bedRoomsPerRoom = totalBedrooms / totalRooms
        PopulationPerHousehold = population / households

        features = {
            'longitude': longitude,
            'latitude': latitude,
            'housingMedianAge': housingMedianAge,
            'totalRooms': totalRooms,
            'totalBedrooms': totalBedrooms,
            'population': population,
            'households': households,
            'medianIncome': medianIncome,
            'roomsPerHousehold': roomsPerHousehold,
            'bedRoomsPerroom': bedRoomsPerRoom,
            'PopulationPerHousehold': PopulationPerHousehold,
        }
        df = pd.DataFrame([features])

        #   (Household Class)
        medianHousingAge = housingMedianAge // 3
        medIncome = medianIncome / 3
        df['householdClass'] = df.apply(
            households_class,
            axis=1,
            medianHousingAge=medianHousingAge,
            medIncome=medIncome
        )

        df_processed = preprocessing_new(df)

        y_pred = model.predict(df_processed)[0]

        return round(y_pred, 2)

    except Exception as e:
        return f"Error: {str(e)}"

#  Gradio
inputs = [
    gr.Number(label="Longitude"),
    gr.Number(label="Latitude"),
    gr.Number(label="Housing Median Age"),
    gr.Number(label="Total Rooms"),
    gr.Number(label="Total Bedrooms"),
    gr.Number(label="Population"),
    gr.Number(label="Households"),
    gr.Number(label="Median Income"),
]

output = gr.Textbox(label="Predicted House Value")

iface = gr.Interface(
    fn=predict_house_value,
    inputs=inputs,
    outputs=output,
    title="House Value Predictor",
    description="Enter the details of the house to predict its value."
)

if __name__ == "__main__":
    iface.launch()