#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import main (major) libraries
import numpy as np
import pandas as pd

# Features engineering
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Model selection and hyperParameter tuning
from sklearn.model_selection import train_test_split

# sklearn pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

# impute null values
from sklearn.impute import SimpleImputer

# other libraries
import os


# In[2]:


# Detect paths
curr_dir = os.getcwd()
file_features_name = "cal_housing_names.csv" # file of feature names
path1 = os.path.join(curr_dir, file_features_name)

file_features_data = "cal_housing_data.csv" # file of features data
path2 = os.path.join(curr_dir, file_features_data)

# load the dataset
feature_names = pd.read_csv(path1)
feature_names = feature_names.iloc[:, 0]

calHousing_df = pd.read_csv(path2, names=feature_names) # read data with names as their features names

# show sample from the dataframe
calHousing_df.head(10)


# In[3]:


# Extract new features from existence features
calHousing_df['roomsPerHousehold'] = calHousing_df['totalRooms'] / calHousing_df['households']
calHousing_df['bedRoomsPerroom'] = calHousing_df['totalBedrooms'] / calHousing_df['totalRooms']
calHousing_df['PopulationPerHousehold'] = calHousing_df['population'] / calHousing_df['households']


# Function to classify households
def households_class(row, medianHousingAge, medIncome):
    if ((row['housingMedianAge'] <= medianHousingAge) & (row['medianIncome'] >= medIncome * 2)) | (row['medianIncome'] >= medIncome * 2):
        return 'A'
    elif ((row['housingMedianAge'] <= medianHousingAge * 2) & (row['medianIncome'] >= medIncome)) | (row['medianIncome'] >= medIncome):
        return 'B'
    else:
        return 'C'

# Compute global intervals
medianHousingAge = np.max(calHousing_df['housingMedianAge']) // 3
medIncome = np.max(calHousing_df['medianIncome']) / 3

# Apply classification function to the DataFrame
calHousing_df['householdClass'] = calHousing_df.apply(
    households_class, axis=1, 
    medianHousingAge=medianHousingAge, 
    medIncome=medIncome
)


# In[4]:


# features and target detection and seperation
X = calHousing_df.drop('medianHouseValue', axis=1)
y = calHousing_df['medianHouseValue']

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

# split numerical features lonely 
num_cols = [col for col in X_train.columns if X_train[col].dtype in ['int32', 'int64', 'float32', 'float64']]

# split categorical features lonely
cat_cols = [col for col in X_train.columns if X_train[col].dtype not in ['int32', 'int64', 'float32', 'float64']]


# In[5]:


# Pipelines
# numerical pipeline
num_pipeline = Pipeline(steps=[
                                ('selector', DataFrameSelector(num_cols)),
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())
                            ])

# categorical pipeline
cat_pipeline = Pipeline(steps=[
#                                 ('selector1', DataFrameSelector(['housingMedianAge', 'medianIncome', 'householdClass'])),
#                                 ('householdClassifier', HouseholdClassTransformer()), ## active it if you want to classify new instances
                                ('selector2', DataFrameSelector(cat_cols)),
                                ('encoder', OrdinalEncoder())
                            ])

# combine pipelines
total_pipeline = FeatureUnion(transformer_list=[
                                ('num', num_pipeline),
                                ('cat', cat_pipeline)
                            ])

# Fit and transform
X_train_transformed = total_pipeline.fit_transform(X_train)


# In[6]:


def preprocessing_new(X_new):
    ''' This function is doing preprocessing for the data to be prepared to pass to the model
    Args:
    *****
    (X_new: 2D array) ---> The features in the same order as follow:
        [longitude, latitude, housingMedianAge, totalRooms, totalBedrooms, 
        population, households, medianIncome, medianHouseValue]
    All features are numerical data
    
    Returns:
    ********
    Array of processed data which are ready to make inference by the model
    '''
    
    return total_pipeline.transform(X_new)

