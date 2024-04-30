import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split



melb_filepath = './Machine Learning/melb_data.csv'

#read the data and store it in a DataFrame
melbourne_data_unfil = pd.read_csv(melb_filepath)
#summary of the data
#print(melbourne_data.describe())

# Filtering the data with by removing the data with Null values
melbourne_data = melbourne_data_unfil.dropna(axis=0)
#print(melbourne_data.columns)

y = melbourne_data.Price

# Ofc features can be varied and better features would land you a more rounded model to
# predict the prices of homes ofc :D
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]

#print(X.describe())

#print(X.head())

#specifying the model
melb_model = DecisionTreeRegressor(random_state=1)

melb_model.fit(X, y)

predictions = melb_model.predict(X)

#Model validation starts with the error of the predictions, ofc if you
# use the sample data to train your model then the predictions themselves will be incorrect,
# however if you split the data sample into two, using one of them to train the model and other to test
# the model would a better model validator, well in theory anyways :D
print(mean_absolute_error(y, predictions))
#print(predictions)
#The prediction and the actual prices can be compared
#print(y.head())
#Problem is the accuracy of the predictions
