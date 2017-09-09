import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# getting input data
data = pd.read_fwf('dataset.txt')

# seperating into 2 parts for linear regression
x = data[['Brain']]
y = data[['Body']]

# creating a linear regression model object
lr = linear_model.LinearRegression()

# fitting the values using the dataset
lr.fit(x,y)

# time to create our plots
plt.scatter(x,y)

# this will be predicted by our model
plt.plot(x, lr.predict(x))

# show the plot on the user screen
plt.show()
