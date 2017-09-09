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

# show labels
plt.xlabel('Brain Size')
plt.ylabel('Body weight')

# show annotations
plt.annotate('given data', xy = (2547, 4603), xytext = (3000, 5000), arrowprops = dict(facecolor = 'green', shrink = 0.10),)
plt.annotate('prediction line', xy = (3000, lr.predict(3000)), xytext = (4000, 3000), arrowprops = dict(facecolor = 'green', shrink = 0.1),)

# show the plot on the user screen
plt.show()
