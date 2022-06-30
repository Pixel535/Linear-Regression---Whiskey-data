import locale
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
from IPython.display import display
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

############################################################################################# WHISKEY
column_list = ["id", "name", "category", "review.point", "price", "currency"]
whiskey_review = pd.read_csv("scotch_review2020.csv", index_col='id', usecols= column_list)
whiskey_review.rename(columns={"review.point": "points", "description.1.2247.": "description"}, inplace=True)
whiskey_review.at[(34, 187, 740, 1549, 1815), 'price'] = 15000
whiskey_review.at[(93), 'price'] = 300
whiskey_review.at[(95, 360), 'price'] = 100
whiskey_review.at[(779), 'price'] = 200
whiskey_review.at[(1011), 'price'] = 44 * .75
whiskey_review.at[(1281), 'price'] = 132 * 1.07
whiskey_review.at[(1826), 'price'] = 39 * .4285
whiskey_review.at[(2028), 'price'] = 35 * .75
whiskey_review.at[(2201), 'price'] = 18 * .4285
whiskey_review['price'] = whiskey_review['price'].str.replace(',', '')
whiskey_review = whiskey_review.dropna()
whiskey_review.sort_values('points', ascending=False, inplace=True)
whiskey_review['price'] = whiskey_review['price'].astype(int)
whiskey_review['price_per_point'] = whiskey_review['price'] / whiskey_review['points']
display(whiskey_review)

X = np.c_[whiskey_review['price_per_point']]
Y = np.c_[whiskey_review['price']]
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

# Visualize the data
whiskey_review.plot(kind='scatter', x="price_per_point", y='price')
plt.scatter(X_train, y_train)

# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
r = model.fit(X_train, y_train)

# Make a prediction
X_new = [[139]]
print(model.predict(X_new))  # outputs [[ 5.96242338]] for all data (5.87590268 for X_train)

# Calculate loss
Y_pred = model.predict(X_test)
loss = mean_absolute_error(y_test, Y_pred)
plt.plot(X_test, Y_pred, color ='red')
plt.title('Whiskey Price By Price Per Review Point')
plt.show()
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': Y_pred.flatten()})
print(df)
print(loss)
