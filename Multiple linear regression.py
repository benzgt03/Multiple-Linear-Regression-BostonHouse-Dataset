import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd  # import libary ต่างๆที่จะใช้
from sklearn.datasets import load_boston #Load Dataset from sklearn


#Preparation
boston = load_boston()
df_boston = pd.DataFrame(boston.data)
df_boston.columns = boston.feature_names
df_boston['PRICE'] = boston.target
print(boston.DESCR)
print(df_boston.head())
print(df_boston.describe())
print('check number =',df_boston.nunique()) # number of unique variable
print('check null =',df_boston.isnull().sum()) # check Null
print(df_boston.corr())
# Heatmap
sns.heatmap(df_boston.corr() , square=True, fmt='.1f', annot=True, cmap='Reds')

df_boston.pop('PRICE')

#Train and linear regression
X_train, X_test, y_train, y_test = train_test_split(boston['data'], boston['target'], random_state=40)
lm = LinearRegression()
lm.fit(X_train, y_train)
list_x = df_boston.columns.values.tolist()
list_y = lm.coef_.reshape(-1,1)
list_x = np.asarray(list_x)
list_x = list_x.reshape(-1,1)
print('lm score =', lm.score(X_train, y_train))
y_pred = lm.predict(X_test)
print("Intercept =", lm.intercept_)
coeff_df = pd.DataFrame(list_y ,list_x, columns=['Coefficient'])
print(coeff_df)
print('Coefficient of determination: %.2f (The best case is 1)' % r2_score(y_test, y_pred))
print('Root Mean squared error: %.2f' % (np.sqrt(mean_squared_error(y_test, y_pred))))
pred_df=pd.DataFrame({"Original Price of House": y_test, "Prediction Price of House": y_pred})
print(pred_df)


#plot result
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.scatter(y_test , y_pred)
axes.set_title('Prediction price vs price')
axes.set_xlabel('Price')
axes.set_ylabel('Prediction price')











plt.show()





