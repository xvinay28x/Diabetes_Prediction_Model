import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("diabetes (1).csv")
print(df.head())

df.drop("DiabetesPedigreeFunction",axis = 1,inplace=True)

x = df.iloc[:,:7].values
y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

reg = LogisticRegression()
reg.fit(x_train,y_train)

pickle.dump(reg,open("model.pkl","wb"))