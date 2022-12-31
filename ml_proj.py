import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
data=pd.read_csv(r"C:\Users\Krishna\Downloads\student\student-mat.csv", sep=";")
data=data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict="G3"
X=np.array(data.drop([predict], 1))
y=np.array(data[predict])
X_train, x_test, y_train, y_test=sklearn.model_selection.train_test_split(X,y, test_size=0.1)
linear= linear_model.LinearRegression()
linear.fit(X_train,y_train)
acc=linear.score(x_test, y_test)
print("coeff: ",linear.coef_)
print("intercept: ", linear.intercept_, "\n")
predictions=linear.predict(x_test)
for i in range(len(predictions)):
    print(f"predicted value:{predictions[i]}, dataset:{x_test[i]}, actual result: {y_test[i]}", sep=";", end="\n\n")
