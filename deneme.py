import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model,preprocessing,model_selection

df=pd.read_csv("breast-cancer.csv")
df=df.drop(["id"],axis=1)

diagnosis=df.iloc[:,0:1].values
le=preprocessing.LabelEncoder()
ohe=preprocessing.OneHotEncoder()
diagnosis=ohe.fit_transform(diagnosis).toarray()
diagnosis=diagnosis[:,1:]

df["diagnosis"]=diagnosis

patients=df[df.diagnosis==1]
healthy=df[df.diagnosis==0]

plt.scatter()

"""
x_train,x_test,y_train,y_test=model_selection.train_test_split(df.iloc[:,1:],df.iloc[:,0:1],test_size=0.2,random_state=1)
sc=preprocessing.StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

reg=linear_model.LinearRegression()
reg.fit(X_train,y_train)
prediction=reg.predict(X_test)
"""