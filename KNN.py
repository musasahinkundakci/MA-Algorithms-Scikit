import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("diabetes.csv")
head=data.head()

patients=data[data.Outcome==1]
healthy=data[data.Outcome==0]

plt.scatter(patients.Age,patients.Glucose,color="red",label="patient",alpha=0.4)#alpha saydamlık
plt.scatter(healthy.Age, healthy.Glucose, color="green",label="healthy",alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()

y=data.Outcome.values#dependent
x_ham=data.drop(["Outcome"],axis=1)#independent


#normalization
x_ham=(x_ham-np.min(x_ham))/(np.max(x_ham)-np.min(x_ham))
    
x_train,x_test,y_train,y_test=train_test_split(x_ham,y,test_size=0.2,random_state=1)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)
print("K=3 için doğruluk oranı: ",knn.score(x_test,y_test))
sayac=1
for k in range(1,10):
    knn_new=KNeighborsClassifier(n_neighbors=k)
    knn_new.fit(x_train,y_train)
    print("K={} için doğruluk oranı => {}".format(k,knn_new.score(x_test,y_test)))