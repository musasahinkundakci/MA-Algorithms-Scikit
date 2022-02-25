
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

df=pd.read_csv("DecisionTreesClassificationDataSet.csv",sep=",")

onezeroMap={"Y":1,"N":0}
egitimMap={"BS":0,"MS":1,"PhD":2}

df["IseAlindi"]=df["IseAlindi"].map(onezeroMap)
df["SuanCalisiyor?"]=df["SuanCalisiyor?"].map(onezeroMap)
df["Top10 Universite?"]=df["Top10 Universite?"].map(onezeroMap)
df["StajBizdeYaptimi?"]=df["StajBizdeYaptimi?"].map(onezeroMap)

#
df["Egitim Seviyesi"]=df["Egitim Seviyesi"].map(egitimMap)

y=df["IseAlindi"]
x=df.drop(["IseAlindi"],axis=1)

#tree
clf=tree.DecisionTreeClassifier()
clf=clf.fit(x,y)

print(clf.predict([[5,1,3,0,0,0]]))