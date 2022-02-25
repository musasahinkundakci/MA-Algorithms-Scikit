import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

df=pd.read_csv("multilinearregression.csv",sep=";")

reg=linear_model.LinearRegression()
reg.fit(df[["alan","odasayisi","binayasi"]],df["fiyat"])


x=list(map(int,input("Alan | Oda sayisi | Bina yasi => ").split()))
print("Tahmin edilen fiyat: ",reg.predict([x])," L")
print(reg.coef_)#katsayı
print(reg.intercept_)#sabit değr

    
