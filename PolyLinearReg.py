import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df=pd.read_csv("polynomial.csv",sep=";")

poly_reg=PolynomialFeatures(degree=4)-
reg=LinearRegression()-
reg.fit(df[["deneyim"]],df[["maas"]])

plt.xlabel("Deneyim")
plt.ylabel("Maaş")

plt.scatter(df["deneyim"], df["maas"])

print(df["deneyim"])
x_ekseni=df["deneyim"]
y_ekseni=reg.predict(df[["deneyim"]])
plt.plot(x_ekseni,y_ekseni,color="green",label="linear regression")


x_poly=poly_reg.fit_transform(df[["deneyim"]])
reg=LinearRegression()
reg.fit(x_poly,df[["maas"]])

print(x_poly)

y_head= reg.predict(x_poly)
plt.plot(df["deneyim"],y_head,color="red",label="polynomial")


x_poly1=poly_reg.fit_transform([[4.5]])
print(reg.predict(x_poly1)
)
plt.legend()
plt.show()
