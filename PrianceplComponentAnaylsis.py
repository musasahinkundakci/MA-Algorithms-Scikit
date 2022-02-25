import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

features=["Sepal Length","Sepal Width","Petal Length","Petal Width","target"]
url="pca_iris.data"

df=pd.read_csv(url,sep=",",names=features)

features.pop()
x=df[features]
y=df[["target"]]

x=StandardScaler().fit_transform(x)

pca=PCA(n_components=2)
principalComponents=pca.fit_transform(x)
principalDf=pd.DataFrame(data=principalComponents,columns=["Principal Component 1","Principal Component 2"])

final_df=pd.concat([principalDf,df[["target"]]],axis=1)

"""
dfsetosa=final_df[df.target=="Iris-setosa"]
dfvirginica=final_df[df.target=="Iris-virginica"]
dfversicolor=final_df[df.target=="Iris-versicolor"]
plt.xlabel("Principal Component1")
plt.ylabel("Principal Component2")
for i,j in [[dfsetosa,"green"],[dfvirginica,"red"],[dfversicolor,"blue"]]:
    plt.scatter(i["Principal Component 1"], i["Principal Component 2"],color=j)
"""
targets=["Iris-setosa","Iris-virginica","Iris-versicolor"]
colors=["g","b","r"]
plt.xlabel("Principal Component1")
plt.ylabel("Principal Component2")

for target,col in zip(targets,colors):
    dftemp=final_df[df.target==target]
    plt.scatter(dftemp["Principal Component 1"], dftemp["Principal Component 2"],color=col)
    

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())