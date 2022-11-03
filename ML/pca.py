import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

data = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
data

X = data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = data['variety']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
# principalDf
finalDf = pd.concat([principalDf,y],axis = 1)
print(finalDf)