from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=10)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)

from sklearn.decomposition import PCA 

# retain 95% of important features 
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)

print(X_pca.shape)
print(X_scaled.shape)

# we can see PCA got rid of 24 cols 

print("Eigen Vectors =\n", pca.components_)
print("\nCo-variance matrix =\n", pca.explained_variance_)

# to see the importance of PCA
# we will train the Logistic Reg model using PCA dataset

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=10)

pca_model = LogisticRegression(max_iter=1000)
pca_model.fit(X_train, y_train)
pca_model.score(X_test, y_test)

# the accuracy is almost similar i.e 96.94 vs 95.94
# Accuracy will be low since info is redduced the advantage is the computation was fast 
