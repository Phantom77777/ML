import pandas as pd 

# in case of unknown dataset, taking any one feature (X) and output Y and form a custom dataframe
# eg: df = df[['some_feature', 'Y']]

df = pd.read_csv('marketing.csv')
df.head()

df.info()

df.describe()

df.shape

X = df['TV']
y = df['Sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

def linear_regression(X, y):
    N = len(X)
    x_mean = X.mean()
    y_mean = y.mean()

    B1_num = ((X - x_mean) * (y - y_mean)).sum()
    B1_den = ((X - x_mean)**2).sum()
    m = B1_num / B1_den

    c = y_mean - (m*x_mean)

    return (c, m)


  c, m = linear_regression(X_train, y_train)


x_input = int(input())
y_output = c + m * x_input
print("y = ", y_output)
