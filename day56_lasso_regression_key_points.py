import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



print("How are coefficients affected")
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size=0.2, random_state=2)

coef = []
r2_scores = []
for i in [0,0.1,1,10]:
    reg = Lasso(alpha=i)
    reg.fit(X_train, y_train)
    coef.append(reg.coef_.tolist())
    y_pred = reg.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))


plt.figure(figsize=(14,9))
plt.subplot(221)
plt.bar(data.feature_names, coef[0])
plt.title('Alpha = 0, r2_score = {}'.format(round(r2_scores[0],2)))

plt.subplot(222)
plt.bar(data.feature_names, coef[1])
plt.title('Alpha = 0.1, r2_score = {}'.format(round(r2_scores[1],2)))

plt.subplot(223)
plt.bar(data.feature_names, coef[2])
plt.title('Alpha = 1, r2_score = {}'.format(round(r2_scores[2],2)))

plt.subplot(224)
plt.bar(data.feature_names, coef[3])
plt.title('Alpha = 10, r2_score = {}'.format(round(r2_scores[3],2)))

plt.show()



print("Higher coeffs are affected more")
"""
The value of alpha increases, the value of the features/cols reaches 0

"""
print("************* Graph1 ****************")
alphas = [0,0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
coefs = []

for i in alphas:
    reg = Lasso(alpha=i)
    reg.fit(X_train, y_train)

    coefs.append(reg.coef_.tolist())

input_array = np.array(coefs)
coef_df = pd.DataFrame(input_array, columns=data.feature_names)
coef_df['alpha'] = alphas
coef_df.set_index('alpha')


print("****************** Graph 2 **************")
alphas = [0,0.0001,0.0005, 0.001,0.005,0.1,0.5,1,5,10]
coefs = []
for i in alphas:
    reg = Lasso(alpha=i)
    reg.fit(X_train,y_train)
    coefs.append(reg.coef_.tolist())

input_array = np.array(coefs).T
plt.figure(figsize=(15,8))
plt.plot(alphas,np.zeros(len(alphas)),color = 'black', linewidth = 5)
for i in range(input_array.shape[0]):
    plt.plot(alphas,input_array[i], label=data.feature_names[i])
plt.legend()
plt.show()


print("Impact on Bias and Variance")
# m = 100
# X = 5 * np.random.rand(m,1) - 2
# y = 0.7 * X ** 2 - 2 * X + 3 + np.random.randn(m,1)
#
# plt.scatter(X,y)
# plt.show()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
#
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(degree=10)
# X_train = poly.fit_transform(X_train)
# X_test = poly.transform(X_test)
#
# from mlxtend.evaluate import bias_variance_decomp
# alphas = np.linspace(0,30,100)
# loss = []
# bias = []
# variance = []
#
# for i in alphas:
#     reg = Lasso(alpha=i)
#     avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
#         reg, X_train, y_train, X_test, y_test, loss = 'mse', random_seed=123
#     )
#     loss.append(avg_expected_loss)
#     bias.append(avg_bias)
#     variance.append(avg_var)
#
# plt.plot(alphas, loss, label = 'loss')
# plt.plot(alphas, bias, label = 'bias')
# plt.plot(alphas, variance, label = 'variance')
# plt.xlabel('Alpha')
# plt.legend()
# plt.show()

print("Effects of regularization on Loss function")
from sklearn.datasets import make_regression
X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20,random_state=1)
plt.scatter(X,y)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)
print(reg.coef_) # 82.48449929
print(reg.intercept_) # 4.054028777621713


def cal_loss(m, alpha):
    return np.sum((y-m*X.ravel() - 4)**2) + alpha * abs(m)

def predict(m):
    return m*X + 4.05

m = np.linspace(-45,100,100)
plt.figure(figsize=(12,12))
for j in [0,100,500, 1000, 2500]:
    loss = []
    for i in range(m.shape[0]):
        loss_i = cal_loss(m[i],j)
        loss.append(loss_i)
    plt.plot(m, loss, label = 'alpha = {}'.format(j))

plt.legend()
plt.xlabel('Alpha')
plt.ylabel('Loss')
plt.show()

