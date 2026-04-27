import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as py

from sklearn.datasets import make_regression



X,y = make_regression(n_samples=4, n_targets=1, n_features=1, n_informative=1, noise=80, random_state=13)
plt.scatter(X,y)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)
print(reg.coef_)
print(reg.intercept_)

plt.scatter(X,y)
plt.plot(X,reg.predict(X),color='red')
# plt.show()

#
# """
# Start with a random number, and reach the point same as the intercept.
# and creates a slow near to the slope, reg.coef_
# """
#
# # Apply Gradient Descent assuming slope is constant m =78.35
# # Lets start assuming value of intercept b = 0
# y_pred = ((78.35 * X) + 0).reshape(4)
# print(y_pred)
#
# plt.scatter(X,y)
# plt.plot(X,reg.predict(X),color='red', label ='OLS')
# plt.plot(X,y_pred,color='blue', label ='b = 0')
#
# """
# -2 (summation) (y(i)-mx(i) - b)
#
# """
# m = 78.35
# b = 100
#
# loss_slope = -2 * np.sum(y-m*X.ravel() - b)
# print(loss_slope)
#
# # learning rate n = 0.1
# lr = 0.1
# step_size = loss_slope * lr
# print(step_size)
#
# # calculationg new intercept
# b = b - step_size
# print(b)
#
# y_pred1 = ((78.35 * X) + b).reshape(4)
# plt.scatter(X,y)
# plt.plot(X,reg.predict(X),color='red', label ='OLS')
# plt.plot(X,y_pred,color='blue', label ='b = 0')
# plt.plot(X,y_pred1,color='green', label ='b = {}'.format(b))
# plt.legend()
# plt.show()
#
#
# print("************************** Iteration 2 **************************")
# loss_slope = -2 * np.sum(y-m*X.ravel() - b)
# print(loss_slope)
#
# # learning rate n = 0.1
# lr = 0.1
# step_size = loss_slope * lr
# print(step_size)
#
# # calculationg new intercept
# b = b - step_size
# print(b)
#
# y_pred2 = ((78.35 * X) + b).reshape(4)
# plt.scatter(X,y)
# plt.plot(X,reg.predict(X),color='red', label ='OLS')
# plt.plot(X,y_pred,color='blue', label ='b = 0')
# plt.plot(X,y_pred1,color='green', label ='b = {}'.format(b))
# plt.plot(X,y_pred2,color='purple', label ='b = {}'.format(b))
# plt.legend()
# plt.show()
#
# print("************************** Iteration 3 **************************")
# loss_slope = -2 * np.sum(y-m*X.ravel() - b)
# print(loss_slope)
#
# # learning rate n = 0.1
# lr = 0.1
# step_size = loss_slope * lr
# print(step_size)
#
# # calculationg new intercept
# b = b - step_size
# print(b)
#
# y_pred3 = ((78.35 * X) + b).reshape(4)
# plt.scatter(X,y)
# plt.plot(X,reg.predict(X),color='red', label ='OLS')
# plt.plot(X,y_pred,color='blue', label ='b = 0')
# plt.plot(X,y_pred1,color='green', label ='b = {}'.format(b))
# plt.plot(X,y_pred2,color='purple', label ='b = {}'.format(b))
# plt.plot(X,y_pred3,color='grey', label ='b = {}'.format(b))
# plt.legend()
# plt.show()
#
# print("************************** Iteration 4 **************************")
# loss_slope = -2 * np.sum(y-m*X.ravel() - b)
# print(loss_slope)
#
# # learning rate n = 0.1
# lr = 0.1
# step_size = loss_slope * lr
# print(step_size)
#
# # calculationg new intercept
# b = b - step_size
# print(b)
#
# y_pred4 = ((78.35 * X) + b).reshape(4)
# plt.scatter(X,y)
# plt.plot(X,reg.predict(X),color='red', label ='OLS')
# plt.plot(X,y_pred,color='blue', label ='b = 0')
# plt.plot(X,y_pred1,color='green', label ='b = {}'.format(b))
# plt.plot(X,y_pred2,color='purple', label ='b = {}'.format(b))
# plt.plot(X,y_pred3,color='grey', label ='b = {}'.format(b))
# plt.plot(X,y_pred4,color='yellow', label ='b = {}'.format(b))
# plt.legend()
# plt.show()


b = -100
m = 78.35
lr = 0.1
epochs = 1000
for i in range(epochs):

    loss_slope = -2 * np.sum(y - m * X.ravel() - b)
    b = b- (loss_slope * lr)
    y_pred6 = m * X + b
    plt.plot(X,y_pred6)


plt.scatter(X,y)

plt.show()