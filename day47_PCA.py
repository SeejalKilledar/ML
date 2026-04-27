from statistics import covariance

import numpy as np
import pandas as pd


np.random.seed(23) # random numbers will be generated same everytime

# mean vector, creates 3D mean Vector
mu_vec1 = np.array([0,0,0])
print(mu_vec1)

# 3*3 covariance matrix
cov_mat1 = np.array([[1,0,0],[0,1,0], [0,0,1]])
print(cov_mat1)

# creates 20 random 3D points
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)
print(class1_sample)
print(class1_sample.shape)

df = pd.DataFrame(class1_sample, columns=['feature1', 'feature2', 'feature3'])
df['target'] = 1

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0], [0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20)
df1 = pd.DataFrame(class2_sample, columns=['feature1', 'feature2', 'feature3'])
df1['target'] = 0

df = df._append(df1, ignore_index=True)

print(df.sample(40))

# Plot the above 3D data, see how it looks
import plotly.express as px
fig = px.scatter_3d(df, x= df['feature1'], y = df['feature2'], z = df['feature3'],
                    color=df['target'].astype('str'))

fig.update_traces(marker = dict(size=12, line=dict(width = 2, color = 'DarkSlateGrey')),
                  selector = dict(mode = 'markers'))

#fig.show()

print("************ Step1: Mean Centering *************")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df.iloc[:,0:3] = scaler.fit_transform(df.iloc[:,0:3])

print("************** Step2: Covariance Matrix *************")
covariance_matrix = np.cov([df.iloc[:,0], df.iloc[:,1], df.iloc[:,2]])
print(covariance_matrix)

print("******* Step3: Eigen Values and Eigen Vector *********")
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
print(eigen_values) # will get 3 coz 3D space
print(eigen_vectors) # will get 3 coz 3D space

print("Fetch 1st 2 eigen vectors")
pc = eigen_vectors[0:2]
print(pc)

print("Transform 3D space to 2D space (40,2)")
# Data - 40,3
# PC -  3, 2
# new data is 40,2
transformed_df = np.dot(df.iloc[:,0:3], pc.T) # all 40 values to be transformed
new_df = pd.DataFrame(transformed_df, columns=['PC1', 'PC2'])
new_df['target'] = df['target'].values
print(new_df.head())

fig = px.scatter(x=new_df['PC1'], y=new_df['PC2'], color=new_df['target'],
                 color_discrete_sequence=px.colors.qualitative.G10)
fig.update_traces(marker = dict(size=12, line=dict(width = 2, color = 'DarkSlateGrey')),
                  selector = dict(mode = 'markers'))
fig.show()

# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import proj3d
# from matplotlib.patches import FancyArrowPatch
#
#
# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs
#
#     def draw(self, renderer):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         FancyArrowPatch.draw(self, renderer)
#
#     def do_3d_projection(self, renderer=None):
#         if renderer is None:
#             # return a neutral z-value
#             return 0
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#         # return z value for depth sorting
#         return np.mean(zs)
#
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot(df['feature1'], df['feature2'], df['feature3'], 'o', markersize=8, color='blue', alpha=0.2)
# ax.plot([df['feature1'].mean()], [df['feature2'].mean()], [df['feature3'].mean()], 'o', markersize=10, color='red',
#         alpha=0.5)
# for v in eigen_vectors.T:
#     a = Arrow3D([df['feature1'].mean(), v[0]], [df['feature2'].mean(), v[1]], [df['feature3'].mean(), v[2]],
#                 mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
#     ax.add_artist(a)
# ax.set_xlabel('x_values')
# ax.set_ylabel('y_values')
# ax.set_zlabel('z_values')
#
# plt.title('Eigenvectors')
#
# plt.show()

