# Dot product
import time
import numpy
import array

# 8 bytes size int
a = array.array('q')
for i in range(100000):
    a.append(i);
p
b = array.array('q')
for i in range(100000, 200000):
    b.append(i)

# classic dot product of vectors implementation
tic = time.process_time()
dot = 0.0

for i in range(len(a)):
    dot += a[i] * b[i]

toc = time.process_time()

print("dot_product = " + str(dot));
print("Computation time = " + str(1000 * (toc - tic)) + "ms")

n_tic = time.process_time()
n_dot_product = numpy.dot(a, b)
n_toc = time.process_time()

print("\nn_dot_product = " + str(n_dot_product))
print("Computation time = " + str(1000 * (n_toc - n_tic)) + "ms")


#%% outer multiplication
# Outer product
import time
import numpy
import array

a = array.array('i')
for i in range(200):
    a.append(i);

b = array.array('i')
for i in range(200, 400):
    b.append(i)

# classic outer product of vectors implementation
tic = time.process_time()
outer_product = numpy.zeros((200, 200))

for i in range(len(a)):
    for j in range(len(b)):
        outer_product[i][j] = a[i] * b[j]

toc = time.process_time()

print("outer_product = " + str(outer_product));
print("Computation time = " + str(1000 * (toc - tic)) + "ms")

n_tic = time.process_time()
outer_product = numpy.outer(a, b)
n_toc = time.process_time()

print("outer_product = " + str(outer_product));
print("\nComputation time = " + str(1000 * (n_toc - n_tic)) + "ms")

#%%
# Element-wise multiplication
import time
import numpy
import array

a = array.array('i')
for i in range(50000):
    a.append(i);

b = array.array('i')
for i in range(50000, 100000):
    b.append(i)

# classic element wise product of vectors implementation
vector = numpy.zeros((50000))

tic = time.process_time()

for i in range(len(a)):
    vector[i] = a[i] * b[i]

toc = time.process_time()

print("Element wise Product = " + str(vector));
print("\nComputation time = " + str(1000 * (toc - tic)) + "ms")

n_tic = time.process_time()
vector = numpy.multiply(a, b)
n_toc = time.process_time()

print("Element wise Product = " + str(vector));
print("\nComputation time = " + str(1000 * (n_toc - n_tic)) + "ms")

#%% maximum likelihood esitmation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats

N = 100
x = np.linspace(0, 20, N)
epsilon = np.random.normal(loc = 0.0, scale = 5.0, size = N)
y = 3 * x + epsilon

df = pd.DataFrame({'y':y, 'x':x})
# print(df)
df['constant'] = 1
sns.regplot(df.x, df.y)
plt.show()



#%%

import pandas as pd
import numpy as np

def expection_max(data, max_iter = 1000):
    data = pd.DataFrame(data)
    mu0 = data.mean()
    c0 = data.cov()

    for j in range(max_iter):
        w = []
        for i in data:
            wk = (5 + len(data)) / (5 + np.dot(np.transpose(i
                - mu0), np.linalg.solve(c0, (i - mu0))))
            w.append(wk)
            w = np.array(w)

        mu = (np.dot(w, data)) / (np.sum(w))

        c = 0
        for i in range(len(data)):
            c += w[i] * np.dot((data[i] - mu0), (np.transpose(data[i] - mu0)))
        cov = c / len(data)

        mu0 = mu
        c0 = cov

    return mu0, c0

#%% test of library functions
from scipy.stats import norm
print(norm.pdf(4, 2, 10) == norm.pdf(0, 2, 10))

#%% MLE
'''
Compare the likelihood of the random samples to the two distributions
'''
import numpy as np
from scipy.stats import norm

def compare_data_to_dist(x, mu_1 = 5, mu_2 = 7, sd_1 = 3, sd_2 = 3):
    ll_1 = 0
    ll_2 = 0
    # log likelihood function is the sum of the log pdf
    for i in x:
        ll_1 += np.log(norm.pdf(i, mu_1, sd_1))
        ll_2 += np.log(norm.pdf(i, mu_2, sd_2))
    print("hello world")
    print("the LL of x for mu = {:d} and sd = {:d} is {:.4f}".format(mu_1, sd_1, ll_1))
    print("the LL of x for mu = {:d} and sd = {:d} is {:.4f}".format(mu_2, sd_2, ll_2))

x = [4, 5, 7, 8, 8, 9, 10, 5, 2, 3, 5, 4, 8, 9]
compare_data_to_dist(x)

def plot_ll(x):
    plt.figure(figsize = (5, 8))
    plt.title("maximum likelihood functions")
    plt.xlabel("mean estimate")
    plt.ylabel("log likelihood")
    plt.ylim(-40, -30)
    plt.xlim(0, 12)
    # plt.show()

    mu_set = np.linspace(0, 16, 1000)
    sd_set = [.5, 1, 1.5, 2.5, 3, 3.5]
    max_val = max_val_location = None

    for i in sd_set:
        ll_array = []
        for j in mu_set:
            temp_mm = 0
            for k in x:
                temp_mm += np.log(norm.pdf(k, j, i))  # ll function
            ll_array.append(temp_mm)

            if (max_val is None):
                max_val = max(ll_array)
            elif (max(ll_array) > max_val):
                max_val = max(ll_array)
                max_val_location = j

        # plot the results
        plt.plot(mu_set, ll_array, label="sd: %.1f" % i)
        print("the max ll for sd {:0.2f} is {:.2f}".format(i, max(ll_array)))
        plt.axvline(x=max_val_location, color='black', ls='-.')
        plt.legend(loc='lower left')

    plt.show()

plot_ll(x)

#%% test of print fuinctions
print("hello", "how are you?", sep = "---")

print('')
print("difference")
print('\n')
print("hello")

import os
print("hello, " + os.getlogin() + "!, how are you!")

print("th is is the just a test {:05d}".format(int(235345634563.234444)))



#%% MLE maximum likelihood estimation
# https://www.bogotobogo.com/python/scikit-learn/Maximum-Likelyhood-Estimation-MLE.php

import matplotlib.pyplot as plt
import numpy as np

phi = np.arange(0.01, 1.0, 0.01)
j1 = -np.log(phi)
j0 = -np.log(1 - phi)
plt.plot(phi, j1, color = 'green', label = 'y = 1')
plt.plot(phi, j0, color = 'blue', label = 'y = 0')
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.title('cost functions')
plt.grid(False)
plt.legend(loc = 'upper center')
plt.show()

#%% different distribution tests
from scipy.stats import chi2
# Calculate the probability density function for values of x in [0;10]
x = np.linspace(start = 0, stop = 10, num = 200)
#
_ = plt.figure(num = 1, figsize = (10, 8))
_ = plt.plot(x, chi2.pdf(x, df = 1), color = "black", label = "k = 1")
for i in range(2, 7):
    _ = plt.plot(x, chi2.pdf(x, df = i), label = "k = " + str(i))
_ = plt.ylim((0, 0.6))
_ = plt.title("Probability density function of $\\chi^2_k$")
_ = plt.ylabel("Density")
_ = plt.legend()
plt.show()



# import numpy
# def triangle(n):
#     # Create the mesh in barycentric coordinates
#     bary = (
#         numpy.hstack(
#             [[numpy.full(n - i + 1, i), numpy.arange(n - i + 1)] for i in range(n + 1)]
#         )
#         / n
#     )
#     bary = numpy.array([1.0 - bary[0] - bary[1], bary[1], bary[0]])
#
#     # Some applications rely on the fact that not values like -1.4125e-16 appear.
#     bary[bary < 0.0] = 0.0
#     bary[bary > 1.0] = 1.0
#
#     cells = []
#     k = 0
#     for i in range(n):
#         j = numpy.arange(n - i)
#         cells.append(numpy.column_stack([k + j, k + j + 1, k + n - i + j + 1]))
#         #
#         j = j[:-1]
#         cells.append(
#             numpy.column_stack([k + j + 1, k + n - i + j + 2, k + n - i + j + 1])
#         )
#         k += n - i + 1
#
#     cells = numpy.vstack(cells)
#
#     return bary, cells

# bary, cells = triangle(3)


# def display_path(Gd):
#     nt = Gd.shape[0]
#     ns = Gd.shape[1]
#
#     t = np.linspace(0, 1, nt)
#     s = np.linspace(0, 1, ns)
#
#     ss, tt = np.meshgrid(s, t)
#     ind_t, ind_s = np.nonzero(Gd)
#     s_path = [s[i] for i in list(ind_s)] # define horizontal coordinates
#     # fig = plt.figure()
#     fig, ax = plt.subplots()
#     plt.plot(ss, tt, 'k.')
#     plt.plot(s_path, list(t[::-1]), 'bo')
#     # plt.show()


#%%
import numpy as np
import itertools

x = np.arange(-1, 2)
y = np.arange(-1, 2)

c = list(itertools.product(x, y))

d = np.array(c)
a = np.array([3, 5])
d = d + a
print(d)



#%%

def insert_new_waypoint(pos, grid_size, path):
    row = pos[0]
    col = pos[1]

    n_row = grid_size[0]    # number of elements in the row
    n_col = grid_size[1]    # number of elements in the column
    new_path = np.zeros([1, n_row * n_col])
    print(new_path)
    print(new_path.shape)

    new_ind = n_row * row + col
    print(new_ind)

    new_path[new_ind] = True

    path = np.vstack(path, new_path)
    return path


pos = [3, 3]
# print(pos[0])
# print(pos[0])
nt = 5
ns = 5
grid_size = [nt, ns]
p = np.zeros([1, nt * ns])
# print(p)
# print(p.shape)

p_new = insert_new_waypoint(pos, grid_size, p)



#%%

def pos_candidates(x, xlim, y, ylim):

    pos_x = []
    pos_y = []

    for i in range(3):
        x_temp = x + i - 1
        y_temp = y + i - 1

        if (x_temp >= 0) and (x_temp < xlim):
            pos_x.append(x_temp)

        if (y_temp >= 0) and (y_temp < ylim):
            pos_y.append(y_temp)

    return np.array(pos_x), np.array(pos_y)

a, b = pos_candidates(8, 10, 1, 10)
print(a)
print(b)

#%%

import numpy as np
import matplotlib.pyplot as plt

a = np.eye(10)
print(a)
plt.imshow(a)

plt.show()

#%%
# importing os module
import os

# path
path = os.getcwd() + "/test_2"

# Create the directory
# 'GeeksForGeeks' in
# '/home / User / Documents'
try:
    os.mkdir(path)
except OSError as error:
    print(error)

#%%
try:
  print(x)
except:
  print("An exception occurred")

#%%

try:
  print(x)
except NameError:
  print("Variable x is not defined")
except:
  print("Something else went wrong")

try:
  print("Hello")
except:
  print("Something went wrong")
else:
  print("Nothing went wrong")


try:
  print(x)
except:
  print("Something went wrong")
finally:
  print("The 'try except' is finished")

try:
  f = open("demofile.txt")
  f.write("Lorum Ipsum")
except:
  print("Something went wrong when writing to the file")
finally:
  f.close()


  #%%
  import os.path
  from os import path


  def main():

      print("File exists:" + str(path.exists('guru99.txt')))
      print("File exists:" + str(path.exists('career.guru99.txt')))
      print("directory exists:" + str(path.exists('myDirectory')))


  if __name__ == "__main__":
      main()



#%%
import os.path
from os import path

def main():

	print ("Is it File?" + str(path.isfile('guru99.txt')))
	print ("Is it File?" + str(path.isfile('myDirectory')))

if __name__== "__main__":
	main()



#%%
# Suppose this is foo.py.

print("before import")
import math

print("before functionA")
def functionA():
    print("Function A")

print("before functionB")
def functionB():
    print("Function B {}".format(math.sqrt(100)))

print("before __name__ guard")
if __name__ == '__main__':
    functionA()
    functionB()
print("after __name__ guard")

#%%
import os
import datetime
# using now() to get current time
current_time = datetime.datetime.now()

a = os.getcwd() + "/fig_" + str(current_time.hour) + "_" + str(current_time.minute)

print(os.path.exists(a))
# print(a)
if not(os.path.exists(a)):
    try:
        os.mkdir(a)
        print("Successfully created a new directory")
    except:
        print("Failed to create new directory")
else:
    print("Path is already existed")

#%%
# Getting current date and time using
# now().

# importing datetime module for now()
import datetime

# using now() to get current time
current_time = datetime.datetime.now()

# Printing value of now.
print("Time now at greenwich meridian is : "
      , end="")
print(current_time)

#%%
from PIL import Image

background = Image.open(os.getcwd() + "/fig_12_1" + "/path/" + "{:03d}".format(49) + ".png")
foreground = Image.open(os.getcwd() + "/fig_12_1" + "/mean/" + "{:03d}".format(49) + ".png")

# background.paste(foreground, (0, 0))
background = Image.blend(foreground, background, alpha = .5)
# background.paste(foreground, (0, 0), foreground)
background.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.mlab as mlab


def get_rgba_bitmap(fig):
    fig.canvas.draw()
    tab = fig.canvas.copy_from_bbox(fig.bbox).to_string_argb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(tab, dtype=np.uint8).reshape(nrows, ncols, 4)

def black_white_to_black_transpa(rgba):
    rgba[:, :, 3] = 255 - rgba[:, :, 0]
    rgba[:, :, 0:3] = 0

def over(rgba1, rgba2):
    if rgba1.shape != rgba2.shape:
        raise ValueError("rgba1 and rgba2 shall have same size")
    alpha = np.expand_dims(rgba1[:, :, 3] / 255., 3)
    rgba =  np.array(rgba1 * alpha + rgba2 * (1.-alpha), dtype = np.uint8)
    return rgba[:, :, 0:3]


# fig 1)
fig1 = plt.figure(facecolor = "white")
fig1.set_dpi(300)
ax1 = fig1.add_subplot(1, 1, 1, aspect = "equal", facecolor = "black")
ax1.add_artist(plt.Circle((0., 0., .5), color =   "white"))
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
bitmap_rgba1 = get_rgba_bitmap(fig1)
black_white_to_black_transpa(bitmap_rgba1)

# fig 2
fig2 = plt.figure(facecolor = "white")
fig2.set_dpi(300)
delta = 0.025
ax2 = fig2.add_subplot(1, 1, 1, aspect = "equal", facecolor = "black")
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2-Z1  # difference of Gaussians
im = ax2.imshow(Z, interpolation='bilinear', cmap=cm.jet,
                origin='lower', extent=[-5, 5, -5, 5],
                vmax=abs(Z).max(), vmin=-abs(Z).max())
bitmap_rgba2 = get_rgba_bitmap(fig2)

# now saving the composed figure
fig = plt.figure()
fig.patch.set_alpha(0.0)
ax = fig.add_axes([0., 0., 1., 1.])
ax.patch.set_alpha(0.0)
ax.imshow(over(bitmap_rgba1, bitmap_rgba2))
plt.axis('off')
fig.savefig("test_transpa.png", dpi=300)

plt.show()






