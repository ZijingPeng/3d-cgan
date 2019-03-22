from matplotlib import pyplot as plt
import numpy as np
import cv2
import matplotlib.colors
from mpl_toolkits import mplot3d


# scale [0, 255] to [-1, 1]
def scale(x):
    x = x / 255
    x = x * 2 - 1

    return x


# display binvox
def display_color_model(model, n=32):
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    ax.plot3D(xline, yline, zline, 'gray')
    for x in range(0, n):
      for y in range(0, n):
        for z in range(0, n):
          if all(i >= 0 for i in model[x, y, z]) and model[x, y, z][0] < 0.9:
            color = matplotlib.colors.rgb2hex(model[x, y, z])
            ax.scatter3D(x, y, z, c=color,  cmap=None)
    plt.show()
    print('display')
    return ax


def display_grey_model(model, n=32):
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    ax.plot3D(xline, yline, zline, 'gray')
    for x in range(0, n):
      for y in range(0, n):
        for z in range(0, n):
          if model[x, y, z] > 0:
            ax.scatter3D(x, y, z, c='g', cmap=None)
    plt.show()
    print('display')
    return ax


#display view
def display_view(view):
    view = np.add(np.multiply(view, 0.5), 0.5)
    cv2.imshow("image", view)
    cv2.waitKey()


# crop view randomly
def random_crop(view, old_n, new_n):
    if old_n <= new_n:
        return -1
    w = np.random.randint(0, old_n - new_n + 1)
    h = np.random.randint(0, old_n - new_n + 1)

    return view[w : w + new_n, h : h + new_n]


# add noise to background
def generate_random_background(view, n):
    for i in range(n):
        for j in range(n):
            if view[i][j] == 0:
                view[i][j] = np.random.randint(156) + 100
    return view


def display_model(shape, color, n=32):
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    ax.plot3D(xline, yline, zline, 'gray')
    for x in range(0, n):
      for y in range(0, n):
        for z in range(0, n):
          if shape[x, y, z] > 0 and all(i >= 0 for i in color[x, y, z]):
            c = matplotlib.colors.rgb2hex(color[x, y, z])
            ax.scatter3D(x, y, z, c=c,  cmap=None)
    plt.show()
    print('display')
    return ax