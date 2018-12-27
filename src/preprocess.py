from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from src.binvox_rw import Voxels, write

def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min()) / (255 - x.min()))

    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min

    return x


# crop pics randomly
def random_crop(pic, old_n, new_n):
    if old_n <= new_n:
        return -1
    w = np.random.randint(0, old_n - new_n + 1)
    h = np.random.randint(0, old_n - new_n + 1)

    return pic[w : w + new_n, h : h + new_n]


# add noise to background
def generate_background(pic, n):
    for i in range(n):
        for j in range(n):
            if pic[i][j] == 0:
                pic[i][j] = 255
    return pic


def display_binvox(model):
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    ax.plot3D(xline, yline, zline, 'gray')
    for x in range(0, 32):
      for y in range(0, 32):
        for z in range(0, 32):
          if model[x, y, z] > 0:
            ax.scatter3D(x, y, z, c='g', cmap=None)
    plt.show()
    print('display')
    return ax


def display_pic(pic, n):
    #pic = np.array(n)
    pic = np.add(np.multiply(pic, 0.5), 0.5)
    plt.imshow(pic.reshape([n, n]), cmap='gray')
    #plt.imshow(pic)
    plt.show()

