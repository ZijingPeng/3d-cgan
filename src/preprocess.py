from matplotlib import pyplot as plt
import numpy as np
import cv2


# scale [0, 255] to [-1, 1]
def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min()) / (255 - x.min()))

    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min

    return x


# preprocess pics before train
def pics_preprocess(pics, dim_in, dim_out):
    out = []
    for pic in pics:
        pic = generate_white_background(pic, dim_in)
        pic = cv2.resize(pic, (dim_out, dim_out))
        out.append(pic)
    out = np.array(out)
    return out


# add white to background
def generate_white_background(pic, n):
    for i in range(n):
        for j in range(n):
            if pic[i][j] == 0:
                pic[i][j] = 255
    return pic


# display binvox
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


#display pic
def display_pic(pic, n):
    pic = np.add(np.multiply(pic, 0.5), 0.5)
    plt.imshow(pic.reshape([n, n]), cmap='gray')
    plt.show()


# resize single pic to feed the network
def resize_input_pic(pic):
    pic = cv2.resize(pic, (128, 128))
    pic = np.reshape(pic, (128, 128, 1))
    return pic
