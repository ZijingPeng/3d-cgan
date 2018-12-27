import os
from src.binvox_rw import *
from src.preprocess import *
import cv2
import zipfile


class Dataset:
    def __init__(self):
        self.train_pic = []
        self.train_vox = []
        self.count = 4

    def batches(self, batch_size):
        self.count = self.count + 1
        if self.count == 5:
            self.train_pic = Dataset.load_pics()
            self.train_vox = Dataset.load_binvoxs()
            self.count = 0

        idx = np.arange(self.train_vox.shape[0])
        np.random.shuffle(idx)
        self.train_pic = self.train_pic[idx]
        self.train_vox = self.train_vox[idx]

        n_batches = self.train_pic.shape[0] // batch_size
        for ii in range(0, int(n_batches * 1)):
            x = self.train_pic[ii * batch_size:(ii + 1) * batch_size]
            y = self.train_vox[ii * batch_size:(ii + 1) * batch_size]

            yield x, y

    @staticmethod
    def load_binvoxs():
        voxels = []
        with zipfile.ZipFile('../data/binvox.zip') as z:
            for filename in z.namelist():
                if '.binvox' in filename:
                    with z.open(filename) as fp:
                        model = read_as_3d_array(fp)
                        model = np.array(model.data.astype(np.float32))
                        model = model * 2 - 1
                        model = np.expand_dims(model, 3)
                        voxels.append(model)

        print(np.array(voxels).shape)
        return np.array(voxels)

    @staticmethod
    def load_pics():
        pics = []
        index = '{}{}.png'.format(np.random.randint(2), np.random.randint(9))
        with zipfile.ZipFile('../data/rendering.zip') as z:
            for filename in z.namelist():
                if index in filename:
                    fp = z.read(filename)
                    image = cv2.imdecode(np.frombuffer(fp, np.uint8), 0)
                    image = cv2.resize(image, (128, 128))
                    image = generate_background(image, 128)
                    image = scale(image)
                    image = np.expand_dims(image, 2)

                    pics.append(image)

        print(np.array(pics).shape)
        return np.array(pics)

    def read_vox_from_path(self, path):
        with open(path, 'rb') as f:
            model = read_as_3d_array(f)
            model = np.array(model.data.astype(np.float32))
            model = model * 2 - 1
            model = np.expand_dims(model, 3)
            # （32， 32， 32， 1）

        return np.array(model)

    def read_pic_from_path(self, path):
        image = cv2.imread(path, 0)
        # image = random_crop(image, 137, 128)
        image = cv2.resize(image, (128, 128))
        image = generate_background(image, 128)
        image = scale(image)
        image = np.expand_dims(image, 2)

        return image
