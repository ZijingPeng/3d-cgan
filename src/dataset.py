from src.binvox_rw import *
from src.preprocess import *
import cv2
import zipfile
import copy


class Dataset:
    def __init__(self, prob, pic_dim_in=137, pic_dim_out=128, vox_dim=32):
        self.count = 0
        self.prob = prob
        self.pic_dim_in = pic_dim_in
        self.pic_dim_out = pic_dim_out
        self.vox_dim = vox_dim

        # load images
        pics = self.load_pics()
        self.pics = pics_preprocess(copy.deepcopy(pics), self.pic_dim_in, self.pic_dim_out)
        self.pics = np.reshape(self.pics, (-1, self.pic_dim_out, self.pic_dim_out, 1))

        # load models
        self.voxs = self.load_binvoxs()
        self.voxs = np.reshape(self.voxs, (-1, self.vox_dim, self.vox_dim, self.vox_dim, 1))


    def batches(self, batch_size):
        self.count = self.count + 1
        if self.count == 5:
            # reload images
            pics = self.load_pics()
            self.pics = pics_preprocess(copy.deepcopy(pics), self.pic_dim_in, self.pic_dim_out)
            self.pics = np.reshape(self.pics, (-1, self.pic_dim_out, self.pic_dim_out, 1))

            # load models
            self.voxs = self.load_binvoxs()
            self.voxs = np.reshape(self.voxs, (-1, self.vox_dim, self.vox_dim, self.vox_dim, 1))

            self.count = 0

        idx = np.arange(self.pics.shape[0])
        np.random.shuffle(idx)
        self.pics = self.pics[idx]
        self.voxs = self.voxs[idx]

        n_batches = self.pics.shape[0] // batch_size
        for ii in range(0, int(n_batches * self.prob)):
            x = self.pics[ii * batch_size:(ii + 1) * batch_size]
            y = self.voxs[ii * batch_size:(ii + 1) * batch_size]

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
                        voxels.append(model)

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
                    pics.append(image)
        print(index)
        return np.array(pics)

    @staticmethod
    def read_vox_from_path(path):
        with open(path, 'rb') as f:
            model = read_as_3d_array(f)
            model = np.array(model.data.astype(np.float32))
            model = model * 2 - 1
            model = np.expand_dims(model, 3)
            # （32， 32， 32， 1）

        return np.array(model)
