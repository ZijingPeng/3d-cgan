from src.preprocess import *
import cv2
import glob
import h5py


# Dataset for stage1
class Dataset1:
    def __init__(self, prob=0.8):
        self.count = -1
        self.prob = prob

        # load views and models
        self.views = self.load_grey_views()
        self.models = self.load_shape_models()


    def batches(self, batch_size=32):
        self.count = self.count + 1
        if self.count == 10:
            # reload views
            self.views = self.load_grey_views()
            self.models = self.load_shape_models()

            self.count = 0

        idx = np.arange(self.views.shape[0])
        np.random.shuffle(idx)
        self.views = self.views[idx]
        self.models = self.models[idx]

        n_batches = self.views.shape[0] // batch_size
        for ii in range(0, int(n_batches * self.prob)):
            x = self.views[ii * batch_size:(ii + 1) * batch_size]
            y = self.models[ii * batch_size:(ii + 1) * batch_size]

            yield x, y


    def load_grey_views(self):
        views = []
        index = '{}.png'.format(np.random.randint(12))
        for path in glob.glob('../ShapeNetCore_im2avatar/train/03001627/*/views/{}'.format(index)):

            view = cv2.imread(path, 0)
            # scale to (-1, 1)
            view = scale(view)
            views.append(view)

        return np.expand_dims(np.array(views), 3)


    def load_shape_models(self):
        models = []
        for path in glob.glob('../ShapeNetCore_im2avatar/train/03001627/*/models/model_shape_32.h5'):
            f = h5py.File(path, 'r')
            model = np.array(f['data'])
            #_ = display_grey_model(model)
            models.append(model)

        return np.array(models)


# Dataset for stage2
class Dataset2:
    def __init__(self, prob=0.8):
        self.count = -1
        self.prob = prob

        # load views and models
        self.color_views = self.load_color_views()
        self.color_models = self.load_color_models()


    def batches(self, batch_size=32):
        self.count = self.count + 1
        if self.count == 10:
            # reload views
            self.color_views = self.load_color_views()
            self.color_models = self.load_color_models()

            self.count = 0

        idx = np.arange(self.color_views.shape[0])
        np.random.shuffle(idx)
        self.color_views = self.color_views[idx]
        self.color_models = self.color_models[idx]

        n_batches = self.color_views.shape[0] // batch_size
        for ii in range(0, int(n_batches * self.prob)):
            x = self.color_views[ii * batch_size:(ii + 1) * batch_size]
            y = self.color_models[ii * batch_size:(ii + 1) * batch_size]

            yield x, y


    def load_color_models(self):
        models = []
        for path in glob.glob('../ShapeNetCore_im2avatar/train/03001627/*/models/model_color_32.h5'):
            f = h5py.File(path, 'r')
            model = np.array(f['data'])
            models.append(model)

        print('models')
        return np.array(models)


    def load_color_views(self):
        color_views = []
        index = '{}.png'.format(np.random.randint(12))

        for path in glob.glob('../ShapeNetCore_im2avatar/train/03001627/*/views/{}'.format(index)):
            color_view = cv2.imread(path, 1)
            color_view = color_view / 255
            color_view = color_view[..., ::-1]
            color_view = cv2.resize(color_view, (32, 32))
            color_views.append(color_view)

        print(index)
        return np.array(color_views)
