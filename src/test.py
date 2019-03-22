from src.trainer import *
from src.preprocess import *
import glob
import h5py


# test
if __name__ == '__main__':

    stage1 = Stage1()
    pics = []
    pics2 = []
    for path in glob.glob('../ShapeNetCore_im2avatar/test/03001627/*/views/5.png'):
    #for path in glob.glob('../ShapeNetCore_im2avatar/img_from_Google/*.jpg'):
        pic = cv2.imread(path, 0)
        # resize single pic to the feed the network
        pic = scale(pic)
        pic = cv2.resize(pic, (128, 128))
        pic = np.expand_dims(pic, 3)
        pics.append(pic)

        pic2 = cv2.imread(path, 1)
        pic2 = pic2 / 255
        pic2 = pic2[..., ::-1]
        pic2 = cv2.resize(pic2, (32, 32))
        pics2.append(pic2)

    models1 = []
    for path in glob.glob('../ShapeNetCore_im2avatar/test/03001627/*/models/model_shape_32.h5'):
        f = h5py.File(path, 'r')
        model1 = np.array(f['data'])
        models1.append(model1)


    models2 = []
    for path in glob.glob('../ShapeNetCore_im2avatar/test/03001627/*/models/model_color_32.h5'):
        f = h5py.File(path, 'r')
        model2 = np.array(f['data'])
        models2.append(model2)

    #print(np.array(pics2).shape)
    sample1 = stage1.restore(np.array(pics), num=32, sess_path='../ckp/stage1/dm3.ckpt')
    stage2 = Stage2()
    sample2 = stage2.restore(np.array(pics2), num=32, sess_path='../ckp/stage2/dm2.ckpt')

    for i in range(len(sample1)):
        _ = display_model(sample1[i], sample2[i])
        #_ = display_model(models1[i], models2[i])
        #_ = display_grey_model(sample1[i])
        # _ = display_color_model(sample2[i])



