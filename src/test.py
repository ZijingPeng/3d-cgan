from src.trainer import Stage
from src.preprocess import *
import glob

# test
if __name__ == '__main__':
    stage = Stage()

    pics = []
    for path in glob.glob('../data/renders/*.png'):
        pic = cv2.imread(path, 0)
        # resize single pic to the feed the network
        pic = resize_input_pic(pic)
        pics.append(pic)

    stage.restore(pics, 5, sess_path='../ckp/dm0.ckpt')