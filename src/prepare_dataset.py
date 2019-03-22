import h5py
import glob
import os
import numpy as np
from scipy.ndimage import zoom
import cv2

if __name__ == '__main__':

    # zoom 64x64x64 to 32x32x32
    for path in glob.glob(
            '../ShapeNetCore_im2avatar/ShapeNetCore_im2avatar/03001627/*/models/model_normalized_64.h5'):
        f = h5py.File(path, 'r')
        print(path)

        # Get the data
        model = np.array(f['data'])

        model = zoom(model, (0.5, 0.5, 0.5, 1), order=0)

        # dilate models after zomming
        kernel = np.ones((2, 2), np.uint8)
        dilations = []
        for i in model:
            dilation = cv2.dilate(i, kernel, iterations=1)
            dilations.append(dilation)
        dilations = np.array(dilations)

        data_file = h5py.File(path.replace("normalized_64", "color_32"), 'w')
        data_file.create_dataset('data', data=dilations, compression='gzip')
        data_file.close()


    # write 1-channel model file for first stage train
    for path in glob.glob('../ShapeNetCore_im2avatar/ShapeNetCore_im2avatar/03001627/*/models/model_color_32.h5'):
        f = h5py.File(path, 'r')
        a_group_key = list(f.keys())[0]
        print(path)

        output = []
        # Get the data
        datas = list(f[a_group_key])
        for data in datas:
            data = data[:, :, :1]
            data = data > -1
            data = data * 2 - 1
            output.append(data)

        data_file = h5py.File(path.replace("color", "shape"), 'w')
        data_file.create_dataset('data', data=output, compression='gzip')
        data_file.close()


    # delete useless files
    def delete():
        for path in glob.glob('../ShapeNetCore_im2avatar/ShapeNetCore_im2avatar/03001627/*/models/model_normalized_64.h5'):
            print(path)
            os.remove(path)


    # rename .png files
    data_index = ['0.000', '0.523', '1.047', '1.570', '2.093', '2.617', '3.140', '3.663', '4.187', '4.710', '5.233',
                  '5.757']
    index = 0
    for path in glob.glob('../ShapeNetCore_im2avatar/ShapeNetCore_im2avatar/03001627/*/views/*.png'):
        print(path)
        os.rename(path, path.replace(data_index[index], str(index)))
        index += 1
        if index == 12:
            index = 0


    # remove useless .h5 files
    for path in glob.glob('../ShapeNetCore_im2avatar/ShapeNetCore_im2avatar/03001627/*/models/*coor.h5'):
        print(path)
        os.remove(path)


    # I changed a more precise shape file
    def changeShapeFile():
        from src.binvox_rw import read_as_3d_array
        for path in glob.glob('../ShapeNetCore_im2avatar/train/03001627/*/models/model_shape_32.h5'):
            path_vox = path.replace('ShapeNetCore_im2avatar', 'data').replace('train', 'binvox').replace('03001627',
                                                                                                         'binvox') \
                .replace('models\model_shape_32.h5', 'model.binvox')

            with open(path_vox, 'rb') as f:
                model = read_as_3d_array(f)
                model = np.array(model.data.astype(np.float32))
                model = model * 2 - 1
                # reverse it
                model = np.flip(model, 0)
                model = np.flip(model, 1)

                data_file = h5py.File(path, 'w')
                data_file.create_dataset('data', data=model, compression='gzip')
                data_file.close()
                print(path)


