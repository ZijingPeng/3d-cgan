# 3d-cgan
Image-to-Model Translation with Conditional Adversarial Nets 

### Getting started

```
git clone https://github.com/ZijingPeng/3d-cgan.git
```

### Datasets

- ShapeNet rendered images <http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz>
- ShapeNet voxelized models <http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz>

I used the datasets in [3D-R2N2](https://github.com/chrischoy/3D-R2N2)

### Checkpoints

Download pretrained checkpoints [here](https://drive.google.com/open?id=1OPMyMb5frKPJLxU24JP5_l_rgiJq8W-V). I trained for 100 epochs.

### How to run

If you want to use pretrained checkpoints, download and copy to `ckp`  folder, then run the `test.py` .

If you want to train yourself, download datasets and copy to `data` folder, then run `train.py`.

### License

MIT License






