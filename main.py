
from fastai import dataset

from lib.dataset3D import *
from lib.transforms3D import *
from lib.cube3D import *
from lib.cnn3D import *


def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i],cmap='gray',interpolation='nearest')


if __name__ == '__main__':

    #Prepare paths to csv
    PATH_CUBES_NEW = Path('data/')
    train_csv = PATH_CUBES_NEW/'train.csv'
    test_csv = PATH_CUBES_NEW/'test.csv'
    sub_train_csv = PATH_CUBES_NEW/'sample_train_local.csv'
    train_path = PATH_CUBES_NEW/'train'
    test_path = PATH_CUBES_NEW/'test'

    #Load data
    dfTrain = pd.read_csv(str(sub_train_csv))


    #Define some 3D augmentations
    aug_tfms_3D = [
        Flip(1),
        #Zoom(0,-0.5),
        Lighting(0.3,0.3),
        #Stretch(1.5,0),,
        #Blur(2),
        Rotate(45),
        ]

    #Define data generator
    sz=32
    bs=64
    dataset.open_image = open_cube
    tfms = tfms_from_stats(None, sz, aug_tfms=aug_tfms_3D)
    data = ImageClassifier3DData.from_csv(PATH_CUBES_NEW, 'train', sub_train_csv, tfms = tfms, bs=bs, test_name='test',
                                        suffix='.npy', skip_header=True, num_workers=None)

    #Define 3D model
    net = nn.Sequential(
        C3D(1, 64, 3, 1),
        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        C3D(64, 128, 3, 1),
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        Flatten(),
        nn.Linear(131072, 2)
    )

    #Convert 3D model to fastai
    bm = BasicModel(net, name='simplenet')
    learn = ConvLearner(data, bm)
    learn.crit = nn.CrossEntropyLoss()
    learn.opt_fn = optim.Adam
    learn.unfreeze()
    learn.metrics = [accuracy]
    lr = 1e-3

    #Show 3D model
    print(model_summary(learn.model, [torch.rand(1, 1, learn.data.sz, learn.data.sz, learn.data.sz)]))

    #Find learning rate
    #lrf = learn.lr_find()

    #Train 3D model
    #learn.fit(lr,3,cycle_len=1)

    #Show transforms
    mid = sz // 2
    for x, _ in iter(data.trn_dl):
        print("Batch size of", len(x))
        for s in range(len(x)):
            cb = x[s][0].reshape(sz, sz, sz)
            plots([
                cb[mid-1, :, :], cb[mid, :, :], cb[mid+1, :, :],
                cb[:, mid - 1, :], cb[:, mid, :], cb[:, mid + 1, :],
                cb[:, :, mid - 1], cb[:, :, mid], cb[:, :, mid + 1]
                   ],
                  figsize=(12, 18), rows=3, titles=None)
            plt.show()
