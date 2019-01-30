from fastai.conv_learner import *
from fastai.model import *

from lib.transforms3D import Transform3D, CoordTransform3D

class Image3DDataset(FilesIndexArrayDataset):

    def get1item(self, idx):
        x ,y = self.get_x(idx) ,self.get_y(idx)
        return self.get(self.transform, x, y)

    def __getitem__(self, idx):
        if isinstance(idx ,slice):
            xs ,ys = zip(*[self.get1item(i) for i in range(*idx.indices(self.n))])
            return np.stack(xs) ,ys
        return self.get1item(idx)

    def get(self, tfm, x, y):
        if tfm is None:
            return (x,y)
        else:
            p_transform = []
            changed_x = []
            changed_y = None

            #Prepare random 3D transformations
            for t in tfm.tfms:
                if isinstance(t, Transform3D) or isinstance(t, CoordTransform3D):
                    t.randomize_state()
                    p_transform.append(1)#random.randint(0, 1))
                else:
                    p_transform.append(1)

            #Looping through all slices of the cube
            for i in range(x.shape[0]):
                img_x = x[i]
                img_y = y

                #Looping through all transformations
                for j,t in enumerate(tfm.tfms):
                    if p_transform[j]:
                        if len(img_x.shape) == 2:
                            img_x = np.expand_dims(img_x, axis=2)
                        img_x, img_y = t(img_x, img_y)

                changed_x.append(img_x)
                changed_y = img_y

            changed_x = np.rollaxis(np.stack(changed_x, axis=0),1,0)
            return (changed_x, changed_y)

class ImageClassifier3DData(ImageClassifierData):

    @classmethod
    def from_names_and_array(cls, path, fnames, y, classes, val_idxs=None, test_name=None,
            num_workers=8, suffix='', tfms=(None,None), bs=64, continuous=False):
        val_idxs = get_cv_idxs(len(fnames)) if val_idxs is None else val_idxs
        ((val_fnames,trn_fnames),(val_y,trn_y)) = split_by_idx(val_idxs, np.array(fnames), y)

        test_fnames = read_dir(path, test_name) if test_name else None
        if continuous: f = FilesIndexArrayRegressionDataset
        else:
            f = Image3DDataset if len(trn_y.shape)==1 else FilesNhotArrayDataset
        datasets = cls.get_ds(f, (trn_fnames,trn_y), (val_fnames,val_y), tfms,
                               path=path, test=test_fnames)
        return cls(path, datasets, bs, num_workers, classes=classes)
