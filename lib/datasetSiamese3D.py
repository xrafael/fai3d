from fastai.conv_learner import *
from fastai.model import *

from .transforms3D import Transform3D, CoordTransform3D

class DualImage3DDataset(FilesArrayDataset):

    def get1item(self, idx):
        x ,y = self.get_x(idx) ,self.get_y(idx)
        return self.get(self.transform, x, y), idx

    def __getitem__(self, idx):
        if isinstance(idx ,slice):
            xs ,ys = zip(*[self.get1item(i) for i in range(*idx.indices(self.n))])
            return np.stack(xs) ,ys
        return self.get1item(idx)

    def get(self, tfm, xs, y):
        if tfm is None:
            return (xs,y)
        else:

            # Prepare random 3D transformations
            p_transform = []
            for t in tfm.tfms:
                if isinstance(t, Transform3D) or isinstance(t, CoordTransform3D):
                    t.randomize_state()
                    p_transform.append(random.randint(0, 1))
                else:
                    p_transform.append(1)

            # Randomly select the plane when input is a cube
            if xs[0].shape[0] == xs[0].shape[1] == xs[0].shape[2]:
                axis = np.random.choice(3, 1)
            else:
                axis = 0

            #Apply transforms
            all_changed_x = []
            for x in xs:
                changed_x = []
                changed_y = None

                #Looping through all slices of a plane of the cube
                for i in range(x.shape[0]):
                    img_x = x[i] if axis == 0 else x[:,i,:] if axis == 1 else x[:,:,i]
                    img_y = y

                    #Looping through all transformations
                    for j,t in enumerate(tfm.tfms):
                        if p_transform[j]:
                            if len(img_x.shape) == 2:
                                img_x = np.expand_dims(img_x, axis=2)
                            img_x, img_y = t(img_x, img_y)

                    changed_x.append(img_x)
                    changed_y = img_y

                changed_x = np.stack(changed_x, axis=axis+1)
                all_changed_x.append(changed_x)

            all_changed_x = np.array(all_changed_x)
            return (all_changed_x, changed_y)


class FilesIndexArrayDataset3D(DualImage3DDataset):
    def get_c(self): return int(self.y.max())+1


class FilesNhotArrayDataset3D(DualImage3DDataset):
    @property
    def is_multi(self): return True


class FilesIndexArrayRegressionDataset3D(DualImage3DDataset):
    def is_reg(self): return True


class ImageClassifier3DData(ImageClassifierData):

    @classmethod
    def from_names_and_array(cls, path, fnames, y, classes, val_idxs=None, test_name=None,
            num_workers=8, suffix='', tfms=(None,None), bs=64, continuous=False):
        val_idxs = get_cv_idxs(len(fnames)) if val_idxs is None else val_idxs
        ((val_fnames,trn_fnames),(val_y,trn_y)) = split_by_idx(val_idxs, np.array(fnames), y)

        test_fnames = read_dir(path, test_name) if test_name else None
        if continuous: f = FilesIndexArrayRegressionDataset3D
        else:
            f = FilesIndexArrayDataset3D if len(trn_y.shape)==1 else FilesNhotArrayDataset3D
        datasets = cls.get_ds(f, (trn_fnames,trn_y), (val_fnames,val_y), tfms,
                               path=path, test=test_fnames)
        return cls(path, datasets, bs, num_workers, classes=classes)
