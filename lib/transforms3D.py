from fastai.conv_learner import *
from fastai.model import *


class Transform3D(Transform):

    def randomize_state(self, params): pass


class CoordTransform3D(CoordTransform):

    def randomize_state(self): pass


class Crop(CoordTransform3D):
    """ A class that represents a Random Crop transformation.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments
    ---------
        targ: int
            target size of the crop.
        tfm_y: TfmType
            type of y transformation.
    """

    def __init__(self, targ_sz, r=0.5, c=0.5, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.targ_sz, self.sz_y = targ_sz, sz_y
        self.r, self.c = r, c

    def randomize_state(self):
        self.store.rand_r = random.uniform(0, 1)
        self.store.rand_c = random.uniform(0, 1)

    def do_transform(self, x, is_y):
        r, c, *_ = x.shape
        sz = self.sz_y if is_y else self.targ_sz
        start_r = np.floor(self.store.rand_r * (r - sz)).astype(int)
        start_c = np.floor(self.store.rand_c * (c - sz)).astype(int)
        return crop(x, start_r, start_c, sz)


class Rotate(CoordTransform3D):
    """ Rotates images and (optionally) target y.

    Rotating coordinates is treated differently for x and y on this
    transform.
     Arguments:
        deg (float): degree to rotate.
        mode: type of border
        tfm_y (TfmType): type of y transform
    """

    def __init__(self, deg, mode=cv2.BORDER_REFLECT, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.deg = deg
        if tfm_y == TfmType.COORD or tfm_y == TfmType.CLASS:
            self.modes = (mode, cv2.BORDER_CONSTANT)
        else:
            self.modes = (mode, mode)

    def randomize_state(self):
        self.store.rdeg = rand0(self.deg)

    def do_transform(self, x, is_y):
        x = rotate_cv(x, self.store.rdeg, mode=self.modes[1] if is_y else self.modes[0],
                      interpolation=cv2.INTER_NEAREST if is_y else cv2.INTER_AREA)
        return x


class Flip(CoordTransform3D):
    """
    Flip image left-right or up-down
    """

    def __init__(self, t=0, tfm_y=TfmType.NO):
        super().__init__(tfm_y=tfm_y)
        self.flipType = t

    def do_transform(self, x, is_y): return np.fliplr(x) if self.flipType == 0 else np.flipud(x)


class Lighting(Transform3D):
    """
    Adjust image balance and contrast
    """

    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b, self.c = b, c

    def randomize_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1 / (c - 1) if c < 0 else c + 1
        x = lighting(x, b, c)
        return x


class Zoom(CoordTransform3D):
    """
    Zoom the center of image x by a factor of z+1 while retaining the original image size and proportion.
    """

    def __init__(self, zoom_max, zoom_min=0, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.zoom_max, self.zoom_min = zoom_max, zoom_min

    def randomize_state(self):
        self.store.zoom = self.zoom_min + (self.zoom_max - self.zoom_min) * random.random()

    def do_transform(self, x, is_y):
        return zoom_cv(x, self.store.zoom)


class Stretch(CoordTransform3D):
    """
    Stretches image x horizontally by sr+1, and vertically by sc+1 while
    retaining the original image size and proportion.
    """

    def __init__(self, max_stretch, stretch_dir=0, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.max_stretch = max_stretch
        self.stretch_dir = stretch_dir

    def randomize_state(self):
        self.store.stretch = self.max_stretch * random.random()

    def do_transform(self, x, is_y):
        if self.stretch_dir == 0:
            x = stretch_cv(x, self.store.stretch, 0)
        else:
            x = stretch_cv(x, 0, self.store.stretch)
        return x


class Blur(Transform3D):
    """
    Adds a gaussian blur to the image at chance.
    Multiple blur strengths can be configured, one of them is used by random chance.
    """

    def __init__(self, blur_strengths=5, tfm_y=TfmType.NO):
        # Blur strength must be an odd number, because it is used as a kernel size.
        super().__init__(tfm_y)
        self.blur_strengths = (np.array(blur_strengths, ndmin=1) * 2) - 1
        if np.any(self.blur_strengths < 0):
            raise ValueError("all blur_strengths must be > 0")
        kernel_size = np.random.choice(self.blur_strengths)
        self.store.kernel = (kernel_size, kernel_size)

    def randomize_state(self):
        kernel_size = np.random.choice(self.blur_strengths)
        self.store.kernel = (kernel_size, kernel_size)

    def do_transform(self, x, is_y):
        return cv2.GaussianBlur(src=x, ksize=self.store.kernel, sigmaX=0)
