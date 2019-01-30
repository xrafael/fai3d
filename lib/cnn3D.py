
from fastai.model import *

# 3D-CNN module
class C3D(nn.Module):
    def __init__(self, ic=3, oc=64, ks=3, pd=1):
        super().__init__()
        self.conv = nn.Conv3d(ic, oc, kernel_size=ks, padding=pd)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.act(x)
