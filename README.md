# fai3d
A simple wrapper to 3D image random augmentation for the fast.ai (v.0.7) library.

In the lib folder you will find dataset3D.py and transforms3D.py
that are the main responsible to make the trick.

Both files contain classes that extend from the same ones implemented in the original library
to perform most of the basic transformations but for each slice of
a volume.

To know how to use it, you can find the "main.py" file which implements a
dummy but complete example of application of use. In this file the fast.ai
library is used to build, find the learning rate, fit and visualize the results
of training a basic 3D CNN classifier.

To run the example, first check that you have fast.ai (version 0.7) installed.
Then create a data folder and unzip the data.zip files in it. This zip file contains
few 3D volumes stored in numpy arrays (32x32x32) and a csv file
with the annotations for each volume (id, class). Finally just call main.py file.

If you find this code useful for your work, please don't forget to cite us:





