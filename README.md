# fai3d

A simple wrapper to 3D image random augmentation for fast.ai (v.0.7).

In the lib folder you will find dataset3D.py and transforms3D.py
that are the main responsible to make the trick.

Both files contain classes that extend from the same ones implemented in the original library
to perform most of the basic transformations (e.g. rotate, zoom, lighting, blur, crop and Flip)
but for each slice of a random plane of the volume.

To know more about how to use it in your code, you can find the "main.py" file which implements an
example of a 3D classification problem. In this file the fast.ai library is used to build, find
the learning rate, fit and visualize the results of training a basic 3D CNN classifier.

To run the example, first check that you have fast.ai (version 0.7) installed.
Then unzip data.zip in the data folder. This zip file contains a couple of folders
with some few examples of 3D image volumes (32x32x32) stored in numpy arrays and a csv file
with annotations for each volume (id, class). Finally just call main.py file.

If you find this code useful for your research/work, please don't forget to cite us:

[![DOI](https://zenodo.org/badge/168328059.svg)](https://zenodo.org/badge/latestdoi/168328059)




