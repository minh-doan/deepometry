# Update August 05,2017:
Please visit: http://github.com/broadinstitute/deepometry for the ResNet50 version of deepometry. This example page is an implementation of VGG-like neural network.

# deepometry
A complete open-source workflow to allow a non-expert to use deep learning algorithms to analyze bioimaging data of single cells. The workflow was primarily built to analyze .CIF output of an Imaging Flow Cytometer, it was later generalized to analyze any imagery data of single cells, i.e. accepting .TIF images of identified objects.

The workflow includes every major stage of bioimage analysis: 
- reformating the image data to an appropriate input data structure
- feature extraction and classification by deep learning
- prediction on new images 
- unsupervised clustering using t-SNE / PCA

# Dependencies:
Prior to installation of deepometry itself, user needs the following packages pre-installed:

- python 3.6.1
- numpy 1.12.1 
- scipy 0.19.0
- click 6.7
- pandas 0.19.2
- jupyter 1.0.0
- h5py 2.7.0
- matplotlib 2.0.0
- seaborn 0.7.1
- scikit-image 0.13.0
- scikit-learn 0.18.1
- imageio 2.1.2
- tensorflow 1.0.1 (CPU or GPU version)
- keras 2.0.2
- Java development kit
- python-bioformats 1.1.0
- javabridge 1.0.14

Note: Java development kit (32- or 64- bit version to be matched with operating system) should be installed before python-bioformats and javabridge.

Note: Tensorflow python package is sufficient for CPU use. However, in order to utilize a CUDA-compatible GPU, Tensorflow-GPU as well as CUDA and cuDNN packages are required; more details are described on Tensorflow homepage.

Note: Windows user will need Microsoft Visual C++ Build tools and its compilers installed, with respect to Python versions of 2.7 or 3.5+ accordingly. Windows user is also advised to install numpy (numpy+mkl version), scipy and scikit-image as wheel packages: numpy-1.12.1+mkl-cp35-cp35m-win_amd64.whl, scipy-0.19.0-cp35-cp35m-win_amd64.whl, scikit_image-0.13.0-cp35-cp35m-win_amd64.whl

# Installation of deepometry:
$ git clone https://github.com/minh-doan/deepometry.git

$ cd deepometry

# Usage:
Analytic steps are described in iPython notebooks:
- https://github.com/minh-doan/deepometry/blob/master/STEP_2_Digest_data.ipynb
- https://github.com/minh-doan/deepometry/blob/master/STEP_3_Train_built-in_CNN.ipynb
- https://github.com/minh-doan/deepometry/blob/master/STEP_4_Test_and_Visualization_built-in_CNN.ipynb
