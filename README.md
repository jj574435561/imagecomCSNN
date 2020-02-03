# imagecomCSNN
For more efficient lossless compression of hyperspectral
image, we propose an adaptive prediction algorithm
based on concatenated shallow neural networks (CSNN). The
neural networks are capable of extracting both spatial and
spectral correlations for accurate pixel value prediction. Unlike
most of neural network based methods reported in literature,
the proposed neural network serves as an adaptive filter and
thus does not need to store decompressed data to pre-train
the networks.

load_hyperdata.py:

Pre-processing the hyperspectral data, selecting the context and forming the input data.

pytorch_concate.py:

Employing concatenated Shallow Neural Network for hyperspectral image predicitve filtering. 
