# play-vgg
Analysis of convolutional neural network

## Plans

1. Analysis of sparse connection layer
1. Analysis of linear convolutional layer

## Sparse Connection

|Name|Dropout prob|Sparsitys|Total params|Classifier params|accurancy|
|:---|:---:|:---:|:---:|:---:|:---:|
|VGG_Me|NaN|2.48e-05, 9.77e-05|9225290|535562|91.66|
|VGG_Me_Sparse|0.5|2.10e-05, 1.62e-05, 1.53e-05, 0.00e+00|10529610|1839882|91.16|
|VGG_Me_Sparse|0.8|1.34e-05, 1.05e-05, 0.00e+00, 0.00e+00|10529610|1839882|90.92|


## Linear Convolution

|Name|Sparsitys|Total params|Classifier params|accurancy|
|:---|:---:|:---:|:---:|:---:|
|VGG_Me|2.48e-05, 9.77e-05|9225290|535562|91.66|
|VGG_Conv_Linear|9.54e-06, 9.77e-05|9225290|535562|85.92|
|VGG_Linear|4.58e-05, 0.00e+00|9225290|535562|87.22|
|Full_Linear|3.02e-05, 1.40e-04|16792586|16792586|35.44|

