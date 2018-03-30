# ReBNet
This repository implements the residual binarization scheme in ReBNet: Residual Binarized Neural Network. Please use the following reference for citation:

https://arxiv.org/abs/1711.01243

## Training ReBNet
### Prerequisites
For training RebNet, you should have the following packages installed:
  * Keras
  * Tensorflow
  * mxnet (only if you want to train Imagenet)
### Training Scripts
Use Tensorflow backend for training RenNet for the following datasets:
  #### MNIST:
   * Open "training-software/MNIST-CIFAR-SVHN/Binary.py". 
   * On top of the file, set:
     ```
     dataset="MNIST"
     Train=True
     Evaluate=False
     ``` 
   * Save the script and run:
     ```
     python Binary.py
     ```
  #### CIFAR-10:
   * Open "training-software/MNIST-CIFAR-SVHN/Binary.py". 
   * On top of the file, set:
   * On top of the file, set:
     ```
     dataset="CIFAR-10"
     Train=True
     Evaluate=False
     ``` 
   * Save the script and run: 
     ```
     python Binary.py
     ```
  #### SVHN:
   
   * Download the SVHN dataset from the these three links: [train.mat](http://ufldl.stanford.edu/housenumbers/train_32x32.mat), [test.mat](http://ufldl.stanford.edu/housenumbers/test_32x32.mat), [extra.mat](http://ufldl.stanford.edu/housenumbers/extra_32x32.mat)
   * Place the three downloaded files in "training-software/MNIST-CIFAR-SVHN/svhn_data"
   
   * Open "training-software/MNIST-CIFAR-SVHN/Binary.py". 
   * On top of the file, set:
     ```
     dataset="SVHN"
     Train=True
     Evaluate=False
     ``` 
     
   * Save the script and run:
     ```
     python Binary.py
     ```

 #### Imagenet:
  For speedup of Imagenet training, you need to uninstall your Keras and install an older version of it.
  If you do not wish to do that, you need to write the training script yourself since our code works with the older version of Keras.

  * The script that trains Imagenet is located at "training-software/Imagenet/Binary.py". Before running this script, you need to install the MXNET backend for Keras. MXNET allows you to use multiple GPUs when training with Keras and speeds up the training. Please be advised that, since MXNET is not officially supported by the latest version of Keras, you need to install an older version of Keras. Follow this link to install the old Keras library with MXNET backend support: https://devblogs.nvidia.com/scaling-keras-training-multiple-gpus

   

  * Once you install the old version of Keras with MXNET support, you need to download the Imagenet dataset and create two record (.rec) files containing training and validation images. The .rec file is a single file that contains all of the images of Imagenet. Using .rec files helps you speed-up the training. Instruction on how to create and use .rec files can also be found in the above link.
  
  * you need to open "Binary.py" and specify the directory where the Imagenet data is. Change the following two lines and put the correct address to the .rec files you created:
    ```
    args.data_train='/home/hamid/imagenet/train.rec'
    args.data_val='/home/hamid/imagenet/val.rec'
    ``` 

  * to use the MXNET backend, change the .json file (~/.keras/keras.json): 
  ```
  "backend": "tensorflow" -> "backend": "mxnet"
  ```

  * Run the script located at "training-software/Imagenet/Binary.py" to train Imagenet. The following command trains using 4 gpus, each gpu with a batch size of 64  (total batch size = 256):
   ```
   python Binary.py --batch-per-gpu 64 --num-gpus 4
   ```

## Accuracy Evaluation of ReBNet:
  * Open the corresponding "Binary.py" script, 
  * On top of the file, set:
  ```
  Train=False
  Evaluate=True
  ```
  * Run 
  ```
  python Binary.py
  ```
  We are providing the pretrained weights in "models/DATASET/x_residuals.h5" with x being the number of levels in residual binarization. These weights will be replaced by your trained weights in case you train the models from scratch.



