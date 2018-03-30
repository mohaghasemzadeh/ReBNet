# ReBNet: Residual Binarized Neural Network


## Citation
If you find ReBNet useful, please cite the <a href="https://arxiv.org/abs/1711.01243" target="_blank">ReBNet paper</a>:

    @inproceedings{finn,
    author = {Mohammad Ghasemzadeh, Mohammad Samragh, Farinaz Koushanfar},
    title = {ReBNet: Residual Binarized Neural Network},
    booktitle = {Proceedings of the 26th IEEE International Symposium on Field-Programmable Custom Computing Machines},
    series = {FCCM '18},
    year = {2018}
    }

 
## Repo organization 
The repo is organized as follows:

- training-software: contains python codes for training ReBNet.
	- MNIST-CIFAR-SVHN: Contains training files and pretrained parameters for the three applications.
	- Imagenet: Contains training files for Imagenet.
	
- bnn: contains the FPGA hardware implementation.
	- src: contains the sources of the 3 networks, and the libraries to rebuild them:
		- library: ReBNet library for HLS BNN descriptions, (please refer to the paper for more details)
		- network: BNN topologies (for MNIST, CIFAR10, and IMAGENET) HLS top functions, and make script for HW built

## Training ReBNet
- ### Prerequisites
	For training RebNet, you should have the following packages installed:
  	- Keras
  	- Tensorflow
  	- mxnet (only if you want to train Imagenet)
- ### Training Scripts
	Use Tensorflow backend for MNIST, CIFAR-10, and SVHN. Use MXNET backend for Imagenet.
  	- #### MNIST:
   		- Open "training-software/MNIST-CIFAR-SVHN/Binary.py". 
   		- On top of the file, set:
		```
		dataset="MNIST"
		Train=True
		Evaluate=False
		``` 
   		- Save the script and run:
		```
		python Binary.py
		```
  	- #### CIFAR-10:
		- Open "training-software/MNIST-CIFAR-SVHN/Binary.py". 
		- On top of the file, set:
		```
		dataset="CIFAR-10"
		Train=True
		Evaluate=False
		``` 
		- Save the script and run: 
		```
		python Binary.py
		```
  	- #### SVHN:
		- Download the SVHN dataset from the these three links: [train.mat](http://ufldl.stanford.edu/housenumbers/train_32x32.mat), [test.mat](http://ufldl.stanford.edu/housenumbers/test_32x32.mat), [extra.mat](http://ufldl.stanford.edu/housenumbers/extra_32x32.mat)
		- Place the three downloaded files in "training-software/MNIST-CIFAR-SVHN/svhn_data"
		- Open "training-software/MNIST-CIFAR-SVHN/Binary.py". 
		- On top of the file, set:
		```
		dataset="SVHN"
		Train=True
		Evaluate=False
		``` 
		- Save the script and run:
		```
		python Binary.py
		```
 	- #### Imagenet:
	  For speedup of Imagenet training, you need to uninstall your Keras and install an older version of it.
	  If you do not wish to do that, you need to write the training script yourself since our code works with the older version of Keras.
		- The script that trains Imagenet is located at "training-software/Imagenet/Binary.py". Before running this script, you need to install the MXNET backend for Keras. MXNET allows you to use multiple GPUs when training with Keras and speeds up the training. Please be advised that, since MXNET is not officially supported by the latest version of Keras, you need to install an older version of Keras. Follow this link to install the old Keras library with MXNET backend support: https://devblogs.nvidia.com/scaling-keras-training-multiple-gpus
		- Once you install the old version of Keras with MXNET support, you need to download the Imagenet dataset and create two record (.rec) files containing training and validation images. The .rec file is a single file that contains all of the images of Imagenet. Using .rec files helps you speed-up the training. Instruction on how to create and use .rec files can also be found in the above link. 
		- You need to open "Binary.py" and specify the directory where the Imagenet data is. Change the following two lines and put the correct address to the .rec files you created:
		```
		args.data_train='/home/hamid/imagenet/train.rec'
		args.data_val='/home/hamid/imagenet/val.rec'
		``` 
		- To use the MXNET backend, change the .json file (~/.keras/keras.json): 
		```
		"backend": "tensorflow" -> "backend": "mxnet"
		```
		- Run the script located at "training-software/Imagenet/Binary.py" to train Imagenet. The following command trains using 4 gpus, each gpu with a batch size of 64  (total batch size = 256):
		```
		python Binary.py --batch-per-gpu 64 --num-gpus 4
		```
## Accuracy Evaluation of ReBNet
  - Open the corresponding "Binary.py" script, 
  - On top of the file, set:
  ```
  Train=False
  Evaluate=True
  ```
  - Run 
  ```
  python Binary.py
  ```
  We are providing the pretrained weights in "models/DATASET/x_residuals.h5" with x being the number of levels in residual binarization. These weights will be replaced by your trained weights in case you train the models from scratch.
  
## Hardware design rebuilt
In order to rebuild the hardware designs, the repo should be cloned in a machine with installation of the Vivado Design Suite (tested with 2017.1). Following the step-by-step instructions:

- Clone the repository on your linux machine: git clone https://github.com/mohaghasemzadeh/ReBNet.git;
- Move to `clone_path/ReBNet/bnn/src/network/`
- Set the XILINX_BNN_ROOT environment variable to `clone_path/ReBNet/bnn/src/`
- Launch the shell script make-hw.sh with parameters the target network, target platform and mode, with the command `./make-hw.sh {network} {platform} {mode}` where:
	- network can be MNIST or CIFAR10 or IMAGENET;
	- platform is pynq;
	- mode can be `h` to launch Vivado HLS synthesis, `b` to launch the Vivado project (needs HLS synthesis results), `a` to launch both.
- The results will be visible in `clone_path/ReBNet/bnn/src/network/output/` that is organized as follows:
	- bitstream: contains the generated bitstream(s);
	- hls-syn: contains the Vivado HLS generated RTL and IP (in the subfolder named as the target network);
	- report: contains the Vivado and Vivado HLS reports;
	- vivado: contains the Vivado project.

