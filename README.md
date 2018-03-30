# ReBNet: Residual Binarized Neural Network


## Citation
If you find ReBNet useful, please cite the <a href="https://arxiv.org/abs/1711.01243" target="_blank">ReBNet paper</a>:

    @inproceedings{finn,
    author = {Mohammad Ghasemzadeh, Mohammad Samragh, Farinaz Koushanfar},
    title = {FReBNet: Residual Binarized Neural Network},
    booktitle = {Proceedings of the 26th IEEE International Symposium on Field-Programmable Custom Computing Machines},
    series = {FCCM '18},
    year = {2018}
    }

 
## Repo organization 

The repo is organized as follows:

-	bnn: contains the BNN class description
	-	src: contains the sources of the 3 networks, and the libraries to rebuild them:
		- library: ReBNet library for HLS BNN descriptions, (please refer to the paper for more details)
		- network: BNN topologies (for MNIST, CIFAR10, and IMAGENET) HLS top functions, and make script for HW built

## Hardware design rebuilt

In order to rebuild the hardware designs, the repo should be cloned in a machine with installation of the Vivado Design Suite (tested with 2017.1). 
Following the step-by-step instructions:

1.	Clone the repository on your linux machine: git clone https://github.com/Xilinx/BNN-PYNQ.git;
2.	Move to `clone_path/ReBNet/bnn/src/network/`
3.	Set the XILINX_BNN_ROOT environment variable to `clone_path/ReBNet/bnn/src/`
4.	Launch the shell script make-hw.sh with parameters the target network, target platform and mode, with the command `./make-hw.sh {network} {platform} {mode}` where:
	- network can be MNIST or CIFAR10 or IMAGENET;
	- platform is pynq;
	- mode can be `h` to launch Vivado HLS synthesis, `b` to launch the Vivado project (needs HLS synthesis results), `a` to launch both.
5.	The results will be visible in `clone_path/ReBNet/bnn/src/network/output/` that is organized as follows:
	- bitstream: contains the generated bitstream(s);
	- hls-syn: contains the Vivado HLS generated RTL and IP (in the subfolder named as the target network);
	- report: contains the Vivado and Vivado HLS reports;
	- vivado: contains the Vivado project.

