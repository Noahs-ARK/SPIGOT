# SPIGOT

C++ implementation for [Backpropagating through Structured Argmax using a SPIGOT](https://homes.cs.washington.edu/~hapeng/paper/peng2018backprop.pdf).
More detailed documentation coming soon.

## Required software

The following software needs to be installed:

 * A C++ compiler supporting the [C++11 language standard] (https://en.wikipedia.org/wiki/C%2B%2B11)
 * [CMake](http://www.cmake.org/) (tested with version 3.6.2)
 * [Boost](http://www.boost.org/) libraries (tested with version 1.61.0)
 * [Git LFS](https://git-lfs.github.com/) (optional)

Other dependencies including [Eigen](http://eigen.tuxfamily.org), [DyNet](https://github.com/clab/dynet), [gflags](https://github.com/gflags/gflags), [glog](https://github.com/google/glog), and [AD3](https://github.com/andre-martins/AD3), and included in this repo to keep it self-contained.

## Checking out the project for the first time

	git clone https://github.com/Noahs-ARK/SPIGOT
	cd SPIGOT
	./install_deps.sh

## To compile the parser
	
	mkdir -p SyntacticSemanticParsing/build
	cd SyntacticSemanticParsing/build
	cmake ..; make -j4
	cd ../..

## To fetch the GloVe embeddings and pretrained pruners:
	
	git lfs fetch
	git lfs checkout

Or you can simply download the files and put them to the corresponding places.

## Data
The data format follows that by [NeurboParser](https://github.com/Noahs-ARK/NeurboParser). Several training samples are included under 'data/'.

## Training/running 
You can use the scripts in './SyntacticSemanticParsing' to train/evaluate the parser.
