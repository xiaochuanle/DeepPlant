# DeepPlant

## Brief Introduction

Recent nanopore sequencing system (R10.4) has enhanced base calling accuracy and is being increasingly utilized for determining the methylation state of genomic CpGs. However, the robustness and universality of its methylation calling model in officially supplied Dorado remains poorly tested. In this study, we obtained heterogeneous datasets from human and plant sources to carry out comprehensive evaluations, which showed that Dorado displays significantly different performances across the datasets. Therefore, we developed deep neural networks and trained a nanopore-based CpG methylation calling model called DeepPlant. DeepPlant achieved superior and more stable areas under the receiver operating characteristic curves (97.80% on average), balanced accuracies (95.96%), and F1 scores (94.97%) across the datasets. DeepPlant-based methylation frequency had >0.95 correlations with BS-seq on four of five datasets, outperforming Dorado in all instances. We also showed that DeepPlant enables uncovering haplotype-specific methylation patterns including partial repetitive regions. The enhanced performance of DeepPlant paves the way for broader applications of nanopore sequencing in CpG methylation studies.

DeepPlant is a specialized training and inference framework for Oxford Nanopore sequencing data. The model leverages Bi-LSTM architecture and is implemented in Python for training. For feature extraction and modification calling, it utilizes C++ integrated with libtorch.

## Building from Scratch

### Preparing the Python Environment

Create a virtual environment using Conda. The Python scripts require numpy (version 20.0 or higher) and pytorch (version 2.0 or higher) with CUDA 11.8 support.

```bash
conda create -n DeepPlant python=3.11
conda activate DeepPlant
pip install numpy torch==2.0.1
```

### Building the C++ Program

DeepPlant was tested and runed in **NVIDIA GeForce RTX 3090**,  ensure you have a **GPU** and **CUDA Toolkit 11.8** installed.  Download **libtorch 2.0.1** if it's not already included in your Python environment. This C++ program is compiled using g++-11.2 on Ubuntu 22.04. Compatibility issues may arise on other systems, so feel free to raise an issue if you encounter any problems.

**If you are not familiar about how to install CUDA Toolkit 11.8, here is a example for set up CUDA Toolkit 11.8 in ubuntu 22.04 x86_64 system**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

**Install the following packages before building the program:**

1. boost

2. [spdlog](https://github.com/gabime/spdlog "spdlog"): a Fast C++ logging library

3. zlib

**And the these projects are already included in `3rdparty/`**

4. [argparse](https://github.com/p-ranav/argparse "argparse"): Argument Parser for Modern C++

5. [pod5](https://github.com/nanoporetech/pod5-file-format "pod5"): C++ abi for nanopore pod5-file-format

6. [cnpy](https://github.com/rogersce/cnpy "cnpy"): library to read/write .npy and .npz files in C/C++

7. [ThreadPool](https://github.com/progschj/ThreadPool "ThreadPool"): A simple C++11 Thread Pool implementation (slightly modified from the original version in github)
```bash
git clone https://github.com/huicongyao/Deep-Bam.git
cd Deep-Bam/cpp
mkdir build && cd build
conda activate DeepPlant # Activate the previously created environment
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` .. # Determine the cmake path # if you haven`t set up the python environment, you should directy include libtorch path here.
make -j
```

## DeepPlant Usage

After successfully building the program, you can use our pre-trained model or train your own. The executable is located at `Deep-Bam/cpp/build/DeepPlant`.

### DeepPlant: Extracting High-Confidence Sites

This process extracts features for model training.

```bash
Usage: extract_hc_sites [--help] [--version] pod5_dir bam_path reference_path ref_type write_dir pos neg kmer_size num_workers sub_thread_per_worker motif_type loc_in_motif

Extract features for model training using high-confidence bisulfite data.

Positional arguments:
  pod5_dir               Path to the pod5 directory 
  bam_path               Path to the BAM file (sorted by file name required) 
  reference_path         Path to the reference genome 
  ref_type               Reference genome type (default: "DNA")
  write_dir              Directory for output files, format: ${pod5filename}.npy 
  pos                    Positive high-accuracy methylation sites 
  neg                    Negative high-accuracy methylation sites 
  kmer_size              K-mer size for feature extraction (default: 51)
  num_workers            Number of workers in feature extraction thread pool, each handling one pod5 file and its corresponding SAM reads (default: 10)
  sub_thread_per_worker  Number of subthreads per worker (default: 4)
  motif_type             Motif type, default: CG (default: "CG")
  loc_in_motif           Location in motif set 
```

The extracted features are saved as `npz` files containing site information and data. Site info is stored as a tab-delimited string in a uint8 array, and the data array is used for training.

The `extract_hc_sites` mode allows training of customized models on your data. After extraction, run the script `py/train_lstm.py` to train your model. Refer to the `README.md` in the py directory for further instructions.

### DeepPlant: Extract and Call Modifications

The process for calling modifications. 

```bash
Usage: extract_and_call_mods [--help] [--version] pod5_dir bam_path reference_path ref_type write_file module_path kmer_size num_workers sub_thread_per_worker batch_size motif_type loc_in_motif

Asynchronously extract features and pass data to the model for modification results.

Positional arguments:
  pod5_dir               Path to the pod5 directory 
  bam_path               Path to the BAM file (sorted by file name required) 
  reference_path         Path to the reference genome 
  ref_type               Reference genome type (default: "DNA")
  write_dir             Path for the detailed modification results file 
  module_dir            Path to the trained model 
  cpg_kmer_size         K-mer size for cpg feature extraction (default: 51)
  chg_kmer_size         K-mer size for chg feature extraction (default: 51)
  chh_kmer_size         K-mer size for chh feature extraction (default: 13)
  num_workers            Number of workers in the feature extraction thread pool, each handling one pod5 file and its corresponding SAM reads (default: 10)
  sub_thread_per_worker  Number of subthreads per worker (default: 4)
  batch_size             Default batch size (default: 1024)
```

The `call_mods` process outputs a `tsv` file containing the following data:

1. read_id
2. reference_start: Start position of the read on the reference genome
3. reference_end: End position of the read on the reference genome
4. chromosome: Reference name of the read on the reference genome
5. pos_in_strand: Position of the current CpG site on the reference genome
6. strand: Aligned strand of the read on the reference (+/-)
7. methylation_rate: Methylation rate of the current CpG sites as determined by the model.

You could find trained  torch script modules in `traced_script_module` file that contains different k-mer.

## Publication

...

