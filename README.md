# DeepPlant

## Brief Introduction

Nanopore sequencing enables comprehensive detection of 5-methylcytosine (5mC), particularly in transposable elements and centromeric regions. However, CHH methylation detection in plants is limited by the scarcity of high-methylation positive samples, reducing generalization across species. Dorado, the only tool for plant 5mC detection on the R10.4 platform, lacks extensive species testing. To address this, we reanalyzed bisulfite sequencing (BS-seq) data to screen species with abundant high-methylation CHH sites, generating new datasets that cover diverse 9-mer motifs. We developed DeepPlant, a deep learning model incorporating both Bi-LSTM and Transformer architectures, which significantly improves CHH detection accuracy and performs well for CpG and CHG motifs. Evaluated across species, DeepPlant achieved high whole-genome methylation frequency correlations (0.705 to 0.881) with BS-seq data on CHH motifs, improved by 14.0% to 117.6% compared to Dorado. DeepPlant also demonstrated superior single-molecule accuracy, F1-score, and stability, offering strong generalization for plant epigenetics research.

DeepPlant is a specialized training and inference framework for Oxford Nanopore sequencing data. The model leverages Bi-LSTM architecture and is implemented in Python for training. For feature extraction and modification calling, it utilizes C++ integrated with libtorch.

## Using DeepPlant in Docker

To minimize potential issues during the building process, we provide a Docker image for ease of use. If Docker is not installed, it can be downloaded and installed from https://docs.docker.com/engine/install/. Additionally, the **NVIDIA Container Toolkit** is required to utilize GPU resources. For installation instructions, please refer to https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html.

```bash
docker pull chenhx26/deepplant:1.1.0
```

Please run the following command to test. If the GPU status is displayed, the installation is successful; otherwise, please check the installation of the NVIDIA Container Toolkit.
```bash
docker run --rm --gpus all deepplant:1.1.0 nvidia-smi
```

Run the following command to start a container. The `data_path` is the directory containing the data to be processed. It is recommended to map it to the same path inside the container.
```bash
docker run -i -t -v data_path:data_path:rw --gpus all deepplant:1.1.0 /bin/bash
```

You can directly run `DeepPlant extract_and_call_mods -h` to view the input parameters. The model directory is located at `/DeepPlant/model/bilstm`.


## Building from Scratch

You can also build DeepPlant in your own environment by following the instructions below.

### Preparing the Python Environment

Create a virtual environment using Conda. The Python scripts require numpy (version 1.26 is recommended) and pytorch (version 2.2.1 is recommended) with CUDA 11.8 support.

```bash
conda create -n DeepPlant python=3.12
conda activate DeepPlant
pip install numpy==1.26 torch==2.2.1
```

### Building the C++ Program

DeepPlant was tested and runed in **NVIDIA L40s**,  ensure you have a **GPU**, and **CUDA Toolkit 11.8** is recommended. This C++ program is compiled using **g++ 11.4.0** on **Ubuntu 22.04**.

**Install the following packages before building the program:**

1. boost

2. [spdlog](https://github.com/gabime/spdlog "spdlog")

3. zlib

4. ```bash
   apt install zlib1g-dev liblzma-dev libbz2-dev libcurl4-openssl-dev
   ```

**And the these projects are already included in `3rdparty/`**

1. [argparse](https://github.com/p-ranav/argparse "argparse"): Argument Parser for Modern C++

2. [pod5](https://github.com/nanoporetech/pod5-file-format "pod5"): C++ abi for nanopore pod5-file-format

3. [cnpy](https://github.com/rogersce/cnpy "cnpy"): library to read/write .npy and .npz files in C/C++

4. [ThreadPool](https://github.com/progschj/ThreadPool "ThreadPool"): A simple C++11 Thread Pool implementation (slightly modified from the original version in github)

5. [htslib](https://github.com/samtools/htslib "htslib"): An implementation of a unified C library for accessing common file formats

```bash
cd 3rdparty/htslib
tar -xjf htslib-1.21.tar.bz2
cd htslib-1.21
make
ln -s libhts.so ../libhts.so
ln -s libhts.so ../libhts.so.3
cd ../../..
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make -j
```

## DeepPlant Usage

After successfully building the program, you can use our pre-trained model or train your own. The executable is located at `build/DeepPlant`.

### DeepPlant: Extract and Call Modifications

The process for calling modifications.

```bash
Usage: extract_and_call_mods [--help] [--version] pod5_dir bam_path reference_path ref_type write_dir model_dir cpg_kmer_size chg_kmer_size chh_kmer_size num_workers sub_thread_per_worker batch_size

Asynchronously extract features and pass data to the model for modification results.

Positional arguments:
  pod5_dir               Path to the pod5 directory 
  bam_path               Path to the BAM file (sorted by file name required) 
  reference_path         Path to the reference genome 
  ref_type               Reference genome type (default: "DNA")
  write_dir              Path for the detailed modification results files 
  model_dir              Path to the trained models 
  cpg_kmer_size          K-mer size for cpg feature extraction (default: 51)
  chg_kmer_size          K-mer size for chg feature extraction (default: 51)
  chh_kmer_size          K-mer size for chh feature extraction (default: 13)
  num_workers            Number of workers in the feature extraction thread pool, each handling one pod5 file and its corresponding SAM reads (default: 3)
  sub_thread_per_worker  Number of subthreads per worker (default: 3)
  batch_size             Default batch size (default: 1024)
```

The `call_mods` process outputs a `tsv` file containing the following data:

1. read_id
2. reference_start: Start position of the read on the reference genome
3. reference_end: End position of the read on the reference genome
4. chromosome: Reference name of the read on the reference genome
5. pos_in_reference: Position of the current cytosine sites on the reference genome
6. strand: Aligned strand of the read on the reference (+/-)
7. methylation_rate: Methylation rate of the current cytosine sites as determined by the model.


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
  kmer_size              K-mer size for feature extraction (default: 13)
  num_workers            Number of workers in feature extraction thread pool, each handling one pod5 file and its corresponding SAM reads (default: 5)
  sub_thread_per_worker  Number of subthreads per worker (default: 4)
  motif_type             Motif type (default: "CHH")
  loc_in_motif           Location in motif set (default: 0)
```

The extracted features are saved as `npz` files containing site information and data. Site info is stored as a tab-delimited string in a uint8 array, and the data array is used for training.

The `extract_hc_sites` mode allows training of customized models on your data. After extraction, run the script `py/train.py` to train your model.
