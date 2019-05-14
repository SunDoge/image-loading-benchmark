# Image Loading Benchmark

This benchmark is for ImageNet.

Baseline:

- PIL

Plan to bench:

- OpenCV
- DALI(CPU/GPU)
- accimage

## Install

### OpenCV

Install 4.1.0 from `conda-forge` instead of 3.4.2 from `base`.

```bash
conda install opencv -c conda-forge
```

### DALI

Install CUDA 10.0 based build.

```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
```


### accimage

```bash
conda install -c conda-forge accimage
```