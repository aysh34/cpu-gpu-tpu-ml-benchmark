## Overview

A comprehensive benchmarking suite for comparing CPU, GPU, and TPU performance on machine learning workloads.

## Features

- Multi-device benchmarking (CPU, GPU, TPU)
- Support for popular ML frameworks
- Detailed performance metrics and comparisons
- Easy-to-use CLI interface

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow or PyTorch
- CUDA toolkit (for GPU support)

### Installation

```bash
git clone https://github.com/aysh34/cpu-gpu-tpu-ml-benchmark.git
cd cpu-gpu-tpu-ml-benchmark
pip install -r requirements.txt
```

### Usage

```bash
python benchmark.py --device cpu,gpu,tpu --framework tensorflow
```

<!-- ## Results

Results are saved in the `results/` directory with detailed timing and throughput metrics. -->

## License

MIT
