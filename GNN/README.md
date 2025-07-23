# GNN for FPGA Resource Prediction

This repository contains the code for a Graph Neural Network (GNN) based surrogate model for predicting FPGA resource utilization and latency for High-Level Synthesis for Machine Learning (hls4ml).

## Introduction

As Machine Learning (ML) models become more prevalent in hardware applications, the need for rapid design iteration and optimization is crucial. High-Level Synthesis (HLS) tools like hls4ml have significantly accelerated the process of deploying ML models on FPGAs. However, predicting the resource usage (LUTs, FFs, DSPs, BRAM) and latency of a given ML model on a specific FPGA architecture remains a challenging and time-consuming task.

This project introduces a GNN-based surrogate model that can accurately and efficiently predict FPGA resource utilization and latency directly from the ML model's architecture description. By representing the ML model as a graph, the GNN can learn the complex relationships between the model's structure and its hardware implementation, providing fast and accurate resource estimates without the need for lengthy HLS synthesis runs.

## GNN Model Architecture

The GNN model is based on the Graph Attention Network (GATv2) architecture. The model takes a graph representation of the ML model as input, where each node in the graph represents a layer in the ML model. The node features include information about the layer's type, dimensions, precision, and other synthesis parameters.

The GNN model consists of five stacked GATv2 layers with five attention heads each. The GATv2 layers allow the model to learn the importance of different layer connections and propagate information through the graph. The model also includes residual connections, layer normalization, and a multi-strategy pooling mechanism to improve training stability and performance.

The final graph-level embedding is concatenated with global features (e.g., synthesis strategy, I/O type) and passed through a Multi-Layer Perceptron (MLP) to predict the six target values: LUT, FF, DSP, BRAM, latency, and initiation interval (II).

## How to Use the Code

The code in this repository is organized as follows:

- `Models.py`: Contains the PyTorch implementation of the GNN models (FPGA_GNN, FPGA_GNN_GATv2, FPGA_GNN_GATv2_Enhanced).
- `training_scripts/`: Contains the scripts for training the GNN models.
- `utils/`: Contains utility functions for data loading, processing, and plotting.
- `Dataset_to_csvs6_with_ii.py`: Script to process the dataset.
- `DatasetGNN.py`: Script to process the dataset.

To train a GNN model, you can run the corresponding script in the `training_scripts/` directory. For example, to train the baseline GNN model, you can run:

```bash
python training_scripts/y_03_baseline.py
```

## Dataset

The dataset used for training and evaluating the GNN model is the `wa-hls4ml` dataset, which contains over 680,000 synthesized dataflow models. The dataset includes a wide variety of ML models, including fully connected and convolutional neural networks, with different architectures, precisions, and synthesis parameters.

The dataset is available for download at: [https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml](https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml)

## Results

The GNN-based surrogate model achieves high accuracy in predicting FPGA resource utilization and latency. The model outperforms the baseline MLP model and shows competitive performance compared to the transformer-based model. The GNN model is particularly effective in predicting DSP and Cycles, with R2 scores of 0.89 and 0.89, respectively, on the test set.

The results demonstrate that the GNN-based approach is a promising solution for fast and accurate FPGA resource prediction, enabling rapid design space exploration and optimization of ML models for FPGAs.
