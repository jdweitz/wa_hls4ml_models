# wa-hls4ml Surrogate Models

This repository contains surrogate models and utilities for the wa-hls4ml dataset.

## **Repository Structure**

### **Directories**
- `transformer/`: Contains transformer based model and utilities.
  - `data.py`: Data preprocessing scripts.
  - `model.py`: Transformer model definitions.
  - `train.py`: Training scripts for the transformer model.
  - `requirements.txt`: Dependencies for transformer workflow.

- `GNN/`: Contains Graph Neural Network (GNN) model and utilities.
  - `DatasetMay29Complete.py`: Dataset preparation for GNN model.
  - `Models.py`: GNN model definitions.
  - `training_scripts/`: Scripts for training the GNN model.
  - `requirements.txt`: Dependencies for GNN workflows.

- `dataset/`: Contains dataset preparation scripts.
  - `Dataset_to_csvs6_with_ii.py`: Converts datasets to CSV format.

### **Files**
- `5_26_requirements.txt`: Dependencies for the project.
- `.gitignore`: Specifies files and directories to ignore.
