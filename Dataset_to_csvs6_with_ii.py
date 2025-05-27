# %%
import json
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import os
import glob
from tqdm import tqdm


# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#set logging off
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INDEPENDENT_LAYER_NAMES = [
    'QConv2D', 'QDense', 'QConv1D', 'MaxPooling2D', 'MaxPooling1D', 
    'AveragePooling2D', 'AveragePooling1D', 'Flatten', 'QDepthwiseConv2D', 
    'QDepthwiseConv1D', 'QSeparableConv2D', 'QSeparableConv1D',
    'Activation', 'QActivation'  # Added activation layers
]



LAYER_TYPE_MAPPING = {
    'activation': 0,
    'QConv2D': 3,
    'QDense': 1,
    'QConv1D': 2,
    'MaxPooling2D': 9,
    'MaxPooling1D': 9,
    'AveragePooling2D': 10,
    'AveragePooling1D': 10,
    'Flatten': 8,
    'QDepthwiseConv2D': 7,
    'QDepthwiseConv1D': 6,
    'QSeparableConv2D': 5,
    'QSeparableConv1D': 4,
    'Activation': 0,
    'QActivation': 0
}

# Strategy mapping: latency=0, resource=1
STRATEGY_MAPPING = {
    'latency': 0,
    'resource': 1
}

ACTIVATION_MAPPING = {
    'NA': 0, # Represents non-activation layers
    'linear': 1,
    'quantized_relu': 2,
    'quantized_tanh': 3,
    'quantized_sigmoid': 4,
    'quantized_softmax': 5
}


FEATURES = {
    "d_in1": 0,
    "d_in2": 1,
    "d_in3": 2,
    "d_out1": 3,
    "d_out2": 4,
    "d_out3": 5,
    "prec": 6,
    "rf": 7,
    "strategy": 8,
    "layer_type": 9,
    "activation_type": 10,
    "filters": 11,
    "kernel_size": 12,
    "stride": 13,
    "padding": 14,
    "pooling": 15

}

PADDING_MAPPING = {
    'NA': 0,
    'valid': 1,
    'same': 2,
}


# %%

class ModelProcessor:
    """Process neural network model JSON configurations and convert to CSV format."""
    
    def __init__(self):
        """Initialize the ModelProcessor with feature column definitions."""
        # self.feature_columns = [
        #     "d_in1", "d_in2", "d_in3", "d_out1", "d_out2", "d_out3", 
        #     "prec", "rf", "strategy", "rf_times_precision", "layer_type", 
        #     "activation_type", "filters", "kernel_size", "stride", 
        #     "padding", "pooling"
        # ]
        self.feature_columns = list(FEATURES.keys())


        self.feature_index = {name: i for i, name in enumerate(self.feature_columns)}
        
        # Define label columns for resource metrics
        # self.label_columns = ["cycles_max", "ff", "lut", "bram", "dsp"]
        self.label_columns = ["cycles_max", "ff", "lut", "bram", "dsp", "interval_max"]
        

    
    def has_valid_resource_report(self, file_path: str) -> bool:
        """Quickly check if a JSON file contains non-empty resource report.
        
        This method only reads the file once and checks for resource report without
        fully processing the JSON, making it much faster for filtering.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            True if file contains non-empty resource report, False otherwise
        """
        try:
            # First, do a quick check using string search without full JSON parsing
            with open(file_path, 'r') as file:
                content = file.read()
                
            # Quick check for hls_resource_report pattern (since that's where the data actually is)
            import re
            pattern = r'"hls_resource_report":\s*{[^{}]*[^}]+}'
            if not re.search(pattern, content):
                return False
            
            # If pattern found, fully parse to verify it's not empty
            import json
            data = json.loads(content)
            
            # Get hls_resource_report which has the actual resource metrics
            hls_resource_report = data.get('hls_resource_report', {})
            
            # Check if it has at least one of the expected fields with non-empty string values
            return any(key in hls_resource_report and hls_resource_report[key] 
                      for key in ["ff", "lut", "bram", "dsp"])
        
            # # Check if it has at least three of the expected fields with non-zero values
            # zero_count = 0
            # for key in ["ff", "lut", "bram", "dsp"]:
            #     if key not in resource_report or resource_report[key] == 0:
            #         zero_count += 1
            # # If at least three fields are non-zero, consider it valid
            # if zero_count >= 3:
            #     return True
        
                    
        except Exception as e:
            logger.error(f"Error checking resource report in {file_path}: {e}")
            return False
    
    def load_json(self, file_path: str) -> Dict:
        """Load a JSON file and return its contents as a dictionary.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the JSON data
        """
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            raise
    
    def get_resource_report(self, data: Dict) -> Dict[str, int]:
        """Extract resource usage information from the data.
        
        Args:
            data: Dictionary containing the model data
            
        Returns:
            Dictionary with resource metrics
        """
        lat = data.get("latency_report", {})
        hls_res = data.get("hls_resource_report", {})

        
        return {
            "cycles_max": int(lat.get("cycles_max", 0)),
            "ff": int(hls_res.get("ff", 0)),
            "lut": int(hls_res.get("lut", 0)),
            "bram": float(hls_res.get("bram", 0)),
            "dsp": int(hls_res.get("dsp", 0)),
            "interval_max": int(lat.get("interval_max", 0))
        }
    
    def get_layers(self, data: Dict) -> Tuple[List[str], List[int]]:
        """Identify independent layers in the model configuration.
        
        Args:
            data: Dictionary containing the model data
            
        Returns:
            Tuple of (layer_names, layer_indices)
        """
        layers = []
        layers_indices = []
        
        for i, layer in enumerate(data.get('model_config', [])):
            class_name = layer.get('class_name', '')
            
            # Check if the layer is an independent layer
            if class_name in INDEPENDENT_LAYER_NAMES:
                layers.append(class_name)
                layers_indices.append(i)
            
            # Add the last layer index
            if i == len(data.get('model_config', [])) - 1:
                layers_indices.append(i)
        
        return layers, layers_indices
    
    def parse_weight_string(self, weight_str: str) -> Optional[str]:
        """Parse a weight string to extract the precision value.
        
        Args:
            weight_str: String representation of weights
            
        Returns:
            Extracted precision value or None if not found
        """
        try:
            start_index = weight_str.find("<") + 1
            end_index = weight_str.find(">")
            
            if start_index > 0 and end_index > start_index:
                content = weight_str[start_index:end_index]
                return content.split(',')[0]
            
            return None
        except (ValueError, AttributeError, IndexError):
            logger.warning(f"Could not parse weight string: {weight_str}")
            return None
        
    def get_layer_info(self, data: Dict, layer_index: int, next_layer_index: int, 
                       is_last_layer: bool = False) -> List[Any]:
        """Extract detailed information about a specific layer.
        
        Args:
            data: Dictionary containing the model data
            layer_index: Index of the current layer
            next_layer_index: Index of the next layer
            is_last_layer: Whether this is the last layer in the model
            
        Returns:
            List of layer features
        """
        # Initialize layer information with None values
        layer_info = [None] * len(self.feature_columns)
        model_info = data['model_config'][layer_index]
        
        # Extract input and output shapes
        input_shape = model_info.get('input_shape', [0])[1:]
        output_shape = model_info.get('output_shape', [0])[1:]
        
        # Set input dimensions
        for i in range(3):
            layer_info[i] = input_shape[i] if i < len(input_shape) else 0
            
        # Set output dimensions
        for i in range(3):
            layer_info[i + 3] = output_shape[i] if i < len(output_shape) else 0
        
        # Process by layer type
        class_name = model_info.get('class_name', '')
        
        # Handle activation layers
        if class_name in ('Activation', 'QActivation'):
            # Set layer type to 0 for activation layers
            layer_info[self.feature_index['layer_type']] = LAYER_TYPE_MAPPING[class_name]
            layer_info[self.feature_index['pooling']] = 0
            
            # Get activation type from layer config
            activation_value = model_info.get('activation', 'linear')
            
            # Find activation type and set it
            activation_type = 0  # Default to 0 (linear)
            for activation_name, activation_code in ACTIVATION_MAPPING.items():
                if activation_name in activation_value:
                    activation_type = activation_code
                    break
            
            layer_info[self.feature_index['activation_type']] = activation_type
            
        # Handle pooling layers
        elif class_name in ('MaxPooling2D', 'MaxPooling1D'):
            layer_info[self.feature_index['pooling']] = 2
            layer_info[self.feature_index['layer_type']] = LAYER_TYPE_MAPPING['MaxPooling1D']
            layer_info[self.feature_index['activation_type']] = 0
        elif class_name in ('AveragePooling2D', 'AveragePooling1D'):
            layer_info[self.feature_index['pooling']] = 2
            layer_info[self.feature_index['layer_type']] = LAYER_TYPE_MAPPING['AveragePooling1D']
            layer_info[self.feature_index['activation_type']] = 0
        else:
            layer_info[self.feature_index['pooling']] = 0
            layer_info[self.feature_index['activation_type']] = 0
        
        # Set layer-specific parameters
        if class_name in LAYER_TYPE_MAPPING and class_name not in ('Activation', 'QActivation'):
            layer_info[self.feature_index['layer_type']] = LAYER_TYPE_MAPPING[class_name]
        
        # Initialize all spatial parameters to 0 by default
        for param in ['filters', 'kernel_size', 'stride', 'padding']:
            layer_info[self.feature_index[param]] = 0
        # Set filters= d_out3 , kernel size = 1, stride = 1, and padding = 1 for convolutional layers
        # if class_name in ('QConv2D'):
        #     layer_info[self.feature_index['filters']] = output_shape[2]
        #     layer_info[self.feature_index['kernel_size']] = 1
        #     layer_info[self.feature_index['stride']] = 1
        #     layer_info[self.feature_index['padding']] = 0

        # elif class_name in ('QConv1D'):
        #     layer_info[self.feature_index['filters']] = output_shape[1]
        #     layer_info[self.feature_index['kernel_size']] = 1
        #     layer_info[self.feature_index['stride']] = 1
        #     layer_info[self.feature_index['padding']] = 0
        if class_name in ('QConv2D', 'QConv1D'):
            layer_info[self.feature_index['filters']] = model_info['filters']
            kernel_size = model_info['kernel_size']
            layer_info[self.feature_index['kernel_size']] = kernel_size[0] #change to below if doesnt work
            # if isinstance(kernel_size, list) and len(kernel_size) > 0:
            #     layer_info[self.feature_index['kernel_size']] = kernel_size[0]
            # else:
            #     layer_info[self.feature_index['kernel_size']] = kernel_size
            strides = model_info['strides']
            layer_info[self.feature_index['stride']] = strides[0]

            # if isinstance(strides, list) and len(strides) > 0:
            #     layer_info[self.feature_index['stride']] = strides[0]
            # else:
            #     layer_info[self.feature_index['stride']] = strides

            padding = model_info['padding']
            layer_info[self.feature_index['padding']] = PADDING_MAPPING.get(padding, 1)
            


        
        return layer_info
    
    

    def process_json_to_csv(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Process a JSON file and convert it to a DataFrame and get resource report.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Tuple of (features_dataframe, resource_report_dict)
        """
        logger.info(f"Processing file: {file_path}")
        
        # Create empty DataFrame
        input_features = pd.DataFrame(columns=self.feature_columns)
        
        # Load data
        try:
            data = self.load_json(file_path)
        except Exception as e:
            logger.error(f"Failed to load JSON data: {e}")
            return input_features, {}
        
        # Get resource report for labels
        resource_report = self.get_resource_report(data)
        
        # Extract layers
        model_layers_names, model_layers_indices = self.get_layers(data)
        logger.info(f"Found {len(model_layers_names)} layers: {model_layers_names}")
        logger.debug(f"Layer indices: {model_layers_indices}")
        
 
        for i, layer_name in enumerate(model_layers_names):
           
        # Check if we should keep this activation layer
            # if not self.should_keep_activation(data, model_layers_names, i):
            #     # print(f"Skipping activation layer {layer_name}")
            #     continue
            
            # Get layer information
            is_last_layer = (i == len(model_layers_names) - 1)
            layer_info = self.get_layer_info(
                data, 
                model_layers_indices[i], 
                model_layers_indices[i + 1], 
                is_last_layer
            )
            # Add to DataFrame
            new_df = pd.DataFrame([layer_info], columns=self.feature_columns)
            input_features = pd.concat([input_features, new_df], ignore_index=True)
        # Process HLS configuration
        # self._process_hls_config(data, input_features)
        try:
            self._process_hls_config(data, input_features)
        except Exception as e:
            logger.error(f"Error processing HLS configuration: {e}")
        
        return input_features, resource_report
    
    def _process_hls_config(self, data: Dict, input_features: pd.DataFrame) -> None:
        """Process HLS configuration data and update the features DataFrame.
        
        Args:
            data: Dictionary containing the model data
            input_features: DataFrame to update with HLS configuration
        """
        hls_config = data.get('hls_config', {})
        model_config = hls_config.get('Model', {})
        layer_configs = hls_config.get('LayerName', {})
    
        
        # Filter out non-layer configs
        layer_configs_new = {
            k: v for k, v in layer_configs.items() 
            # if not any(x in k for x in ['linear', 'alpha', 'input'])
            if not any(x in k for x in ['alpha', 'input'])
        }

        input_features_layer_type = input_features['layer_type'].tolist()
        layer_configs_new_list = list(layer_configs_new.keys())
        layers_to_remove = []

        input_features_count = 0
        #identify layers to remove
        for i, layer_name in enumerate(layer_configs_new_list):
            # print(f" input_features: {input_features.iloc[i]['layer_type']}, hls_config: {layer_name}")
                        
            if "linear" in layer_name:
                if i + 1 < len(layer_configs_new_list):
                    next_layer_name = layer_configs_new_list[i + 1]
                    if "activation" in next_layer_name or "flatten" in next_layer_name or input_features_layer_type[input_features_count] != 0: #comparing with the input features layer type bc bug in data where dense layers in a row
                        layers_to_remove.append(layer_name)
                        input_features_count -= 1
                elif i == len(layer_configs_new_list) - 1:
                    # if input_features_layer_type[input_features_count] != 0:
                    if input_features_layer_type[-1] != 0:

                        # print("input feature count: ", input_features_count)
                        layers_to_remove.append(layer_name)
                        # input_features_count -= 1
            input_features_count += 1


        #remove identified layers
        for layer_name in layers_to_remove:
            del layer_configs_new[layer_name]

        layer_configs_new_list = list(layer_configs_new.keys())

        # print("layer_configs_new_list: ", layer_configs_new_list)
        # print("input_features_layer_type: ", input_features_layer_type)
                 
        # for i, layer_val in enumerate(input_features_layer_type):
        #     print(f" input_features: {layer_val}, hls_config: {layer_configs_new_list[i]}")
        #print hls config vs input features
        
        # print("layer_configs_new: ", layer_configs_new)
        # print("length of input features layer_type vs hls config: ", len(input_features_layer_type), len(layer_configs_new))
        # for i, layer_name in enumerate(layer_configs_new_list):
        #     # inputput_feature_layer_name = input_features.iloc[i]['layer_type']
        #     print(f"input_features: {input_features.iloc[i]['layer_type']}, hls_config: {layer_name}")


        
        # Check if number of layers matches
        if len(layer_configs_new) != len(input_features):
            print("Mismatch in number of layers: layer_config: ", len(layer_configs_new)," input_features:", len(input_features))
            logger.warning(
                f"Mismatch in number of layers: {len(layer_configs_new)} in HLS config, "
                f"{len(input_features)} in model"
            )
            return
        
        # Process each layer
        layer_names = list(layer_configs_new.keys())
        
        for i, layer_name in enumerate(layer_names):
            layer_config = layer_configs_new[layer_name]
            
            # Extract precision and reuse factor
            if ("conv" in layer_name or "dense" in layer_name) and "linear" not in layer_name and "activation" not in layer_name:
                try:
                    weight = self.parse_weight_string(layer_config.get('Precision', {}).get('weight', ''))
                    input_features.at[i, 'rf'] = layer_config.get('ReuseFactor', 0)
                except Exception as e:
                    logger.warning(f"Error processing layer {layer_name}: {e}")
                    continue
            else:
                # Use previous layer's precision values for non-conv/dense layers and rf of 1
                if i > 0:
                    # input_features.at[i, 'rf'] = input_features.iloc[i - 1]['rf']
                    input_features.at[i, 'rf'] = 1
                    weight = input_features.iloc[i - 1]['prec']
                else:
                    logger.warning(f"Cannot determine precision for layer {layer_name} at index 0")
                    continue
            
            # Set precision
            input_features.at[i, 'prec'] = weight
            
            
        
        # Set strategy
        strategy_value = model_config.get('Strategy', 'latency')
        input_features['strategy'] = STRATEGY_MAPPING.get(strategy_value.lower(), 0)
        
        return input_features
    
    def save_to_csv(self, input_features: pd.DataFrame, file_path: str, savedir=None) -> str:
        """Save the processed features to a CSV file.
        
        Args:
            input_features: DataFrame containing the processed features
            file_path: Original JSON file path, used to derive CSV filename
            savedir: Optional directory to save the output file
            
        Returns:
            Path to the saved CSV file
        """
        # Create output filename
        file_name = Path(file_path).stem
        if savedir:
            output_path = os.path.join(savedir, f"{file_name}_input_features.csv")
        else:
            output_path = f"{file_name}_input_features.csv"
        
        # Save to CSV
        input_features.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")
        
        return output_path
    
    def save_labels_to_csv(self, resource_report: Dict[str, int], file_path: str, savedir=None) -> str:
        """Save the resource report (labels) to a CSV file.
        
        Args:
            resource_report: Dictionary with resource metrics
            file_path: Original JSON file path, used to derive CSV filename
            savedir: Optional directory to save the output file
            
        Returns:
            Path to the saved CSV file
        """
        # Create output filename
        file_name = Path(file_path).stem
        if savedir:
            output_path = os.path.join(savedir, f"{file_name}_labels.csv")
        else:
            output_path = f"{file_name}_labels.csv"
        
        # Convert resource report to DataFrame
        labels_df = pd.DataFrame([resource_report])
        
        # Save to CSV
        labels_df.to_csv(output_path, index=False)
        logger.info(f"Saved labels to {output_path}")
        
        return output_path
    
    def process_file(self, file_path: str, savedir=None) -> Tuple[Optional[str], Optional[str]]:
        """Process a JSON file and convert it to features CSV and labels CSV.
        
        This method now checks for valid resource report before processing.
        
        Args:
            file_path: Path to the JSON file
            savedir: Optional directory to save output files
            
        Returns:
            Tuple of (features_csv_path, labels_csv_path) or (None, None) if no valid resource report
        """
        # Check for valid resource report first
        if not self.has_valid_resource_report(file_path):
            logger.debug(f"Skipping {file_path} - no valid resource report")
            return None, None
        
        # Process JSON to DataFrame and get resource report
        input_features, resource_report = self.process_json_to_csv(file_path)
        
        # Save features to CSV
        features_path = self.save_to_csv(input_features, file_path, savedir=savedir)
        
        # Save labels to CSV
        labels_path = self.save_labels_to_csv(resource_report, file_path, savedir=savedir)
        
        # Display summary
        logger.info(f"Processed {len(input_features)} layers")
        
        return features_path, labels_path


    def process_files_to_numpy(self, file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Process multiple JSON files into NumPy arrays for ML training.
                
        Args:
            file_paths: List of paths to JSON files
            
        Returns:
            Tuple of (features_array, labels_array)
        """
        # First pass: filter valid files and determine max layers
        valid_files = []
        max_layers = 0
        all_features = []
        all_labels = []
        
        logger.info(f"Filtering {len(file_paths)} files for valid resource reports...")
        
        for file_path in file_paths:
            if not self.has_valid_resource_report(file_path):
                continue
                
            try:
                features_df, resource_report = self.process_json_to_csv(file_path)
                max_layers = max(max_layers, len(features_df))
                all_features.append(features_df)
                all_labels.append(resource_report)
                valid_files.append(file_path)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        logger.info(f"Found {len(valid_files)} files with valid resource reports")
        
        if not all_features:
            logger.error("No valid files were processed")
            return np.array([]), np.array([])
        
        # Create NumPy arrays with appropriate dimensions
        num_models = len(all_features)
        num_features = len(self.feature_columns)
        num_labels = len(self.label_columns)
        
        X = np.full((num_models, max_layers, num_features), -1, dtype=float)
        y = np.zeros((num_models, num_labels), dtype=float)
        
        # Fill arrays with data
        for i, (features_df, resource_report) in enumerate(zip(all_features, all_labels)):
            # Fill features (X)
            num_layers = len(features_df)
            for j, feature_name in enumerate(self.feature_columns):
                X[i, :num_layers, j] = features_df[feature_name].values
            
            # Fill labels (y)
            for j, label_name in enumerate(self.label_columns):
                y[i, j] = resource_report[label_name]
        
        return X, y
    
    def save_numpy_arrays(self, X: np.ndarray, y: np.ndarray, output_dir: str = '.', 
                         prefix: str = '') -> Tuple[str, str]:
        """Save features and labels as NumPy arrays.
        
        Args:
            X: Features array with shape [num_models, max_layers, num_features]
            y: Labels array with shape [num_models, num_labels]
            output_dir: Directory to save the arrays
            prefix: Optional prefix for the filenames
            
        Returns:
            Tuple of (features_path, labels_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Add prefix if provided
        features_filename = f"{prefix}_features.npy" if prefix else "features.npy"
        labels_filename = f"{prefix}_labels.npy" if prefix else "labels.npy"
        
        X_path = os.path.join(output_dir, features_filename)
        y_path = os.path.join(output_dir, labels_filename)
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        logger.info(f"Saved features array with shape {X.shape} to {X_path}")
        logger.info(f"Saved labels array with shape {y.shape} to {y_path}")
        
        return X_path, y_path
    
    def process_folders(self, folder_paths: List[str], output_dir: str = '.', 
                    prefix: str = '', save_csv: bool = False) -> Tuple[str, str]:
        """Process multiple folders containing JSON files and combine them into NumPy arrays.
        
        Args:
            folder_paths: List of paths to folders containing JSON files
            output_dir: Directory to save the output files
            prefix: Optional prefix for output filenames
            save_csv: Whether to save individual CSV files (default: False)
            
        Returns:
            Tuple of (features_path, labels_path) for the saved NumPy arrays
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect all JSON files from all folders
        all_json_files = []
        for folder in folder_paths:
            json_files = glob.glob(os.path.join(folder, "*.json"))
            logger.info(f"Found {len(json_files)} JSON files in {folder}")
            all_json_files.extend(json_files)
        
        logger.info(f"Total of {len(all_json_files)} JSON files found across all folders")
        print(f"Total of {len(all_json_files)} JSON files found across all folders")
        
        # Filter files with valid resource reports
        valid_files = []
        # for file_path in all_json_files:
        for file_path in tqdm(all_json_files, desc="Filtering files", unit="file"):
            if self.has_valid_resource_report(file_path):
                valid_files.append(file_path)
        
        logger.info(f"Found {len(valid_files)} files with valid resource reports")
        print(f"Found {len(valid_files)} files with valid resource reports")
        
        # Process each file one by one
        all_features = []
        all_labels = []
        processed_count = 0
        
        # for file_path in valid_files:
        for file_path in tqdm(valid_files, desc="Processing files", unit="file"):
            try:
                features_df, resource_report = self.process_json_to_csv(file_path)
                
                # Save CSV files if requested
                if save_csv:
                    self.save_to_csv(features_df, file_path, savedir=output_dir)
                    self.save_labels_to_csv(resource_report, file_path, savedir=output_dir)
                
                all_features.append(features_df)
                all_labels.append(resource_report)
                
                processed_count += 1
                if processed_count % 5000 == 0:
                    logger.info(f"Processed {processed_count} of {len(valid_files)} files")
                    # print(f"Processed {processed_count} of {len(valid_files)} files")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        # Check if any valid files were processed
        if not all_features:
            logger.error("No valid files were processed")
            return None, None
        
        # Create NumPy arrays with appropriate dimensions
        max_layers = max(len(df) for df in all_features)
        num_models = len(all_features)
        num_features = len(self.feature_columns)
        num_labels = len(self.label_columns)
        
        X = np.full((num_models, max_layers, num_features), -1, dtype=float)
        y = np.zeros((num_models, num_labels), dtype=float)
        
        # Fill arrays with data
        for i, (features_df, resource_report) in enumerate(zip(all_features, all_labels)):
            # Fill features (X)
            num_layers = len(features_df)
            for j, feature_name in enumerate(self.feature_columns):
                X[i, :num_layers, j] = features_df[feature_name].values
            
            # Fill labels (y)
            for j, label_name in enumerate(self.label_columns):
                y[i, j] = resource_report[label_name]
        
        # Save the combined NumPy arrays
        X_path, y_path = self.save_numpy_arrays(X, y, output_dir, prefix)

        print(f"Saved features array with shape {X.shape} to {X_path}")
        print(f"Saved labels array with shape {y.shape} to {y_path}")
        
        return X_path, y_path

if __name__ == "__main__":
    import glob
    import os
    from pathlib import Path
    
    processor = ModelProcessor()
    
    # Use the new process_folders function
    # folder_paths = [
 
    #     "./May_14_full_data/2layer/",
    #     "./May_14_full_data/3layer/",
    #     "./May_14_full_data/conv1d_may14/",
    #     "./May_14_full_data/conv2d_may14/"

    # ]
    folder_paths = [
        "./May_14_full_data/may15_conv_ii/conv1d/",
        "./May_14_full_data/2layer/",
        "./May_14_full_data/3layer/",
        "./May_14_full_data/may15_conv_ii/conv2d/",

    ]
    output_dir = "./May_15_processed/"
    
    # Process folders and save combined numpy arrays (skip CSV files)
    X_path, y_path = processor.process_folders(
        folder_paths, 
        output_dir=output_dir,
        prefix="combined",
        save_csv=False
    )
    
    # print(f"Combined data saved to {X_path} and {y_path}")
    
    # # Or, process folders and also save individual CSV files
    # X_path, y_path = processor.process_folders(
    #     folder_paths, 
    #     output_dir="./output_with_csvs/",
    #     prefix="with_csvs",
    #     save_csv=True
    # )


