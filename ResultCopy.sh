#!/bin/bash

# --- Configuration ---
# POD_NAME="dd-probe-pod"
POD_NAME="dd-tprobe-pod"
# CONTAINER_NAME="dd-mypod" # Optional, but good practice if multiple containers exist

#older
# REMOTE_BASE_PATH="/dima-pvc/wa-dd-prediction-model/results"
# LOCAL_DEST_BASE_PATH="/Users/dimademler/Desktop/UCSD/labs/wa-dd-prediction-model/results"

# newer
REMOTE_BASE_PATH="/dima-pvc/wa_hls4ml_models/results"
LOCAL_DEST_BASE_PATH="/Users/dimademler/Desktop/UCSD/labs/wa_datasetRemake_mulder_down/gitlab_wa_hls4ml_models/results"




# FOLDER_TO_COPY="SAVE_FOLDER" # The specific folder you want to copy
# FOLDER_TO_COPY="enhanced_GAT__night_May_27_test" # The specific folder you want to copy
# FOLDER_TO_COPY="y_29_GAT_enhanced" # The specific folder you want to copy
# FOLDER_TO_COPY="y_29_GNN_convOnly_baseline" # The specific folder you want to copy
FOLDER_TO_COPY="y_03_GAT_simple" # The specific folder you want to copy

# --- Construct Full Paths ---
REMOTE_FULL_PATH="${REMOTE_BASE_PATH}/${FOLDER_TO_COPY}"
LOCAL_FULL_PATH="${LOCAL_DEST_BASE_PATH}/${FOLDER_TO_COPY}"

# --- User Feedback ---
echo "Attempting to copy from pod: ${POD_NAME}"
echo "Source path (in pod):     ${REMOTE_FULL_PATH}"
echo "Destination path (local): ${LOCAL_FULL_PATH}"
echo "------------------------------------------"

# --- Pre-check: Ensure local destination base exists ---
if [ ! -d "$LOCAL_DEST_BASE_PATH" ]; then
    echo "⚠️  Warning: Local destination base directory '${LOCAL_DEST_BASE_PATH}' does not exist."
    echo "    Attempting to create it..."
    mkdir -p "$LOCAL_DEST_BASE_PATH"
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to create local base directory. Please check permissions."
        exit 1
    fi
    echo "    Local base directory created."
fi

# --- Pre-check: Check if destination folder already exists ---
if [ -d "$LOCAL_FULL_PATH" ]; then
    echo "⚠️  Warning: Destination folder '${LOCAL_FULL_PATH}' already exists."
    echo "    The 'kubectl cp' command might overwrite existing files or fail if non-empty."
    read -p "    Do you want to continue? (y/N) " -n 1 -r
    echo # Move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "    Operation cancelled by user."
        exit 0
    fi
fi

# --- The 'kubectl cp' Command ---
# We use -c ${CONTAINER_NAME} to be explicit, though it might not be strictly
# necessary if it's the only container in the pod.
kubectl cp "${POD_NAME}":"${REMOTE_FULL_PATH}" "${LOCAL_FULL_PATH}" -c "${CONTAINER_NAME}"

# --- Check the result ---
if [ $? -eq 0 ]; then
    echo "✅ Success! Folder '${FOLDER_TO_COPY}' copied to '${LOCAL_DEST_BASE_PATH}'."
else
    echo "❌ Error: 'kubectl cp' command failed. Please check:"
    echo "    - Is the pod '${POD_NAME}' running?"
    echo "    - Does the path '${REMOTE_FULL_PATH}' exist in the pod?"
    echo "    - Do you have 'kubectl' access and permissions?"
    exit 1
fi

exit 0