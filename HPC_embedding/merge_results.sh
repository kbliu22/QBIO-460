#!/bin/bash

#SBATCH --job-name=merge_results
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=qcb
#SBATCH --account=nmherrer_110

# Merge all chunked results back together
# Usage: sbatch merge_results.sh
# Or: bash merge_results.sh (without SLURM)
# To cleanup chunks after merge: bash merge_results.sh --cleanup


# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate the conda environment
source config.sh

# Set parameters
TOTAL_TASKS=100
OUTPUT_DIR="output"
DATA_DIR="/project2/nmherrer_110/kbliu/kbliu/unikp/embedding"
CLEANUP=false

# Check for cleanup flag
if [[ "$1" == "--cleanup" ]]; then
    CLEANUP=true
    echo "Cleanup mode enabled: will remove chunk files after successful merge"
fi

mkdir -p logs

# Create a Python script inline to do the merging
cat > temp_merge.py << 'PYTHON_SCRIPT'
import os
import pickle
import numpy as np
import torch
import sys

def load_original_data(data_type, feature_type, data_dir):
    """Load the original sequences or smiles from text files."""
    if feature_type == 'smiles':
        filename = os.path.join(data_dir, f'{data_type}_smiles.txt')
    else:
        filename = os.path.join(data_dir, f'{data_type}_sequences.txt')
    
    with open(filename, 'r') as f:
        data = [line.strip() for line in f if line.strip()]
    
    return data

def merge_results(data_type, feature_type, total_tasks, output_dir='./output', data_dir='.'):
    """
    Merge all the chunked results back together in the correct order.
    """
    print(f"Merging results for {data_type}_{feature_type}")
    
    # Load original data
    print(f"Loading original {feature_type} data...")
    original_data = load_original_data(data_type, feature_type, data_dir)
    print(f"Loaded {len(original_data)} original entries")
    
    # Collect all results
    results = []
    missing_files = []
    chunk_files = []
    
    for task_id in range(total_tasks):
        filename = os.path.join(
            output_dir, 
            f'{data_type}_{feature_type}_vectors_task_{task_id}.pkl'
        )
        chunk_files.append(filename)
        
        if not os.path.exists(filename):
            print(f"WARNING: Missing file {filename}")
            missing_files.append(task_id)
            continue
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            results.append(data)
        
        print(f"Loaded task {task_id}: indices {data['start_idx']} to {data['end_idx']}")
    
    if missing_files:
        print(f"ERROR: Missing {len(missing_files)} files: {missing_files}")
        return False, []
    
    # Sort by start index to ensure correct order
    results.sort(key=lambda x: x['start_idx'])
    
    # Concatenate all vectors
    if isinstance(results[0]['vectors'], torch.Tensor):
        merged_vectors = torch.cat([r['vectors'] for r in results], dim=0)
        # Convert to numpy for npz storage
        merged_vectors = merged_vectors.cpu().numpy()
    else:
        merged_vectors = np.vstack([r['vectors'] for r in results])
    
    print(f"Merged embeddings shape: {merged_vectors.shape}")
    
    # Verify we have the same number of embeddings as original data
    if len(original_data) != merged_vectors.shape[0]:
        print(f"WARNING: Mismatch between original data ({len(original_data)}) and embeddings ({merged_vectors.shape[0]})")
    
    # Convert original data list to numpy array of strings
    original_data_array = np.array(original_data, dtype=object)
    
    # Save everything in a single npz file
    output_file = os.path.join(output_dir, f'{data_type}_{feature_type}_vectors_merged.npz')
    np.savez_compressed(
        output_file,
        embeddings=merged_vectors,
        original_data=original_data_array,
        data_type=np.array([data_type]),
        feature_type=np.array([feature_type])
    )
    
    print(f"Saved merged results to {output_file}")
    print(f"  - embeddings: shape {merged_vectors.shape}")
    print(f"  - original_data: {len(original_data)} entries")
    print(f"  - data_type: {data_type}")
    print(f"  - feature_type: {feature_type}")
    
    return True, chunk_files

# Get parameters from command line or use defaults
total_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 10
output_dir = sys.argv[2] if len(sys.argv) > 2 else './output'
data_dir = sys.argv[3] if len(sys.argv) > 3 else '.'

# Merge all combinations
combinations = [
    ('kcat', 'smiles'),
    ('kcat', 'sequence'),
    ('km', 'smiles'),
    ('km', 'sequence')
]

success_count = 0
all_chunk_files = []

for data_type, feature_type in combinations:
    try:
        success, chunk_files = merge_results(data_type, feature_type, total_tasks, output_dir, data_dir)
        if success:
            print(f"✓ Successfully merged {data_type}_{feature_type}\n")
            success_count += 1
            all_chunk_files.extend(chunk_files)
        else:
            print(f"✗ Failed to merge {data_type}_{feature_type}\n")
    except Exception as e:
        print(f"✗ Error merging {data_type}_{feature_type}: {str(e)}\n")
        import traceback
        traceback.print_exc()

print(f"\nMerged {success_count}/{len(combinations)} combinations successfully")

# Write chunk files list for cleanup
if success_count == len(combinations):
    with open(f'{output_dir}/chunk_files_to_remove.txt', 'w') as f:
        for chunk_file in all_chunk_files:
            f.write(f"{chunk_file}\n")
    print(f"All merges successful. Chunk file list saved to {output_dir}/chunk_files_to_remove.txt")
PYTHON_SCRIPT

# Run the merge script
echo "Starting merge process..."
python temp_merge.py ${TOTAL_TASKS} ${OUTPUT_DIR} ${DATA_DIR}

# Clean up temporary script
rm temp_merge.py

# Cleanup chunk files if requested and merge was successful
if [ "$CLEANUP" = true ] && [ -f "${OUTPUT_DIR}/chunk_files_to_remove.txt" ]; then
    echo ""
    echo "Cleaning up chunk files..."
    
    # Count files before removal
    file_count=$(wc -l < "${OUTPUT_DIR}/chunk_files_to_remove.txt")
    echo "Removing ${file_count} chunk files..."
    
    # Remove chunk files
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            rm "$file"
            echo "  Removed: $file"
        fi
    done < "${OUTPUT_DIR}/chunk_files_to_remove.txt"
    
    # Remove the list file
    rm "${OUTPUT_DIR}/chunk_files_to_remove.txt"
    
    echo "Cleanup complete!"
else
    echo ""
    if [ "$CLEANUP" = false ]; then
        echo "Chunk files preserved. To remove them later, run:"
        echo "  bash merge_results.sh --cleanup"
        echo "Or manually: rm ${OUTPUT_DIR}/*_task_*.pkl"
    fi
fi

echo "Merge complete!"
