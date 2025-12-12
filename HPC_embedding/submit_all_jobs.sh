#!/bin/bash

# This script submits all four combinations as separate array jobs

# Create a temporary directory for job scripts
mkdir -p temp_jobs

# Generate individual job scripts for each combination
for DATA_TYPE in kcat km; do
    for FEATURE_TYPE in smiles sequence; do
        JOB_FILE="temp_jobs/job_${DATA_TYPE}_${FEATURE_TYPE}.sh"
        
        cat > ${JOB_FILE} << 'EOF'
#!/bin/bash
#SBATCH --job-name=DATA_TYPE_FEATURE_TYPE
#SBATCH --output=logs/DATA_TYPE_FEATURE_TYPE_%A_%a.out
#SBATCH --error=logs/DATA_TYPE_FEATURE_TYPE_%A_%a.err
#SBATCH --array=0-100
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=qcb
#SBATCH --account=nmherrer_110

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p output

# Activate the conda environment
source config.sh

TOTAL_TASKS=100

python process_vectors.py \
    --data_type DATA_TYPE_VALUE \
    --feature_type FEATURE_TYPE_VALUE \
    --task_id ${SLURM_ARRAY_TASK_ID} \
    --total_tasks ${TOTAL_TASKS} \
    --output_dir output

echo "Task ${SLURM_ARRAY_TASK_ID} completed for DATA_TYPE_VALUE_FEATURE_TYPE_VALUE"
EOF
        
        # Replace placeholders with actual values
        sed -i "s/DATA_TYPE_FEATURE_TYPE/${DATA_TYPE}_${FEATURE_TYPE}/g" ${JOB_FILE}
        sed -i "s/DATA_TYPE_VALUE/${DATA_TYPE}/g" ${JOB_FILE}
        sed -i "s/FEATURE_TYPE_VALUE/${FEATURE_TYPE}/g" ${JOB_FILE}
        
        # Submit the job
        echo "Submitting job for ${DATA_TYPE}_${FEATURE_TYPE}"
        sbatch ${JOB_FILE}
    done
done

echo "All jobs submitted!"
