#!/bin/bash
#SBATCH --partition=a1-batch
#SBATCH --job-name=lecture-run
#SBATCH --output=job.stdout
#SBATCH --ntasks=1
#SBATCH --qos=a1-batch-qos

. $HOME/spring2025-lectures/main/bin/activate

"$@"
