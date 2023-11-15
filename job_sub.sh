#!/bin/bash
#
#SBATCH --job-name=lstm
#
#SBATCH --account=psy53c17
#SBATCH -o /data/users2/ziqbal5/abc/MILC_LSTM/training_output/%j.out #STDOUT
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=qTRDGPUH
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
eval "$(conda shell.bash hook)"
conda activate z_env
#%module load python
#python -m run_downstream_model 0 2 4
#python -m Pretraining_HCP 0 1 3

#python -m run_downstream_model --pre-training milc --script-ID 2 --exp NPT --method sub-lstm --jobid $SLURM_JOBID --encoder lstmM &
#python -m run_downstream_model --pre-training milc --script-ID 2 --exp UFPT --method sub-lstm --jobid $SLURM_JOBID --encoder lstmM &


#python -m Pretraining_HCP --pre-training milc --script-ID 2 --exp UFPT --method sub-lstm --jobid $SLURM_JOBID --encoder lstm &
#python -m Pretraining_HCP --pre-training milc --script-ID 2 --exp UFPT --method sub-lstm --jobid $SLURM_JOBID --encoder lstmM &

#python -m Syndata_tr --pre-training milc --script-ID 2 --exp NPT --method sub-lstm --jobid $SLURM_JOBID
python -m Interp_syndata2 --pre-training milc --script-ID 2 --exp UFPT --method sub-lstm --jobid $SLURM_JOBID --viz
#wait

