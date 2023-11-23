#!/bin/sh
#BSUB -J DDPM_Training
#BSUB -o Logs/Training_Logs_%J.out
#BSUB -e Logs/Errors_Logs_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5G]"
#BSUB -W 24:00
#BSUB -N 
#BSUB 
# end of BSUB options

module load cuda/11.1

source venv3/bin/activate

python DDPM_high_res.py --dataset_path "MNIST" --run_name "MNIST_3_23_11"

