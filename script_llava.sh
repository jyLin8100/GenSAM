#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8      # Request 1 core
#$ -l h_rt=20:0:0  # Request 1 hour runtime
#$ -l h_vmem=11G   # Request 1GB RAM
#$ -l gpu=1     # request 1 GPU
##$ -l cluster=andrena # use the Andrena nodes


source /data/DERI-Gong/jl010/envLISA2/bin/activate


config=CHAMELEON_LLaVA1.5
python main.py --config config/$config.yaml --visualization 
