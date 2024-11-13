#!/bin/bash
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=20:mem=12gb:ngpus=1:gpu_type=RTX6000

module load tools/prod
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate fenics2018
  
# Copy input file to $TMPDIR
cp -r $HOME/GAOpt $TMPDIR/

cd $TMPDIR/GAOpt
# cd $PBS_O_WORKDIR/GAOpt

# Run application. use timeout to properly close script
timeout 7.9h python optim_main.py  > logs/printoutputflux
# Outputting everything to file. useful info is in log file

# Copy required files back
cp $TMPDIR/GAOpt/logs/* $HOME/GAOpt/logs/