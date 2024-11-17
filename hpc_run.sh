#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=03:00:00

module load tools/prod
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate fenicsx_torch

# Copy input file to $TMPDIR
cp -r $HOME/BTE-NO $TMPDIR/

cd $TMPDIR/BTE-NO

# Run application
timeout 2.92h python src/main.py --latent-method preloaded --res 12.0 --vf 0.2

# Copy required files back
cp -r $TMPDIR/BTE-NO/logs/* $HOME/BTE-NO/logs/
