#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=02:00:00

module load tools/prod
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate fenicsx_torch
  
# Copy input file to $TMPDIR
cp -r $HOME/BTE-NO $TMPDIR/

cd $TMPDIR/BTE-NO

# Run application. use timeout to properly close script
timeout 7.9h python src/main.py --latent-method random --sources 0.5 50.0 --res 12.0 --vf 0.2
# Outputting everything to file. useful info is in log file

# Copy required files back
cp $TMPDIR/BTE-NO/logs/* $HOME/BTE-NO/logs/