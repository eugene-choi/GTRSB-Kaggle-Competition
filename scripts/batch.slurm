#!/bin/bash
#SBATCH --job-name=vision_test
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ec2684/%j_%x.out
#SBATCH --error=/scratch/ec2684/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=6:30:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --mail-user=<ec2684@nyu.edu>
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

## TODO: Run your code in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.
singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python ./main.py --order 12
python ./main.py --order 13
python ./main.py --order 14
python ./main.py --order 15
python ./main.py --order 16
python ./main.py --order 17
"