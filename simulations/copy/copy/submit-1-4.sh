#!/bin/bash
#SBATCH -N 1               	# request two nodes
#SBATCH -n 16 		       	# specify 64 MPI processes (8 per node)
#SBATCH -c 1			# specify 6 threads per process
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -A hpc_hpc_tyw_01
#SBATCH -o slurm-%j.out-%N # optional, name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # optional, name of the stderr, using job and first node values
#SBATCH --gres=gpu:1



 # Set some handy environment variables.
module purge
module load cuda
#module load namd/2.14/intel-2021.5.0-cuda
#module load namd/2.14/intel-2021.5.0
#module load intel/2021.5.0 intel-mpi/2021.5.1 fftw/3.3.10/intel-2021.5.0-intel-mpi-2021.5.1 namd/2.14/intel-2021.5.0
#module load intel/19.1.3



#for i in `seq 2`; do
 
 /work/jcarde7/NAMD_2.14_Linux-x86_64-multicore-CUDA/namd2 +p16 waterbox.conf > output.log 

#wait

 

 





#srun -n 64 $(which namd2) equi.namd > output.log
#srun -n 128 /home/yxan/softwares/NAMD_Git-2022-07-21_Linux-x86_64-multicore/namd2 equi.namd > output.log
#for node in `cat $PBS_NODEFILE | uniq`; do echo host $node; done > hostfile
#charmrun ++p 128 ++nodelist ./hostfile ++remote-shell ssh `which namd2` equi.namd

#/home/yxan/softwares/NAMD_Git-2022-07-21_Linux-x86_64-multicore/namd2 +p64 equi.namd > output.log
