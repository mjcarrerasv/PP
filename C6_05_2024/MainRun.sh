#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64GB
#SBATCH --time=36:00:00
##SBATCH --account=plg15_econ
##SBATCH --partition=sla-prio
#SBATCH --output=LogFiles/RunSim_%j.log

cd /storage/home/mqc6502/work/Carreras_PP/C4_05_2024/C6_05_2024
git pull
module load julia
julia Main_fn_vssh1.jl