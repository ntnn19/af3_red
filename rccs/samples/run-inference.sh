#!/bin/sh
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:ngpus=1
#PBS -l walltime=2:00:00

if [ ! -z "${PBS_O_WORKDIR}" ]; then
  cd "${PBS_O_WORKDIR}"
fi

module -s purge
. $HOME/apl/alphafold/af301-red_init.sh

RUNAF3=$HOME/apl/alphafold/run-af-301-red.sh

#
# PLEASE INPUT THE DIRECTORY PATH WHERE YOUR MODEL FILE RESIDES
#
MYMODELDIR=~/af3_params/models

# notes:
# - YOU NEED TO OBTAIN MODEL PARAMETER FILE BY YOURSELF.
# - input json file for inference is the output of data-pipeline (MSA) run
# - actual options to run_alphafold.py can be displayed with "--dryrun -e".
# - GPU is required for inference.
#
# options:
#   -j <input json>, --json_path=<input json>
#     Path to input json file (single input file).
#   -i <input directory>, --input_dir=<input directory>
#     Path to directory which contains input json files (multiple input files).
#   -o <output directory>, --output_dir=<output directory>
#     Path to output directory.
#   -m <model directory>, --model_dir=<model directory>
#     Path to model for inference (THIS IS NOT PROVIDED BY RCCS).
#   -M, --msa, --msaonly
#     Do only data pipeline (MSA).
#   -I, --inference, --inferenceonly
#     Skip MSA and do only inference. (GPU required)
#   --max_msa <int>
#     Set the maximum number of MSA sequences.
#   --bias_sigma <float>
#     Set the width of the Gaussian potential.
#   --bias_weight <float>
#     Set the strength of the Gaussian potential.
#   --fit_all (Default: True)
#     Whether to use all chains for computing the biasing forces.
#   --fit_chain <strs>
#     List of chain IDs used for computing the biasing forces.
#   -e, --echo
#     Echo actual options to run_alphafold.py
#   --dryrun
#     Do not run actually.
#
#   Other run_alphafold.py options may also be accepted.

$RUNAF3 \
   -j af_output/tf1beta/tf1beta_data.json \
   -o af_output \
   -m ${MYMODELDIR} \
   -I \
   --fit_all=False \
   --fit_chain=A \
   --bias_sigma=2.0 \
   --bias_weight=90.0