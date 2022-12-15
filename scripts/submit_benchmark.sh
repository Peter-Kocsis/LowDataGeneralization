#!/bin/bash

LOCAL="$1"
SCRIPT_ARGUMENTS="$2"

if [ $LOCAL == "local" ]; then
    bash scripts/local/run_benchmark.sh $SCRIPT_ARGUMENTS
else
    sbatch --export=ALL,SCRIPT_ARGUMENTS=$SCRIPT_ARGUMENTS scripts/cluster/slurm_submit_benchmark.sbatch
fi