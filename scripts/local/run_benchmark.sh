#!/bin/bash

SCRIPT_ARGUMENTS="$@"
JOB_ID="$(date +"%s")"
echo Starting local experiment with arguments: $SCRIPT_ARGUMENTS
python -u -m lowdataregime.active_learning.active_learning $SCRIPT_ARGUMENTS --run_id 0 --job_id $JOB_ID
python -u -m lowdataregime.active_learning.active_learning $SCRIPT_ARGUMENTS --run_id 1 --job_id $JOB_ID
python -u -m lowdataregime.active_learning.active_learning $SCRIPT_ARGUMENTS --run_id 2 --job_id $JOB_ID
python -u -m lowdataregime.active_learning.active_learning $SCRIPT_ARGUMENTS --run_id 3 --job_id $JOB_ID