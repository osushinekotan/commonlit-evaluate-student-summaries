#!/bin/bash

# run.sh
# Script to execute specific tasks for a given experiment.
#
# Usage:
# 1. Single Task Execution:
#    sh run.sh [EXPERIMENT_NUMBER] [TASK_NAME]
#    Example: sh run.sh 000 preprocess
#
# 2. Multiple Task Execution:
#    sh run.sh [EXPERIMENT_NUMBER] [TASK1_NAME] [TASK2_NAME] ...
#    Example: sh run.sh 000 preprocess train

exp=$1
shift  # Remove the first argument (exp) to process tasks

cd ~/../workspace/

# Iterate over the remaining arguments (tasks)
for task in "$@"; do
    case $task in
      preprocess)
        poetry run python src/customs/v1/preprocess.py experiment=${exp}
        ;;
      train)
        poetry run python src/customs/v1/train.py experiment=${exp}
        ;;
      inference)
        poetry run python src/customs/v1/inference.py experiment=${exp}
        ;;
      deploy)
        poetry run python src/customs/deploy.py experiment=${exp}
        ;;
      *)
        echo "Invalid task name: $task! Available tasks: preprocess, train, inference, deploy."
        ;;
    esac
done
