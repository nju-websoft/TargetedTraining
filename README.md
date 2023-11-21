# Targeted Training for Math Word Problems with Large Language Models


## Requirements
- torch=1.13.1
- transformers=4.28.1
- wandb
- nltk

## How to run
Running scripts are put under ``scripts/``. Modify the ``--model_path`` args to select backbone small model (``google/flan-t5-large``, ``google/flan-t5-base``, ``t5-large`` or ``t5-base``).

``scripts/run.sh`` is the script with targeted training and ``scripts/run_wo_targeted_training.sh`` is the script without targeted training.