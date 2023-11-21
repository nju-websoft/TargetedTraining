RUN_NAME=GSM8K_Flant5_large

CUDA_VISIBLE_DEVICES=0 python run.py \
  --batch_size 9 \
  --lr 3e-5 \
  --max_seq_length 192 \
  --epoch 300 \
  --gradient_accumulation_steps 3 \
  --save_path $RUN_NAME \
  --run_name $RUN_NAME \
  --eval_steps 5000 \
  --project_name targeted_training_run \
  --model_path google/flan-t5-large
