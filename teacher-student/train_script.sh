deepspeed --include localhost:0,1,2,3 src/run_summarization.py --fp16 \
  --deepspeed src/deepspeed_config.json \
  --dataset_name xsum \
  --model_name_or_path facebook/bart-large \
  --do_train --evaluation_strategy no \
  --label_smoothing 0.1 --learning_rate 3e-5 --gradient_accumulation_step 4 --per_device_train_batch_size 8 \
  --max_source_length 512 --max_target_length 64 \
  --warmup_steps 500 --max_grad_norm 0.1 --max_steps 15000 --save_strategy no \
  --output_dir out_xsum --overwrite_cache --remove_unused_columns true --additional_reference_file additional_summary.txt

# # Optionally without deepspeed
# python src/run_summarization.py --fp16 \
#   --dataset_name xsum \
#   --model_name_or_path facebook/bart-large \
#   --do_train --evaluation_strategy no \
#   --label_smoothing 0.1 --learning_rate 3e-5 --gradient_accumulation_step 4 --per_device_train_batch_size 8 \
#   --max_source_length 512 --max_target_length 64 \
#   --warmup_steps 500 --max_grad_norm 0.1 --max_steps 15000 --save_strategy no \
#   --output_dir out_xsum --overwrite_cache --remove_unused_columns true