learnrate=(5e-1)
expids=('re_modularPT')
run_idx=0
gpu_id=4
let "seed=42+$run_idx"

data_files=("5stages/50shot_$run_idx rel_$run_idx")
# data_files=("5stages_combined/50shot_$run_idx rel_$run_idx")
stage_ids=('0 none 3')

check_free_gpu() {
  for gpuid in {0..7}
  do
    var=`nvidia-smi --query-gpu=index,memory.used --format=csv | grep "$gpuid, 0"`  
    echo "$gpuid, 0"
    if [ -z "$var" ]; then
      echo -n "$gpuid busy, "
    else
      gpu_id=$gpuid
      echo "set gpu_id to $gpuid "
      sleep 5
      var=`nvidia-smi --query-gpu=index,memory.used --format=csv | grep "$gpu_id, 0"`
      if [ ! -z "$var" ]; then
        break
      fi
    fi
  done
}

check_free_gpu # 1st execution
while [ -z "$var" ]
do
  sleep 5m
  echo "trying $expids\_$run_idx"
  check_free_gpu
done
echo $var

for expid in ${expids[@]}
do
  for onerate in ${learnrate[@]}
  do
    for iter in $data_files
    do
      for stage_id in $stage_ids
      do
          echo "------------------------------" 
          IFS=' ' read data_file run_id <<< "${iter}"
          IFS=' ' read stage_num stage_data ckpt_num <<< "${stage_id}"
          echo "stage_num: "$stage_num
          echo "run_id: "$run_id
          echo "data file: "$data_file
          echo "expid: "$expid
          python -m torch.distributed.launch --nproc_per_node 1 --master_port 2953$gpu_id main.py \
                  --cuda $gpu_id \
                  --log_name $expid\_$run_idx \
                  --wild_version train\
                  --lr $onerate \
                  --lm_adapted_path /export/home/prompting/lm_adapted_models/t5.1.1.lm100k.large/pytorch_model.bin \
                  --cache_dir /export/home/cache \
                  --train_file_name ./data/fewrel/$data_file$/train.txt \
                  --valid_file_name ./data/fewrel/$data_file$/valid.txt \
                  --test_file_name ./data/fewrel/$data_file\_$stage_num/test.txt \
                  --gradient_accumulation_steps 1 \
                  --batch_size_per_gpu 8 \
                  --valid_size_per_gpu 20 \
                  --test_size_per_gpu 15 \
                  --max_epoch 256  \
                  --save_step 18000 \
                  --eval_epoch 10 \
                  --eval_start_epoch 128\
                  --log_step 20 \
                  --concat_mode 'right_concat' \
                  --save_dir T5ModularPrompt_fewshot_right_ckpt_v$expid\_$run_id  \
                  --seed $seed \
                  --model T5ModularPrompt \
                  --max_length 128 \
                  --max_gen_length 16 \
                  --prompt_length 160 \
                  --dataset fewrel \
                  --test_analysis none \
                  --verbose \
                  --wild_do_test all_seen \
                  --label_data_initialization \
                  --subset_inv \
                  --subset_inv_type length_gt \
                  --mean_prob 0.5 \
                  --enable_forward_transfer \
                  --forward_transfer_type label_embedding_similarity_v2 \
                  --forward_transfer_similarity_type top3 \
                  

        echo "++++++++++++++++++++++++++++++"
        #ps aux | grep "cuda 1" | awk '{print $2}' | xargs kill -9
        let "run_id+=1"
        let "seed+=1"
      done
    done
  done
done
