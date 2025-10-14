# this script trains the sft model on sft partition, then dpo on the sft model on dpo partition and then evaluates both the sft and dpo models

model_name="<model_name>" # just the model name e.g llama-7b or mistral-7b
model_path="<hf_model_path>" # full repo name e.g meta-llama/Llama-2-7b-hf or mistralai/Mistral-7B-v0.1
target_dataset="<dataset_name>" # one of three: cspubsumm/news_headline/oasum
sft_dataset_path="<sft_data_train_path>" # absolute path
dpo_dataset_path="<dpo_data_train_path>" # absolute path
test_dataset_path="<test_path>" # absolute path
date=$(date +'%d-%m-%Y')

sft_model_dest_path="ft-models/sft/$target_dataset-$model_name-sft-$date" # modify absolute part of the path
post_sft_merged_model_path="$sft_model_dest_path-merged"  
dpo_model_dest_path="ft-models/dpo/$target_dataset-$model_name-dpo-$date" # modify absolute part of the path
test_results_dest_path="output/inference-$target_dataset-$model_name-$date" # modify absolute part of the path
sft_test_results_dest_path="$test_results_dest_path/sft"
dpo_test_results_dest_path="$test_results_dest_path/dpo"

mkdir -p $sft_model_dest_path
mkdir -p $post_sft_merged_model_path
mkdir -p $dpo_model_dest_path
mkdir -p $test_results_dest_path
mkdir -p $sft_test_results_dest_path
mkdir -p $dpo_test_results_dest_path

python3 sft-on-base-llm.py \
	--model_name $model_path \
	--load_in_4bit \
	--use_peft \
	--batch_size 1 \
	--num_train_epochs 4 \
	--gradient_accumulation_steps 16 \
	--checkpoint_after_num_epochs 2 \
	--truncate_doc_to 800 \
	--seq_length 1600 \
	--learning_rate 0.00005 \
	--logging_step 10 \
	--target_dataset $target_dataset \
    --target_dataset_path $sft_dataset_path \
	--use_auth_token False \
	--output_dir $sft_model_dest_path > $sft_model_dest_path/out.log

python3 merge-model.py \
    --adapter_model_path $sft_model_dest_path \
    --full_model_path $post_sft_merged_model_path

python3 dpo-on-sft-llm.py \
	--model_name $post_sft_merged_model_path \
	--load_in_4bit \
	--use_peft \
	--batch_size 1 \
	--num_train_epochs 4 \
	--gradient_accumulation_steps 16 \
	--checkpoint_after_num_epochs 2 \
	--truncate_doc_to 600 \
	--max_seq_length 1600 \
    --max_prompt_length 1400 \
    --max_target_length 200 \
    --beta 0.01 \
	--learning_rate 0.000005 \
	--logging_step 10 \
	--target_dataset $target_dataset \
    --target_dataset_path $dpo_dataset_path \
	--use_auth_token False \
	--output_dir $dpo_model_dest_path > $dpo_model_dest_path/out.log

python3 evaluate-llm.py \
	--model_dir $dpo_model_dest_path \
	--truncate_doc_to 600 \
	--output_dir $dpo_test_results_dest_path \
	--max_new_tokens 350 \
	--target_dataset $target_dataset \
    --target_dataset_path $test_dataset_path > $dpo_test_results_dest_path/out.log

python3 evaluate-llm.py \
	--model_dir $sft_model_dest_path \
	--truncate_doc_to 600 \
	--output_dir $sft_test_results_dest_path \
	--max_new_tokens 350 \
	--target_dataset $target_dataset \
    --target_dataset_path $test_dataset_path > $sft_test_results_dest_path/out.log

rm -rf $post_sft_merged_model_path
