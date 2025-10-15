from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM
import torch
from argparse import ArgumentParser

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("--adapter_model_path", type=str, help="Path to the sft adapter weights")
	parser.add_argument("--full_model_path", type=str, help="path where to save the merged model")
	
	args = parser.parse_args()
 
	config = PeftConfig.from_pretrained(args.adapter_model_path)
	print("SFT adapter base model:",config.base_model_name_or_path)
	base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto",)
	model = PeftModel.from_pretrained(base_model, args.adapter_model_path)
	merged_model = model.merge_and_unload()

	merged_model.save_pretrained(args.full_model_path, safe_serialization=True)