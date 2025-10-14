# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer
from peft import PeftModel, PeftConfig

from trl import DPOTrainer, DataCollatorForCompletionOnlyLM
import numpy as np
import evaluate
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import pandas as pd
import huggingface_hub
import json

huggingface_hub.login(token='<hf_tokens>')
# nltk.download('punkt')
# nltk.download('wordnet')

tqdm.pandas()

# Define and parse arguments.
@dataclass
class ScriptArguments:

	model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
	dataset_name: Optional[str] = field(
		default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
	)
	log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
	learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "the learning rate"})
	batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
	max_seq_length: Optional[int] = field(default=1600, metadata={"help": "max sequence length"})
	max_prompt_length: Optional[int] = field(default=1000, metadata={"help": "Prompt sequence length"})
	max_target_length: Optional[int] = field(default=800, metadata={"help": "Target sequence length"})
	beta: Optional[float] = field(default=0.01, metadata={"help": "beta for dpo"})
	gradient_accumulation_steps: Optional[int] = field(
		default=16, metadata={"help": "the number of gradient accumulation steps"}
	)
	load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
	load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
	use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
	trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
	output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
	peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
	peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
	logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
	use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
	num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
	max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
	save_steps: Optional[int] = field(
		default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
	)
	save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
	push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
	hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})

	# adding new arguments
	checkpoint_after_num_epochs: Optional[int] = field(default=None, metadata={"help": "the number of epochs after which to checkpoint"})
	truncate_doc_to: Optional[int] = field(default=1200, metadata={"help": "the number of words to truncate the document to"})
	target_dataset: Optional[str] = field(default="all", metadata={"help": "target dataset name: one of oasum/cspubsumm/news_headline"})
	target_dataset_path: Optional[str] = field(default="all", metadata={"help": "target dataset path"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
	raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
	quantization_config = BitsAndBytesConfig(
		load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
	)
	# Copy the model to each device
	# device_map = {"": Accelerator().local_process_index}
	device_map = "auto"
	torch_dtype = torch.bfloat16
else:
	device_map = None
	quantization_config = None
	torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
	script_args.model_name,
	quantization_config=quantization_config,
	device_map=device_map,
	trust_remote_code=script_args.trust_remote_code,
	torch_dtype=torch_dtype,
	use_auth_token=script_args.use_auth_token,
)

model.config.use_cache = False

ALPACA_PROMPT_FORMAT = (
	"You are a helpful, respectful and honest assistant. Your task is to write spotlight of documents. A spotlight is a short, concise overview of the document. It is meant to spark curiosity in the reader to read the entire article. But it does not provide much coverage of the content of the document and that is how it differs from a summary. Below is an instruction, paired with a document that provides further context.\n\n"
	"### Instruction:\n{instruction}\n\n### Document:\n{document}\n\n### Response:\n"
)

# Step 2: Load the dataset
def preprocess_examples(examples, tokenizer, truncate_doc_to, threshold_max_tokens, target_dataset):
	

	if target_dataset == "oasum":

		documents = examples['document']
		
		prefered_summaries = examples['summary']
		rejected_summaries = examples['gpt-summary']

		aspect = examples["aspect"]
		title = examples["title"]
		rel_sents_lists = examples["aspect_sents"]
		dataset = "oasum"
		INSTRUCTION_FORMAT = (
			"Write the spotlight of the following document based on the {aspect} of {title}. The spotlight need not have detailed information coverage but should include only the key points that makes the reader curious about the entire article. Write the spotlight in this way."
		)
	elif target_dataset == "cspubsumm":
		main_titles = examples["main-title"]
		dataset = "cspubsumm"
		documents = []
		prefered_summaries = []
		rejected_summaries = []

		for abstract, introduction, conclusion, highlights, gpt_summary in zip(examples["abstract"], examples["introduction"], examples["conclusion"], examples["author-highlights"], examples["summary"]):

			smoothed_highlights = ". ".join(highlights.split("."))

			if smoothed_highlights.strip()!="" and smoothed_highlights.strip()[-1] not in [".","!","?"]:
				smoothed_highlights += "."
			
			# scientific_article = f"Abstract:\n{truncated_abstract}\n\nIntroduction:\n{truncated_introduction}\n\nConclusion:\n{truncated_conclusion}"
			scientific_article = f"{introduction}"
			documents.append(scientific_article)
			prefered_summaries.append(smoothed_highlights)
			rejected_summaries.append(gpt_summary)

		INSTRUCTION_FORMAT = (
			"Write the spotlight of the following scientific article entitled {main_title} presented as a document. The spotlight need not have detailed information coverage but should include only the key points that makes the reader curious about the entire article. Write the spotlight in this way."
		)
		examples["document"] = documents
		
	elif target_dataset == "news_headline":
		dataset = "news_headline"
		documents = []
		prefered_summaries = []
		rejected_summaries = []
		for article, short_description, headline, summary in zip(examples["doc_text"],examples["short_description"],examples["headline"],examples["summary"]):
			documents.append(article)
			prefered_summaries.append((headline + ". " + short_description).strip().replace("\n",""))
			rejected_summaries.append(summary.replace("\n",""))
		INSTRUCTION_FORMAT = (
			"Write a headline for the following news article presented as document. Also include a short description of the article in not more than 4 sentences that can be presented as its highlight. The headline together with the highlight is the spotlight for the news article. It need not have detailed information coverage but should make the reader curious about the news article. Write the spotlight in this way.{void}"
		)
		examples["document"] = documents
		
	else:
		print("invalid dataset")

	prompts = []
	validity = []
	chosen = []
	rejected = []

	for idx in range(len(documents)):

		if documents[idx] is None:
			prompts.append("")
			validity.append(0)
			chosen.append("")
			rejected.append("")
			continue

		if prefered_summaries[idx].strip() == "" or rejected_summaries[idx].strip() == "":
			prompts.append("")
			validity.append(0)
			chosen.append("")
			rejected.append("")
			continue

		# truncate the document till last fullstop before truncate_doc_to
		documents[idx] = " ".join(documents[idx].split(" ")[:truncate_doc_to])
		documents[idx] = ".".join(documents[idx].split(".")[:-1]) + "."

		if dataset == "oasum":

			sentences_retained = sent_tokenize(documents[idx])

			# if at least one of the relevant sentences is not in the truncated document, remove the example from the dataset (i.e. mark it as invalid)
			if max(rel_sents_lists[idx]) >= len(sentences_retained):
				prompts.append("")
				validity.append(0)
				chosen.append("")
				rejected.append("")
				continue

		if dataset == "oasum":
			instruction = INSTRUCTION_FORMAT.format(aspect=aspect[idx], title=title[idx])
		elif dataset == "cspubsumm":
			instruction = INSTRUCTION_FORMAT.format(main_title=main_titles[idx].replace("\n"," "))
		elif dataset == "news_headline":
			instruction = INSTRUCTION_FORMAT.format(void="")
		else:
			print("invalid dataset")

		_prompt = ALPACA_PROMPT_FORMAT.format(instruction=instruction,document=documents[idx])

		prompts.append(_prompt)
		validity.append(1)
		chosen.append(prefered_summaries[idx])
		rejected.append(rejected_summaries[idx])

	model_inputs = {}
	model_inputs["prompt"] = prompts
	model_inputs["validity"] = validity
	model_inputs["chosen"] = chosen
	model_inputs["rejected"] = rejected

	return model_inputs

def get_filter_dict(dataset):
	filter_dict = {}
	for split in dataset.keys():
		filter_dict[split] = []
		for idx in range(len(dataset[split])):
			if dataset[split][idx]["validity"] == 1:
				filter_dict[split].append(idx)
	return filter_dict

dataset = load_dataset("json", data_files={"train": script_args.target_dataset_path})

column_names = list(dataset["train"].features)

with open(f"{script_args.model_name}/config.json","r") as fin:
    raw_config = json.load(fin)
tokenizer = AutoTokenizer.from_pretrained(raw_config["_name_or_path"])

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
print("llama tokenizer pad token id: ", tokenizer.pad_token_id)
print("llama model pad token id: ", model.config.pad_token_id)

dataset = dataset.map(preprocess_examples,batched=True,fn_kwargs={"tokenizer":tokenizer, "truncate_doc_to":script_args.truncate_doc_to, "threshold_max_tokens":script_args.max_seq_length, "target_dataset":script_args.target_dataset},remove_columns=column_names)

if "document" in dataset["train"].features:
	dataset["train"] = dataset["train"].remove_columns("document")

filter_dict = get_filter_dict(dataset)
for split in dataset.keys():
	dataset[split] = dataset[split].select(filter_dict[split])


print("DATASET SUMMARY AFTER PROCESSING:\n",dataset)

print("TEXT SAMPLE:\n",dataset["train"][0]["prompt"])

print("CHOSEN SAMPLE\n",dataset["train"][0]["chosen"])

print("REJECTED SAMPLE:\n",dataset["train"][0]["rejected"])

###################### SAVING CHECKPOINTS ############################

# making checkpoint_after_num_epochs aggresively override all arguments related to saving checkpoints

# compute the number of epoch intervals between two checkpoints are saved, if checkpoint_after_num_epochs is specified that overrides the save_steps argument
if script_args.checkpoint_after_num_epochs is not None:
	script_args.save_steps = ((len(dataset['train']) // script_args.batch_size) // script_args.gradient_accumulation_steps) * script_args.checkpoint_after_num_epochs

	# if save_total_limits restricts from saving checkpoints at all epochs possible as specified by checkpoint_after_num_epochs, then save_total_limit is set to the number of epochs possible, i.e checkpoint_after_num_epochs overrides save_total_limit
	if (script_args.num_train_epochs // script_args.checkpoint_after_num_epochs) > script_args.save_total_limit:
		script_args.save_total_limit = (script_args.num_train_epochs // script_args.checkpoint_after_num_epochs) + 2

######################################################################

# Step 3: Define the training arguments
training_args = TrainingArguments(
	output_dir=script_args.output_dir,
	per_device_train_batch_size=script_args.batch_size,
	gradient_accumulation_steps=script_args.gradient_accumulation_steps,
	learning_rate=script_args.learning_rate,
	logging_steps=script_args.logging_steps,
	num_train_epochs=script_args.num_train_epochs,
	max_steps=script_args.max_steps,
	report_to=script_args.log_with,
	save_steps=script_args.save_steps,
	# save_total_limit=script_args.save_total_limit,
	push_to_hub=script_args.push_to_hub,
	hub_model_id=script_args.hub_model_id,
	lr_scheduler_type="cosine",
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
	peft_config = LoraConfig(
		r=script_args.peft_lora_r,
		lora_alpha=script_args.peft_lora_alpha,
		bias="none",
		task_type="CAUSAL_LM",
		target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",  "up_proj",  "down_proj"]
	)
else:
	peft_config = None

# Step 5: Define the Trainer
trainer = DPOTrainer(
		model,
		ref_model=None,
		args=training_args,
		beta=script_args.beta,
		train_dataset=dataset["train"],
		tokenizer=tokenizer,
		max_length=script_args.max_seq_length,
		max_target_length=script_args.max_target_length,
		max_prompt_length=script_args.max_prompt_length,
		peft_config=peft_config,
	)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)

