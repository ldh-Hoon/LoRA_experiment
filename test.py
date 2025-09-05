import os
import sys
import yaml
import torch
import argparse
import json
from datasets import load_dataset
from evaluate import load as load_metric

import numpy as np
import importlib.util

from transformers.trainer import Trainer
from peft import LoraConfig, get_peft_model

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import PeftModelForSequenceClassification, PeftConfig, PeftModel, LoraConfig


class EvaluationRunner:
    def __init__(self, config_path: str, model_path: str):
        self.config = self._load_config(config_path)
        self.task = self.config['task']
        self.model_path = model_path
        self.device = self._get_device()
        
        self.tokenizer = self._load_tokenizer()
        self.validation_dataset = self._load_and_preprocess_dataset()
        self.model = self._load_model()
        self._print_lora_parameter_ratio()
        self._remain_pc_lora()
        self._print_lora_parameter_ratio()

        self.trainer = self._setup_trainer()

    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_device(self):
        device = self.config['training'].get('device', 'cpu')
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU.")
            return "cpu"
        return device

    def _load_tokenizer(self):
        model_name = self.config['model']['pretrained_model_name']
        if self.config['model'].get('model_type') == "local":
            print("Local model specified. Using a default tokenizer for text processing.")
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_and_preprocess_dataset(self):
        dataset_config = self.config['dataset']
        dataset_type = dataset_config.get('dataset_type')
        
        if dataset_type == "hub":
            print(f"Loading dataset from Hugging Face Hub: {dataset_config['name']}")
            raw_datasets = load_dataset(dataset_config['name'])
            
            def preprocess_function(examples):
                texts = [f"question: {q} passage: {p}" for q, p in zip(examples['question'], examples['passage'])]
                labels = [1 if a else 0 for a in examples['answer']]
                tokenized_inputs = self.tokenizer(
                    texts, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=512
                )
                tokenized_inputs['labels'] = labels
                return tokenized_inputs

            tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
            validation_dataset = tokenized_datasets[dataset_config['validation_split']]
            return validation_dataset
        
        elif dataset_type == "local":
            print(f"Loading local dataset from class: {dataset_config['dataset_name']}")
            local_dataset_path = os.path.join("datasets", dataset_config['dataset_name'], f"{dataset_config['dataset_name']}.py")
            if not os.path.exists(local_dataset_path):
                raise FileNotFoundError(f"Local dataset file not found: {local_dataset_path}")
            
            spec = importlib.util.spec_from_file_location(dataset_config['dataset_name'], local_dataset_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            dataset_class = getattr(module, dataset_config['dataset_name'])
            
            validation_dataset = dataset_class(split=dataset_config['validation_split'], **dataset_config)
            
            return validation_dataset
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")
        
    def _remain_pc_lora(self):
        if 'peft_method' in self.config['model'] and self.config['model']['peft_method'] == 'PCLORA':
            self.model.set_all_lambda_w(float(self.config['model']['lambda']))
            self.model.set_only_lora(True)
            self.model.del_weighted_orig_and_optimizer()
    def _print_lora_parameter_ratio(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        lora_params = sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n)
        base_params = total_params - lora_params

        total_ratio = lora_params / total_params * 100
        base_ratio = lora_params / base_params * 100

        print(f"Total parameters (base + LoRA): {total_params:,}")
        print(f"Base model parameters: {base_params:,}")
        print(f"LoRA parameters: {lora_params:,}")
        print(f"LoRA ratio vs total: {total_ratio:.4f}%")
        print(f"LoRA ratio vs base model: {base_ratio:.4f}%")

    def _load_model(self):
        pretrained_model_name = self.config['model']['pretrained_model_name']
        model_auto_config = AutoConfig.from_pretrained(pretrained_model_name)
        if self.tokenizer.pad_token_id is None:
            model_auto_config.pad_token_id = self.tokenizer.eos_token_id
        else:
            model_auto_config.pad_token_id = self.tokenizer.pad_token_id
            
        base_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, config=model_auto_config)
        

        model = PeftModel.from_pretrained(base_model, self.model_path)
        model.print_trainable_parameters()
        
        return model.to(self.device)

    def _compute_metrics(self, eval_pred):
        metric = load_metric("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def _setup_trainer(self):
        training_args = TrainingArguments(
            output_dir="./results_test",
            per_device_eval_batch_size=self.config['training']['batch_size'],
            do_train=False,
            do_eval=True,
            report_to="none",
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=self.validation_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
        return trainer

    def run(self):
        print("Starting evaluation...")
        metrics = self.trainer.evaluate()
        print("\nEvaluation Metrics:")
        print(metrics)
        
        # Save the evaluation results to a JSON file
        output_dir = self.trainer.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "results.json")
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\nEvaluation metrics saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model using a YAML config file and model path.')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the YAML configuration file used for training.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model directory containing adapter weights.')
    args = parser.parse_args()
    
    if not os.path.exists(args.config_path):
        print(f"Error: The configuration file path '{args.config_path}' does not exist.")
    elif not os.path.exists(args.model_path):
        print(f"Error: The model path '{args.model_path}' does not exist.")
    else:
        runner = EvaluationRunner(args.config_path, args.model_path)
        runner.run()