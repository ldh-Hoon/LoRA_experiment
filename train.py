import os
import sys
import yaml
import torch
import argparse
from datasets import load_dataset
from evaluate import load as load_metric

import numpy as np
import importlib.util

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, PCLoraConfig
from trainer.PCLoRA_trainer import PCTrainer

class ExperimentRunner:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.task = self.config['task']
        self.device = self._get_device()
        
        self.tokenizer = self._load_tokenizer()
        self.train_dataset, self.validation_dataset = self._load_and_preprocess_dataset()
        self.model = self._load_model()
        self.trainer = self._setup_trainer()

    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_device(self):
        device = self.config['training']['device']
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

        # Explicitly set the padding token
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
                # Ensure padding is correctly applied
                tokenized_inputs = self.tokenizer(
                    texts, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=512
                )
                tokenized_inputs['labels'] = labels
                return tokenized_inputs

            tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
            train_dataset = tokenized_datasets[dataset_config['train_split']]
            validation_dataset = tokenized_datasets[dataset_config['validation_split']]
            return train_dataset, validation_dataset

        elif dataset_type == "local":
            print(f"Loading local dataset from class: {dataset_config['dataset_name']}")
            local_dataset_path = os.path.join("datasets", dataset_config['dataset_name'], f"{dataset_config['dataset_name']}.py")
            if not os.path.exists(local_dataset_path):
                raise FileNotFoundError(f"Local dataset file not found: {local_dataset_path}")
            
            spec = importlib.util.spec_from_file_location(dataset_config['dataset_name'], local_dataset_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            dataset_class = getattr(module, dataset_config['dataset_name'])
            
            train_dataset = dataset_class(split=dataset_config['train_split'], **dataset_config)
            validation_dataset = dataset_class(split=dataset_config['validation_split'], **dataset_config)
            
            return train_dataset, validation_dataset
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    def _load_model(self):
        model_config = self.config['model']
        model_type = model_config.get('model_type')
        pretrained_model_name = model_config['pretrained_model_name']
        
        # PCLoRA 사용 여부를 확인하여 output_hidden_states를 조건부로 설정
        is_pclora = 'peft_method' in model_config and model_config['peft_method'] == 'PCLORA'
        
        if model_type == "hub":
            unique_labels = set(self.train_dataset['labels'])
            num_labels = len(unique_labels)
            
            model_auto_config_params = {
                "pretrained_model_name_or_path": pretrained_model_name,
                "num_labels": num_labels,
            }
            if is_pclora:
                model_auto_config_params["output_hidden_states"] = True

            model_auto_config = AutoConfig.from_pretrained(**model_auto_config_params)
            
            if self.tokenizer.pad_token_id is None:
                model_auto_config.pad_token_id = self.tokenizer.eos_token_id
            else:
                model_auto_config.pad_token_id = self.tokenizer.pad_token_id

            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, config=model_auto_config)
            
        elif model_type == "local":
            print(f"Loading local model from class: {model_config['model_name']}")
            local_model_path = os.path.join("models", model_config['model_name'], f"{model_config['model_name']}.py")
            if not os.path.exists(local_model_path):
                raise FileNotFoundError(f"Local model file not found: {local_model_path}")
            
            spec = importlib.util.spec_from_file_location(model_config['model_name'], local_model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model_class = getattr(module, model_config['model_name'])
            
            num_labels = 2 
            
            model_params = {
                "model_name": model_config['model_name'],
                "num_labels": num_labels,
                "pretrained_model_path": model_config.get('pretrained_model_path')
            }
            if is_pclora:
                model_params["output_hidden_states"] = True
            
            model = model_class.from_pretrained(**model_params)
            
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
            
        if 'peft_method' in model_config and model_config['peft_method'] == 'LORA':
            lora_config = LoraConfig(**model_config['lora_config'])
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        elif 'peft_method' in model_config and model_config['peft_method'] == 'PCLORA':
            lora_config = PCLoraConfig(**model_config['lora_config'])
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
             
        
        return model.to(self.device)

    def _compute_metrics(self, eval_pred):
        metric = load_metric("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def _setup_trainer(self):
        training_args = TrainingArguments(
            output_dir=f"./results/{self.task}/{self.config['model']['peft_method']}/{self.config['model']['lora_config']['r']}/{self.config['model']['pretrained_model_name'].replace('/', '-')}",
            num_train_epochs=self.config['training']['epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            learning_rate=float(self.config['training']['learning_rate']),
            weight_decay=self.config['training']['weight_decay'],
            eval_strategy="epoch",
            logging_dir="./logs",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="none",
            # Add gradient accumulation steps here
            gradient_accumulation_steps=self.config['training'].get('gradient_accumulation_steps', 1),
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        if 'peft_method' in self.config['model'] and self.config['model']['peft_method'] == 'PCLORA':
            # PC-LoRA를 위한 Trainer
            trainer = PCTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.validation_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics,
                alpha=self.config['training'].get('alpha', 0.2) # alpha 값 전달
            )
        else:
            # 기본 Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.validation_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics,
            )
        
        return trainer

    def run(self):
        print("Starting training...")
        self.trainer.train()
        
        print("\nFinal evaluation on the validation set:")
        final_metrics = self.trainer.evaluate()
        print(final_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model using a YAML config file.')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()
    
    if not os.path.exists(args.config_path):
        print(f"Error: The configuration file path '{args.config_path}' does not exist.")
    else:
        runner = ExperimentRunner(args.config_path)
        runner.run()