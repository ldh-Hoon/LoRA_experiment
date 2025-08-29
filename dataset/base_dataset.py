import os
import importlib.util
from datasets import load_dataset, DatasetDict

# local dataset은 dataset class 필요
# huggingface load는 편함
def load_downstream_dataset(config: dict):
    """
    Load a dataset from a local Python file or Hugging Face Hub based on the config.

    Args:
        config (dict): A dictionary containing dataset configuration.
                       Expected keys: 'name', 'train_split', 'validation_split'.

    Returns:
        tuple: A tuple containing the train_dataset and eval_dataset.
    """
    dataset_name = config['name']
    dataset_dir = "dataset"
    root_dir = dataset_name + "_PATH"
    # 1. Check for a local Python file first.
    local_py_file_path = os.path.join(dataset_dir, f"{dataset_name}.py")
    if os.path.exists(local_py_file_path):
        print(f"Loading local dataset from Python class: {local_py_file_path}")
        try:
            # Dynamically import the class from the local file.
            spec = importlib.util.spec_from_file_location(dataset_name, local_py_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            dataset_class = getattr(module, dataset_name)
            
            # Instantiate your custom class with required arguments.
            # You might need to adjust the root_dir path to match your project structure.
            train_data = dataset_class(root_dir=root_dir, split=config['train_split'])
            eval_data = dataset_class(root_dir=root_dir, split=config['validation_split'])
            
            print(f"Successfully loaded local dataset from class '{dataset_name}'.")
            return train_data, eval_data
        except Exception as e:
            print(f"Error loading local dataset from Python class: {e}")
            raise e
    
    # 2. If not a local Python file, try to load from Hugging Face Hub.
    else:
        print(f"Local Python file not found. Loading from Hugging Face Hub: {dataset_name}")
        try:
            dataset_dict = load_dataset(dataset_name)
            
            train_dataset = dataset_dict[config['train_split']]
            eval_dataset = dataset_dict[config['validation_split']]
            
            print("Successfully loaded dataset from Hugging Face Hub.")
            return train_dataset, eval_dataset
        except Exception as e:
            print(f"Error loading dataset from Hugging Face Hub: {e}")
            raise e

# Example Usage (in your main training script):
if __name__ == '__main__':
    # Example 1: Loading a dataset from Hugging Face Hub
    hub_config = {
        'name': 'google/boolq',
        'train_split': 'train',
        'validation_split': 'validation'
    }
    print("--- Testing Hugging Face Hub dataset loading ---")
    try:
        train_data_hub, eval_data_hub = load_downstream_dataset(hub_config)
        print(f"Train dataset size: {len(train_data_hub)}")
        print(f"Evaluation dataset size: {len(eval_data_hub)}")
    except Exception as e:
        print(f"Failed to load Hub dataset: {e}")

    print("\n" + "="*50 + "\n")
    