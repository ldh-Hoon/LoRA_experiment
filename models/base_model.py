import os
from transformers import AutoModelForSequenceClassification, AutoConfig
import torch

# custom model은 class py 필요, model폴더 안에 custom_model폴더 만드는게 깔끔할듯
# huggingface load는 편함

def get_base_model(model_name: str, num_labels: int, model_dir: str = "models"):
    """
    Load a pre-trained model from Hugging Face Hub or a local directory.

    Args:
        model_name (str): The name or path of the model.
                          e.g., "google/bert_uncased_L-2_H-128_A-2" for a Hub model
                          e.g., "custom_bert" for a local model
        num_labels (int): The number of labels for the classification task.
        model_dir (str): The directory containing local model definitions.

    Returns:
        torch.nn.Module: The loaded model.
    """
    
    is_hub_model = "/" in model_name

    if is_hub_model:
        print(f"Loading model from Hugging Face Hub: {model_name}")
        try:
            # Use AutoModelForSequenceClassification to load the model for a classification task
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            print("Successfully loaded model from Hugging Face Hub.")
            return model
        except Exception as e:
            print(f"Error loading model from Hugging Face Hub: {e}")
            raise
    else:
        # Check for a local model in the specified directory structure:
        # models/model_name/model_name.py
        
        # 'os.path.join'을 사용하여 로컬 모델의 전체 경로를 만듭니다.
        # 이렇게 하면 운영체제에 맞는 경로 구분자(Windows의 '\', Linux의 '/')가 자동으로 사용됩니다.
        local_model_path = os.path.join(model_dir, f"{model_name}.py")
        
        if os.path.exists(local_model_path):
            print(f"Loading local model: {model_name}")
            try:
                # Dynamically import the model class from the local file
                import importlib.util
                # module_name을 'models.custom_bert'와 같은 형식으로 설정하여 모듈을 고유하게 식별합니다.
                module_name_for_import = f"{model_dir}.{model_name}"
                spec = importlib.util.spec_from_file_location(module_name_for_import, local_model_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Assume the model class is named after the model_name (e.g., 'CustomBert')
                model_class_name = model_name
                if hasattr(module, model_class_name):
                    model_class = getattr(module, model_class_name)
                    # Instantiate the model with the number of labels
                    model = model_class.from_pretrained(model_name, num_labels=num_labels)
                    print(f"Successfully loaded local model: {model_class_name}")
                    return model
                else:
                    raise AttributeError(f"Model class '{model_class_name}' not found in {local_model_path}")
            except Exception as e:
                print(f"Error loading local model: {e}")
                raise
        else:
            raise FileNotFoundError(f"Model '{model_name}' not found in Hugging Face Hub or as a local file '{local_model_path}'")

# Example Usage:
if __name__ == '__main__':
    # Example 1: Loading a model from Hugging Face Hub
    try:
        hub_model = get_base_model(model_name="google/bert_uncased_L-2_H-128_A-2", num_labels=2)
        print(hub_model)
    except (FileNotFoundError, AttributeError) as e:
        print(f"Failed to load Hub model: {e}")
    
    print("-" * 50)
    
    # Example 2: Loading a local model
    # For this to work, you would need to create a file like 'local_models/custom_bert.py'
    # with a class named 'CustomBert'.
    # e.g.,
    # # In local_models/custom_bert.py:
    # from transformers import BertForSequenceClassification
    # class CustomBert(BertForSequenceClassification):
    #     def __init__(self, config):
    #         super().__init__(config)
    #
    try:
        local_model = get_base_model(model_name="CNN_LSTM_AV", num_labels=2)
        print(local_model)
    except (FileNotFoundError, AttributeError) as e:
        print(f"Failed to load local model: {e}")