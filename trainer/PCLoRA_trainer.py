import torch
from transformers import Trainer
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from peft import PeftModel, PCLoraLayer

class PCTrainer(Trainer):
    def __init__(self, *args, alpha: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        
        if not isinstance(self.model, PeftModel):
            raise TypeError("PCTrainer는 PeftModel 인스턴스를 필요로 합니다.")
        
        # 1. 교사 모델(Teacher) 설정: 베이스 모델을 가져와서 동결
        self.teacher_model = self.model.get_base_model()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # 2. PC-LoRA 레이어 인덱스 찾기
        self.lora_layers_indices = self._get_lora_layer_indices()
        
    def _get_lora_layer_indices(self):
        """모델에서 LoRA가 적용된 레이어의 인덱스를 찾습니다."""
        indices = []
        # PeftModel의 베이스 모델에서 layers 모듈을 직접 순회
        # Llama의 경우 `model.model.layers`가 레이어 리스트를 포함합니다.
        model_layers = self.model.base_model.model.model.layers
        
        for i, layer in enumerate(model_layers):
            lora_found_in_layer = any(isinstance(module, PCLoraLayer) for module in layer.modules())
            
            if lora_found_in_layer:
                # `hidden_states`의 인덱스는 각 레이어의 출력에 해당하므로,
                # layer 인덱스 `i`를 저장합니다.
                indices.append(i + 1) # `hidden_states`는 임베딩 출력을 포함하므로 +1
                print(f"Found LoRA-applied layer at model layer index {i}, using hidden_states index {i + 1}")
        
        # 마지막 레이어의 인덱스를 추가하여 최종 레이어의 hidden_state도 포함
        indices.append(len(model_layers))
        
        if not indices:
            raise ValueError("모델에서 LoRA 레이어를 찾을 수 없습니다. LoRA 설정(config)을 확인해주세요.")

        return indices

    def training_step(self, model, inputs, num_items_in_batch):
        # === 수정된 total_steps 계산: optimizer(업데이트) 스텝 기준 ===
        if self.args.max_steps > 0:
            total_update_steps = self.args.max_steps
        else:
            micro_steps_per_epoch = len(self.get_train_dataloader())
            update_steps_per_epoch = math.ceil(micro_steps_per_epoch / self.args.gradient_accumulation_steps)
            total_update_steps = int(self.args.num_train_epochs * update_steps_per_epoch)

        current_update_step = self.state.global_step  # 이 값은 '업데이트 스텝'에서만 증가합니다

        q = int(total_update_steps * 0.8)
        if current_update_step < q:
            lambda_value = 1.0 - (current_update_step / q)
        else:
            lambda_value = 0.0

        for _, module in model.named_modules():
            if hasattr(module, 'set_lambda_w'):
                module.set_lambda_w(lambda_value)

        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 학습 시 hidden_states 필요
        output_hidden_states = model.training
        labels = inputs.pop("labels")

        # student forward
        student_outputs = model(**inputs, output_hidden_states=output_hidden_states)

        # teacher forward (no grad)
        if model.training:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs, output_hidden_states=True)

            kd_losses = []
            for idx in self.lora_layers_indices:
                if idx < len(student_outputs.hidden_states) and idx < len(teacher_outputs.hidden_states):
                    s_feat = student_outputs.hidden_states[idx]
                    t_feat = teacher_outputs.hidden_states[idx]
                    if s_feat.shape == t_feat.shape:
                        kd_losses.append(F.mse_loss(s_feat, t_feat))

            LfeatKD = torch.stack(kd_losses).mean() if kd_losses else torch.tensor(0.0, device=student_outputs.logits.device)
        else:
            LfeatKD = torch.tensor(0.0, device=student_outputs.logits.device)

        # Task loss
        logits = student_outputs.logits
        Ltask = CrossEntropyLoss()(logits.view(-1, model.num_labels), labels.view(-1))

        total_loss = self.alpha * Ltask + (1.0 - self.alpha) * LfeatKD

        return (total_loss, student_outputs) if return_outputs else total_loss