import torch
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother


class WeightGrowthTrainer(Trainer):
    def __init__(self, *args, weight_growth=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_growth = weight_growth
        self.label_smoother = LabelSmoother(epsilon=0)
        self.step_cnt = 0
        self.reg_loss = 0
        self.norm = 0
        self.base_loss_list = []
        self.target_dict = kwargs.get("target_dict")
        if self.target_dict is None:
            print("WARNING: No target dict provided, will regularize all weights")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.step_cnt += 1
        labels = inputs.get("input_ids")
        outputs = model(**inputs)
        # logits = outputs.logits
        base_loss = self.label_smoother(outputs, labels, shift_labels=True)
        self.base_loss_list.append(base_loss.item())
        self.base_loss = sum(self.base_loss_list) / len(self.base_loss_list)
        if self.step_cnt % self.args.gradient_accumulation_steps != 0:
            loss = base_loss


        else:
            reg_loss = torch.tensor(0.0, requires_grad=True)
            norm = torch.tensor(0.0, requires_grad=True)
            for name, param in model.named_parameters():
                # only regularize one weight per 32 weights
                condition = name in self.target_dict.keys() if self.target_dict is not None else name.endswith(".weight")
                if condition:
                    # param_for_reg = param.view(32, -1)[0, :]
                    param_for_reg = param
                    norm = norm + torch.abs(param_for_reg).sum()
                    reg_loss = reg_loss - torch.abs(param_for_reg).sum()

            if self.base_loss > 1:
                # when base loss is too large, focus on reducing the base loss
                loss = base_loss
            elif self.base_loss > 0.5:
                loss = base_loss + (self.weight_growth * reg_loss / 10)
            else:
                loss = base_loss + self.weight_growth * reg_loss

            self.base_loss_list = []
            self.reg_loss = reg_loss.item()
            self.norm = norm.item()

        self.loss = loss
        return (loss, outputs) if return_outputs else loss

    def log(self, logs):
        logs["base_loss"] = self.base_loss
        logs["reg_loss"] = self.reg_loss
        logs["norm"] = self.norm
        super().log(logs)


class QuantPreserveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_factor = None
        self.original_model_sd = None
        self.label_smoother = LabelSmoother(epsilon=0)
        self.step_cnt = 0
        self.reg_loss = 0
        self.norm = 0
        self.base_loss_list = []
        self.target_dict = kwargs.get("target_dict")

    def check(self):
        assert self.reg_factor is not None
        assert self.original_model_sd is not None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.step_cnt += 1
        labels = inputs.get("input_ids")
        outputs = model(**inputs)
        # logits = outputs.logits
        base_loss = self.label_smoother(outputs, labels, shift_labels=True)
        self.base_loss_list.append(base_loss.item())
        self.base_loss = sum(self.base_loss_list) / len(self.base_loss_list)
        if self.step_cnt % self.args.gradient_accumulation_steps != 0:
            loss = base_loss

        else:
            reg_loss = torch.tensor(0.0, requires_grad=True)
            for name, param in model.named_parameters():
                # only regularize one weight per 32 weights
                condition = name in self.target_dict.keys() if self.target_dict is not None else name.endswith(".weight")
                if condition:
                    reg_loss = reg_loss + ((param - self.original_model_sd[name].to(param.device)) ** 2).sum()

            loss = base_loss + self.reg_factor * reg_loss

            self.base_loss_list = []
            self.reg_loss = reg_loss.item()

        self.loss = loss
        return (loss, outputs) if return_outputs else loss

    def log(self, logs):
        logs["base_loss"] = self.base_loss
        logs["reg_loss"] = self.reg_loss
        super().log(logs)
