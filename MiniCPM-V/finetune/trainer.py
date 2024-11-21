import torch.nn as nn
from transformers import Trainer


class CPMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        if not self.args.use_lora:
            #outputs = self.model(data = inputs, use_cache=False)
            outputs = model(data = inputs, use_cache=False)
        else:
            # with self.model._enable_peft_forward_hooks(**inputs):
            #     outputs = self.model.base_model(data = inputs, use_cache=False)
            with model._enable_peft_forward_hooks(**inputs):
                outputs = model.base_model(data = inputs, use_cache=False)
                
        if labels is not None:
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits.view(-1,
                                         self.model.config.vocab_size).contiguous()
            labels = labels.view(-1).long().contiguous()
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
