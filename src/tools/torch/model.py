import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModel


class CommonLitModelV1(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.model_config = AutoConfig.from_pretrained(cfg.experiment.model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(cfg.experiment.model, config=self.model_config)

        if cfg.experiment.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.out = nn.Linear(
            in_features=self.model_config.hidden_size,
            out_features=cfg.experiment.out_features,
        )
        self._init_weights(self.out)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        feature = outputs[0][:, 0, :]
        return feature

    def forward(self, batch):
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        output = self.out(feature)
        return output
