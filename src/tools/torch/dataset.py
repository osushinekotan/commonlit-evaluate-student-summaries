import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CommonlitDatasetV1(Dataset):
    def __init__(self, cfg: DictConfig, df: pd.DataFrame, is_train=True):
        self.df = df
        self.is_train = is_train
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.max_length = cfg.max_length

        self.texts = (
            df["text"]
            + self.tokenizer.sep_token
            + df["prompt_title"]
            + self.tokenizer.sep_token
            + df["prompt_question"]
            + self.tokenizer.sep_token
            + df["prompt_text"]
        ).tolist()

        if self.is_train:
            self.targets = df[cfg.target].to_numpy()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = self.prepare_input(max_length=self.max_length, tokenizer=self.tokenizer, text=self.texts[item])
        if self.is_train:
            targets = torch.tensor(self.targets[item], dtype=torch.float16)
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "targets": targets,
            }
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

    @staticmethod
    def prepare_input(max_length: int, tokenizer: AutoTokenizer, text: list[str]):
        inputs = tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs
