import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CommonlitTrainDatasetV1(Dataset):
    def __init__(self, cfg: DictConfig, df: pd.DataFrame):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.experiment.model_name)
        self.max_length = cfg.experiment.max_length

        self.texts = (
            df["text"]
            + +self.tokenizer.sep_token
            + df["prompt_title"]
            + self.tokenizer.sep_token
            + df["prompt_question"]
            + self.tokenizer.sep_token
            + df["prompt_text"]
        ).tolist()
        self.targets = df[["content", "wording"]].to_numpy()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = self.prepare_input(max_length=self.max_length, tokenizer=self.tokenizer, text=self.texts[item])
        targets = torch.tensor(self.targets[item], dtype=torch.float16)
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "targets": targets,
        }

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
