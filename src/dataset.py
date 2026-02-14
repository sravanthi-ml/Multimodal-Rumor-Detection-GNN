import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTImageProcessor
from PIL import Image
import os
import pandas as pd


class RumorDataset(Dataset):
    def __init__(self, split="train", data_path="data/sample.csv"):
        """
        Expected CSV format:
        text, image_path, meta1, meta2, ..., label
        """

        self.data = pd.read_csv(data_path)
        self.split = split

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.image_processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        # -----------------------------
        # Text Processing
        # -----------------------------
        encoding = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # -----------------------------
        # Image Processing
        # -----------------------------
        image = Image.open(row["image_path"]).convert("RGB")
        image_inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        # -----------------------------
        # Metadata (example: first 2 metadata columns)
        # -----------------------------
        metadata = torch.tensor(
            row[["meta1", "meta2"]].values.astype("float32")
        )

        # -----------------------------
        # Label
        # -----------------------------
        label = torch.tensor(row["label"]).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "metadata": metadata,
            "labels": label
        }

