import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel


# -----------------------------------
# Multimodal Rumor Detection Model
# -----------------------------------
class MultimodalRumorModel(nn.Module):
    def __init__(self, metadata_dim=10, hidden_dim=256, num_classes=4):
        super(MultimodalRumorModel, self).__init__()

        # -----------------------------------
        # Text Encoder (BERT)
        # -----------------------------------
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Linear(768, hidden_dim)

        # -----------------------------------
        # Image Encoder (ViT)
        # -----------------------------------
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.image_proj = nn.Linear(768, hidden_dim)

        # -----------------------------------
        # Metadata Encoder (MLP)
        # -----------------------------------
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # -----------------------------------
        # Fusion Layer
        # -----------------------------------
        self.fusion_layer = nn.Linear(hidden_dim * 3, hidden_dim)

        # -----------------------------------
        # Classification Head
        # -----------------------------------
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask, pixel_values, metadata):

        # Text Features
        text_outputs = self.text_encoder(input_ids=input_ids,
                                         attention_mask=attention_mask)
        text_feat = self.text_proj(text_outputs.pooler_output)

        # Image Features
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_feat = self.image_proj(image_outputs.pooler_output)

        # Metadata Features
        meta_feat = self.metadata_mlp(metadata)

        # Concatenate multimodal features
        fused = torch.cat([text_feat, image_feat, meta_feat], dim=1)
        fused = F.relu(self.fusion_layer(fused))

        # Classification
        logits = self.classifier(fused)

        return logits

