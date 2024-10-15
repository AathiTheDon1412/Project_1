import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class DreamFusion(nn.Module):
    def _init_(self):
        super(DreamFusion, self)._init_()
        self.text_encoder = TextEncoder()
        self.shape_decoder = ShapeDecoder()

    def forward(self, text_description):
        text_embedding = self.text_encoder(text_description)
        shape = self.shape_decoder(text_embedding)
        return shape

    def generate_3d_model(self, text_description):
        text_embedding = self.text_encoder(text_description)
        shape = self.shape_decoder(text_embedding)
        return shape.detach().numpy()

class TextEncoder(nn.Module):
    def _init_(self):
        super(TextEncoder, self)._init_()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.classification_module = nn.Linear(768, 128)

    def forward(self, text_description):
        input_ids = torch.tensor([self.bert_model.encode(text_description)])
        attention_mask = torch.tensor([[1]*len(input_ids)])
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        text_embedding = self.classification_module(pooled_output)
        return text_embedding

class ShapeDecoder(nn.Module):
    def _init_(self):
        super(ShapeDecoder, self)._init_()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)

    def forward(self, text_embedding):
        x = torch.relu(self.fc1(text_embedding))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        shape = x.view(-1, 3, 32, 32)
        return shape