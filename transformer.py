import torch.nn as nn

class MotionTransformer(nn.Module):
    def __init__(self, input_dim=75, d_model=256, nhead=8, layers=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder, layers)
        self.out = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.out(x)
