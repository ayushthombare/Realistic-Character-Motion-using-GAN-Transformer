import torch
from models.transformer import MotionTransformer

model = MotionTransformer()
model.eval()

motion = torch.randn(1, 120, 75)

with torch.no_grad():
    refined_motion = model(motion)

print("Motion refined successfully.")
