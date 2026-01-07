import torch
from models.transformer import MotionTransformer

model = MotionTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

sequence = torch.randn(8, 120, 75)
target = sequence.clone()

output = model(sequence)
loss = loss_fn(output, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Transformer training step complete.")
