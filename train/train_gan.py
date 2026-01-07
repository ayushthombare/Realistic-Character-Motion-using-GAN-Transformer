import torch
from models.generator import MotionGenerator
from models.discriminator import MotionDiscriminator
from utils.losses import adversarial_loss

G = MotionGenerator()
D = MotionDiscriminator()

g_opt = torch.optim.Adam(G.parameters(), lr=0.0002)
d_opt = torch.optim.Adam(D.parameters(), lr=0.0002)

real = torch.randn(64, 75)
noise = torch.randn(64, 75)

# Train Discriminator
fake = G(noise).detach()
d_loss = adversarial_loss(D(real), torch.ones(64,1)) + \
         adversarial_loss(D(fake), torch.zeros(64,1))

d_opt.zero_grad()
d_loss.backward()
d_opt.step()

# Train Generator
g_loss = adversarial_loss(D(G(noise)), torch.ones(64,1))
g_opt.zero_grad()
g_loss.backward()
g_opt.step()

print("GAN training step complete.")
