# Realistic-Character-Motion-using-GAN-Transformer
This repository presents a deep learning–based system for improving character animation realism using Generative Adversarial Networks (GANs) and Transformer architectures. The project focuses on refining raw or synthetic motion data into smooth, natural, and physically plausible character movements.
# Realistic Character Motion using GAN & Transformer

This repository improves character animation realism using:
- GAN for natural motion generation
- Transformer for long-term temporal consistency

## Use Cases
- Game character animation
- Motion capture cleanup
- NPC locomotion
- Cinematics

## Install
pip install -r requirements.txt

## Train GAN
python train/train_gan.py

## Train Transformer
python train/train_transformer.py

## Inference
python inference/refine_motion.py
