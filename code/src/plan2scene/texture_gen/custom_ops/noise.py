# Code adapted from https://github.com/henzler/neuraltexture/blob/master/code/custom_ops/noise/noise.py

from torch import nn
from torch.autograd import Function
import plan2scene.texture_gen.utils.neural_texture_helper as utils_nt
import torch
import numpy as np

try:
    import noise_cuda
except ImportError:
    class NoiseCUDA:
        @staticmethod
        def forward(position, seed):
            # Debug print to inspect tensor shape
            print(f"Forward - Position shape: {position.shape}, Seed shape: {seed.shape}")
            # Ensure the returned tensor has the correct shape
            return torch.zeros(position.size(0), position.size(1))

        @staticmethod
        def backward(position, seed):
            # Debug print to inspect tensor shape
            print(f"Backward - Position shape: {position.shape}, Seed shape: {seed.shape}")
            # Ensure the returned tensor has the correct shape
            return torch.zeros(position.size(0), position.size(1))

    noise_cuda = NoiseCUDA()

class NoiseFunction(Function):
    @staticmethod
    def forward(ctx, position, seed):
        ctx.save_for_backward(position, seed)
        noise = noise_cuda.forward(position, seed)
        return noise

    @staticmethod
    def backward(ctx, grad_noise):
        position, seed = ctx.saved_tensors
        d_position_bilinear = noise_cuda.backward(position, seed)

        # Debug print to inspect tensor shapes during backward pass
        print(f"Backward - grad_noise shape: {grad_noise.shape}")
        print(f"Backward - d_position_bilinear shape: {d_position_bilinear.shape}")

        # Expand grad_noise to match the shape of d_position_bilinear
        grad_noise_expanded = grad_noise.unsqueeze(1)
        print(f"Backward - grad_noise_expanded shape: {grad_noise_expanded.shape}")

        return grad_noise_expanded * d_position_bilinear, None

class Noise(nn.Module):
    def __init__(self):
        super(Noise, self).__init__()

    def forward(self, position, seed):
        noise = NoiseFunction.apply(position.contiguous(), seed.contiguous())
        return noise
