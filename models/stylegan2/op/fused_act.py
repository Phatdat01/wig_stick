import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)

# Load CUDA extensions if CUDA is available, otherwise use CPU implementation
if torch.cuda.is_available():
    fused = load(
        "fused",
        sources=[
            os.path.join(module_path, "fused_bias_act.cpp"),
            os.path.join(module_path, "fused_bias_act_kernel.cu"),
        ],
    )
else:
    # Use CPU fallback for LeakyReLU
    print("CUDA is not available. Falling back to CPU implementation.")

class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        if torch.cuda.is_available():
            grad_input = fused.fused_bias_act(
                grad_output, empty, out, 3, 1, negative_slope, scale
            )
        else:
            # CPU implementation of LeakyReLU backward
            grad_input = F.leaky_relu(grad_output, negative_slope) * scale

        dim = [0]
        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        (out,) = ctx.saved_tensors
        if torch.cuda.is_available():
            gradgrad_out = fused.fused_bias_act(
                gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
            )
        else:
            # CPU fallback for backward pass
            gradgrad_out = F.leaky_relu(gradgrad_input, ctx.negative_slope)

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        
        if torch.cuda.is_available():
            out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        else:
            # CPU implementation of LeakyReLU forward
            out = F.leaky_relu(input + bias.view(1, bias.shape[0], *[1] * (input.ndimension() - bias.ndimension() - 1)), negative_slope=negative_slope)

        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors

        if torch.cuda.is_available():
            grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
                grad_output, out, ctx.negative_slope, ctx.scale
            )
        else:
            # CPU fallback for backward pass
            grad_input = F.leaky_relu(grad_output, ctx.negative_slope)
            grad_bias = grad_input.sum([0, 2, 3])

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    if input.device.type == "cpu":
        # CPU implementation
        rest_dim = [1] * (input.ndimension() - bias.ndimension() - 1)
        return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope) * scale
    else:
        # CUDA implementation
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
