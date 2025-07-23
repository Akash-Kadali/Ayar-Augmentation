# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

"""Custom replacement for `torch.nn.functional.grid_sample` that
supports arbitrarily high order gradients between the input and output.
Only works on 2D images and assumes
`mode='bilinear'`, `padding_mode='zeros'`, `align_corners=False`."""

import warnings
import torch

# Enable the custom op by setting this to true.
enabled = False  # Custom op is not used anymore due to compatibility issues.

def grid_sample(input, grid):
    if _should_use_custom_op():
        return _GridSample2dForward.apply(input, grid)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

def _should_use_custom_op():
    if not enabled:
        return False
    if any(torch.__version__.startswith(x) for x in ['1.7.', '1.8.', '1.9']):
        return True
    warnings.warn(f'grid_sample_gradfix not supported on PyTorch {torch.__version__}. Falling back to torch.nn.functional.grid_sample().')
    return False

# These classes are no longer used but retained for backward compatibility.
class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        assert input.ndim == 4
        assert grid.ndim == 4
        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        warnings.warn("Backward pass using custom grid_sample is not supported in PyTorch >= 2.0. Falling back.")
        grad_input = torch.autograd.grad(outputs=_GridSample2dForward.apply(input, grid),
                                         inputs=input,
                                         grad_outputs=grad_output,
                                         retain_graph=True,
                                         allow_unused=True)[0]
        grad_grid = None  # Not supported
        return grad_input, grad_grid

class _GridSample2dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid):
        warnings.warn("Custom grid_sample backward is not supported in PyTorch >= 2.0. Using fallback.")
        grad_input = torch.autograd.grad(outputs=torch.nn.functional.grid_sample(input, grid, mode='bilinear',
                                                                                 padding_mode='zeros', align_corners=False),
                                         inputs=input,
                                         grad_outputs=grad_output,
                                         retain_graph=True,
                                         allow_unused=True)[0]
        grad_grid = None
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        return None, None, None
