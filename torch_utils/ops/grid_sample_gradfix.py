# grid_sample_gradfix.py

import torch
import warnings

# Disable the custom op by default for PyTorch 2.6+
enabled = False

#----------------------------------------------------------------------------

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    # If enabled and supported, use custom implementation (only for PyTorch 1.7–1.9)
    if _should_use_custom_op():
        return _GridSample2dForward.apply(input, grid)
    
    # Fallback to standard PyTorch function with backward support
    return torch.nn.functional.grid_sample(
        input=input,
        grid=grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )

#----------------------------------------------------------------------------

def _should_use_custom_op():
    if not enabled:
        return False
    if any(torch.__version__.startswith(x) for x in ['1.7.', '1.8.', '1.9']):
        return True
    warnings.warn(
        f'grid_sample_gradfix not supported on PyTorch {torch.__version__}. '
        'Falling back to torch.nn.functional.grid_sample().'
    )
    return False

#----------------------------------------------------------------------------

# Optional legacy custom implementation (unused for PyTorch ≥2.0)
class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        assert input.ndim == 4
        assert grid.ndim == 4
        output = torch.nn.functional.grid_sample(
            input=input,
            grid=grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=grad_output, retain_graph=True)[0]
        grad_grid = None  # Let PyTorch autograd handle it (or skip)
        return grad_input, grad_grid

#----------------------------------------------------------------------------
