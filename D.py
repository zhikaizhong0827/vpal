import torch
import torch.nn.functional as F

class FiniteDifference(torch.nn.Module):
    """
    Fully tensor-only 2D forward-difference D and its adjoint D^T.
    Input/output are (B,1,H,W) or (B,2,H-1,W-1) 4D tensors.
    No view/reshape anywhere.
    """
    def __init__(self):
        super().__init__()
        # Conv2d kernels for horizontal and vertical differences, stacked together
        # out_ch=2, in_ch=1, kH=2, kW=2
        k = torch.tensor([
          [[[-1., 1.],
            [ 0., 0.]]],
          [[[-1., 0.],
            [ 1., 0.]]]
        ], dtype=torch.get_default_dtype())
        self.register_buffer('kernel', k)

    def forward(self, x):
        # x: (B,1,H,W) -> returns (B,2,H,W)
        x = F.pad(x, (0, 1, 0, 1), mode='replicate')
        return F.conv2d(x, self.kernel)

    def adjoint(self, y):
        # y: (B,2,H,W) -> returns (B,1,H,W)
        x = F.conv_transpose2d(y, self.kernel)
        # Crop to original size
        return x[..., :y.shape[2], :y.shape[3]]

    # If vpal needs a .T() method:
    def T(self):
        return self.adjoint
    

class Laplace(torch.nn.Module):
    """
    Fully tensor-only 2D Laplacian operator and its adjoint.
    Input/output are (B,1,H,W) or (B,1,H-2,W-2) 4D tensors.
    No view/reshape anywhere.
    """
    def __init__(self):
        super().__init__()
        # 2D Laplacian kernel (center -4, neighbors +1)
        # out_ch=1, in_ch=1, kH=3, kW=3
        k = torch.tensor([[[[0., 1., 0.],
                            [1., -4., 1.],
                            [0., 1., 0.]]]], dtype=torch.get_default_dtype())
        self.register_buffer('kernel', k)

    def forward(self, x):
        # x: (B,1,H,W) -> returns (B,1,H-2,W-2)
        return F.conv2d(x, self.kernel)

    def adjoint(self, y):
        # y: (B,1,H-2,W-2) -> returns (B,1,H,W)
        return F.conv_transpose2d(y, self.kernel)

    def T(self):
        return self.adjoint