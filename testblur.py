import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import torch.nn.functional as F
from vpal import vpal

# Reproducibility and default dtype
start = time.time()
torch.set_default_dtype(torch.float32)
np.random.seed(0)

# Load ground-truth image and create a noisy observation b
xtrue = Image.open("cameraman.tif").convert("L")
xtrue = np.array(xtrue, dtype=np.float32)
m, n = xtrue.shape

# Create blur and noisy batch
psf = np.ones((9, 9), dtype=np.float32)
psf = psf / psf.sum()
batch_size = 1  # Change this to >1 for batch processing

# Function to apply blur using convolution in PyTorch
def fast_blur(x, psf, device='cpu'):
    if x.ndim == 2:
        x_torch = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif x.ndim == 3:
        x_torch = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(1)  # (B,1,H,W)
    else:
        raise ValueError("Input x must be 2D or 3D numpy array.")
    kernel = torch.tensor(psf, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    pad_h = psf.shape[0] // 2
    pad_w = psf.shape[1] // 2
    x_torch = torch.nn.functional.pad(x_torch, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    y = torch.nn.functional.conv2d(x_torch, kernel, padding=0)
    if x.ndim == 2:
        return y.squeeze().cpu().numpy()
    else:
        return y.squeeze(1).cpu().numpy()  # (B, H, W)

# Function to create the blurring operator (A)  
class BlurA(torch.nn.Module):
    def __init__(self, psf, img_shape, device=None):
        super().__init__()
        self.m, self.n = img_shape
        psf = torch.tensor(psf, dtype=torch.float32)
        psf = psf / psf.sum()
        self.ph, self.pw = psf.shape
        self.kernel = psf.view(1, 1, self.ph, self.pw)
        self.kernel_T = torch.flip(psf, dims=[0, 1]).view(1, 1, self.ph, self.pw)
        if device:
            self.kernel = self.kernel.to(device)
            self.kernel_T = self.kernel_T.to(device)

    def __call__(self, x):
        # Accepts flat (m*n,) or (1,1,m,n) tensor
        if x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1 and x.shape[1] == self.m * self.n):
            img = x.view(1, 1, self.m, self.n)
        elif x.ndim == 4:
            img = x
        else:
            raise ValueError(f"Input shape {x.shape} not supported.")
        img = img.to(self.kernel.device, dtype=self.kernel.dtype)
        pad_h = self.ph - 1
        pad_w = self.pw - 1
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        img_padded = F.pad(img, padding, mode='reflect')
        y = F.conv2d(img_padded, self.kernel, padding=0)
        return y

    def T(self, x):
        # Accepts flat (m*n,) or (1,1,m,n) tensor
        if x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1 and x.shape[1] == self.m * self.n):
            img = x.view(1, 1, self.m, self.n)
        elif x.ndim == 4:
            img = x
        else:
            raise ValueError(f"Input shape {x.shape} not supported.")
        img = img.to(self.kernel_T.device, dtype=self.kernel_T.dtype)
        pad_h = self.ph - 1
        pad_w = self.pw - 1
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        img_padded = F.pad(img, padding, mode='reflect')
        y = F.conv2d(img_padded, self.kernel_T, padding=0)
        return y

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Prepare batch: if batch_size > 1, stack images
if batch_size == 1:
    xtrue_batch = xtrue[None, ...]
else:
    xtrue_batch = np.stack([xtrue for _ in range(batch_size)], axis=0)  # (B,H,W)

# Apply blur + noise to create observed images
b_batch = []
for i in range(batch_size):
    b_blur = fast_blur(xtrue_batch[i], psf, device=device)
    noise = 10 * np.random.randn(m, n).astype(np.float32)
    b_noisy = b_blur + noise
    b_batch.append(b_noisy)
b_batch = np.stack(b_batch, axis=0)  # (B, H, W)
b_tensor = torch.tensor(b_batch, device=device, dtype=torch.float32).unsqueeze(1)  # (B,1,H,W)


# Create blur operator instance for VPAL
A = BlurA(psf, (m, n), device=device)


# Initialize VPAL solver
vpal_solver = vpal(A=A, D='finite difference', mu=5.0, step_size='linearized', display='iter',lambda_=1)


# Run the solver
x_recon, _ = vpal_solver(b_tensor, return_info=True)
x_recon_np = x_recon.detach().cpu().numpy()

# Reshape output to image form
if x_recon_np.size == m * n:
    # Single image
    x_rec_img = x_recon_np.reshape(m, n)
elif x_recon_np.size % (m * n) == 0:
    batch_size = x_recon_np.size // (m * n)
    x_rec_img = x_recon_np.reshape(batch_size, m, n)
else:
    raise ValueError(f"Unexpected x_recon shape: {x_recon_np.shape}")

print(f"Running time:{time.time() - start:.3f} s")

# Visualize results
if x_recon_np.size == m * n:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(xtrue, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(b_batch[0], cmap='gray')
    plt.title("Noisy")
    err1 = np.linalg.norm(b_batch[0] - xtrue) / np.linalg.norm(xtrue)
    plt.xlabel(f"rel. error: {err1:.4f}")
    plt.axis("on")

    plt.subplot(1, 3, 3)
    plt.imshow(x_rec_img, cmap='gray')
    plt.title("VPAL Reconstruction")
    err2 = np.linalg.norm(x_rec_img - xtrue) / np.linalg.norm(xtrue)
    plt.xlabel(f"rel. error: {err2:.4f}")
    plt.axis("on")

    plt.suptitle("VPAL Denoising Demo (PyTorch)", fontsize=14)
    plt.tight_layout()
    plt.show()

elif x_recon_np.size % (m * n) == 0:
    batch_size = x_recon_np.size // (m * n)

    # 1) Compute rel-error for each sample
    errs_noisy = np.linalg.norm(b_batch - xtrue, axis=(1,2)) / np.linalg.norm(xtrue)
    errs_recon = np.linalg.norm(x_rec_img  - xtrue, axis=(1,2)) / np.linalg.norm(xtrue)

    # 2) Print batch averages
    print(f"Average rel. error (noisy): {errs_noisy.mean():.4f}")
    print(f"Average rel. error (recon): {errs_recon.mean():.4f}")

    # 3) Show all images in the batch
    # Show each image in the batch one at a time
    for i in range(batch_size):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(xtrue, cmap='gray')
        plt.title(f"Original [{i}]")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(b_batch[i], cmap='gray')
        plt.title(f"Noisy [{i}]")
        plt.xlabel(f"rel. error: {errs_noisy[i]:.4f}")
        plt.axis("on")

        plt.subplot(1, 3, 3)
        plt.imshow(x_rec_img[i], cmap='gray')
        plt.title(f"VPAL Reconstruction [{i}]")
        plt.xlabel(f"rel. error: {errs_recon[i]:.4f}")
        plt.axis("on")

        plt.suptitle(f"VPAL Denoising Demo (Batch, sample {i})", fontsize=14)
        plt.tight_layout()
        plt.show()
else:
    raise ValueError(f"Unexpected x_recon shape: {x_recon_np.shape}")