import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

from vpal import vpal


# Reproducibility and default dtype
start = time.time()
torch.set_default_dtype(torch.float32)
np.random.seed(0)

# Load ground-truth image and create a noisy observation b
xtrue = Image.open("cameraman.tif").convert("L")
xtrue = np.array(xtrue, dtype=np.float32)   
m, n = xtrue.shape 

noise = 10 * np.random.randn(m, n).astype(np.float32)
b = xtrue + noise

# Define the forward operator A. Here it is identity (denoising setting)
class IdentityA(torch.nn.Module):
    def __call__(self, x): return x
    #def T(self, x): return x

# Device selection and batching
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
batch_size = 1 # Change this to >1 for batch processing

# Prepare batch: if batch_size > 1, stack images
if batch_size == 1:
    b_tensor = torch.tensor(b, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
else:
    b_stack = np.stack([b for _ in range(batch_size)], axis=0)  # (B,H,W)
    b_tensor = torch.tensor(b_stack, device=device, dtype=torch.float32).unsqueeze(1)  # (B,1,H,W)


# Instantiate the VPAL solver
vpal_solver = vpal(A=IdentityA(), D='finite difference', mu=5.0, step_size='linearized', display='iter',lambda_=1.0)


# Run the solver
x_recon, _ = vpal_solver(b_tensor, return_info=True)
x_recon_np = x_recon.detach().cpu().numpy()

# Post-process the output shape for visualization
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
    # --- Single image visualization ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(xtrue, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(b, cmap='gray')
    plt.title("Noisy")
    err1 = np.linalg.norm(b - xtrue) / np.linalg.norm(xtrue)
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
    # --- Batched visualization ---
    batch_size = x_recon_np.size // (m * n)

    errs1 = np.linalg.norm(b_stack - xtrue, axis=(1,2)) / np.linalg.norm(xtrue)
    errs2 = np.linalg.norm(x_rec_img - xtrue, axis=(1,2)) / np.linalg.norm(xtrue)

    mean_err1 = errs1.mean()
    mean_err2 = errs2.mean()
    print(f"Average rel. error (noisy): {mean_err1:.4f}")
    print(f"Average rel. error (recon): {mean_err2:.4f}")

    for i in range(batch_size):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(xtrue, cmap='gray')
        plt.title(f"Original [{i}]")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(b_stack[i], cmap='gray')
        plt.title(f"Noisy [{i}]")
        plt.xlabel(f"rel. error: {errs1[i]:.4f}")
        plt.axis("on")

        plt.subplot(1, 3, 3)
        plt.imshow(x_rec_img[i], cmap='gray')
        plt.title(f"Recon [{i}]")
        plt.xlabel(f"rel. error: {errs2[i]:.4f}")
        plt.axis("on")

        plt.suptitle(f"VPAL Denoising Demo (Batch, sample {i})", fontsize=14)
        plt.tight_layout()
        plt.show()
else:
    raise ValueError(f"Unexpected x_recon shape: {x_recon_np.shape}")