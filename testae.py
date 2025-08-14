import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from vpal import vpal
from torch.func import vjp
from torch.func import jvp
from D import FiniteDifference  
import time

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 32  # size of the autoencoder's latent code (z)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256,   64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.ReLU(),
            nn.Linear(64,        256),  nn.ReLU(),
            nn.Linear(256,      28*28), nn.Sigmoid(),
            nn.Unflatten(1, (1,28,28))
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
# blur operator
psf = torch.ones((1,1,9,9), device=device) / 81.0
pad = 9//2
def blur(x):
    k = psf.to(x.device)
    xpad = F.pad(x, (pad,pad,pad,pad), mode='reflect')
    return F.conv2d(xpad, k)

# A(z) = blur(decoder(z))
class CompositeA(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder.to(device)
    def forward(self, z):
        # Ensure z is (B, latent_dim)
        if z.ndim > 2:
            z = z.view(z.size(0), -1)
        x_img = self.decoder(z)           # (1,1,28,28)
        y_img = blur(x_img)               # (1,1,28,28)
        return y_img.view(z.size(0), -1)  # (B, 784)

    
# Data: MNIST test set
transform = transforms.ToTensor()
test_ds   = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 10
indices = list(range(batch_size))
imgs = [test_ds[i][0] for i in indices]
x = torch.stack(imgs, dim=0).to(device)

# Load a pre-trained autoencoder
ae = Autoencoder(latent_dim).to(device)
ae.load_state_dict(torch.load('autoencoder_mnist.pth', map_location=device))
ae.eval()

# (Using blurry input to produce a starting guess in latent space)
with torch.no_grad():
    y = blur(x)                       # (1,1,28,28)
    z0 = ae.encoder(y)                # (1,latent_dim)

# D(z) = D_img(decoder(z)), and its adjoint D^T is implemented via VJP    
class DecodedD(nn.Module):

    def __init__(self, decoder: nn.Module, D_img: nn.Module):
        super().__init__()
        self.decoder = decoder
        self.D_img   = D_img
        self._last_z = None  

        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.decoder.eval()

    def forward(self, z):
        # z: [B, latent_dim]; decoder(z) -> [B, C, H, W]
        self._last_z = z
        x = self.decoder(z)
        return self.D_img(x)  

    def T(self):
        parent = self
        DimgT = self.D_img.T() 

        class _Adj(nn.Module):
            def forward(self, y):
                if parent._last_z is None:
                    raise RuntimeError("DecodedD: T() called before forward().")
                def F(z):
                    return parent.decoder(z)
                u = DimgT(y)
                _, vjp_fn = vjp(F, parent._last_z)
                (JT_u,) = vjp_fn(u) 
                return JT_u

            def to(self, device): 
                return self

        return _Adj()

    def to(self, device):
        self.decoder.to(device)
        try:
            self.D_img.to(device)
        except AttributeError:
            pass
        return self

# Optional: a pure image-space blur Module (not used directly below)
class BlurA(nn.Module):
    def __init__(self, psf, pad): super().__init__(); self.register_buffer('k', psf); self.pad=pad
    def forward(self, x):
        if x.ndim==2: B,N=x.shape; H=W=int(N**0.5); x=x.view(B,1,H,W)
        xpad = F.pad(x,(self.pad,)*4, mode='reflect')
        return F.conv2d(xpad, self.k)

# Build the latent regularizer and forward operator for VPAL
# - D_img: total variation via finite differences in image space
# - D    : lifted to latent via DecodedD (D(z) = D_img(decoder(z)))
# - A    : CompositeA (A(z) = blur(decoder(z))) returning vectors   
D_img = FiniteDifference().to(device)  
D     = DecodedD(ae.decoder, D_img).to(device)

A_latent = CompositeA(ae.decoder).to(device)
vp_solver = vpal(
    A=A_latent,
    D=D,
    mu=0.00001,
    step_size='linearized',
    lambda_=1.0,
    display='iter',
    maxIter=20000
)

# Prepare VPAL inputs in the shapes it expects
b   = y.view(batch_size, -1)
x0  = z0

# Run VPAL in latent space to recover z*
start = time.time()
z_star, info = vp_solver(b, x0=x0, return_info=True)
print("VPAL elapsed:", time.time()-start, "s")

# Decode latent solution to image space (no grad)
with torch.no_grad():
    recon_imgs = ae.decoder(z_star).cpu()  


# Visualization: original, blurred, reconstructed
plt.figure(figsize=(9, batch_size*3))
for i in range(batch_size):
    plt.subplot(batch_size, 3, 3*i + 1)
    plt.imshow(x[i,0].cpu(), cmap='gray')
    plt.title('Original'); plt.axis('off')

    plt.subplot(batch_size, 3, 3*i + 2)
    plt.imshow(y[i,0].cpu(), cmap='gray')
    plt.title('Blurred'); plt.axis('off')

    plt.subplot(batch_size, 3, 3*i + 3)
    plt.imshow(recon_imgs[i,0], cmap='gray')
    plt.title('Recon'); plt.axis('off')

plt.tight_layout()
plt.show()
