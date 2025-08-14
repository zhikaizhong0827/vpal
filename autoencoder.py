import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Hyperparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size   = 10
latent_dim   = 32
num_epochs   = 10
learning_rate= 1e-3

# 2. Data Loading —— Keep(B,1,28,28)
transform = transforms.ToTensor()
train_ds   = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_ds    = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# 3. Defining Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),                          # (B,1,28,28) -> (B,784)
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 64),    nn.ReLU(),
            nn.Linear(64, latent_dim)             # (B,latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),    nn.ReLU(),
            nn.Linear(64,        256),    nn.ReLU(),
            nn.Linear(256,      28*28),
            nn.Sigmoid(),                       # (B,784)
            nn.Unflatten(1, (1, 28, 28))        # -> (B,1,28,28)
        )

    def forward(self, x):
        z    = self.encoder(x)
        xrec = self.decoder(z)
        return xrec

# 4. Instantiate, Loss and Optimizer
ae        = Autoencoder(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=learning_rate)

# 5. Training Loop
for epoch in range(1, num_epochs+1):
    ae.train()
    total_loss = 0
    for x_batch, _ in train_loader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        x_rec = ae(x_batch)
        loss  = criterion(x_rec, x_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{num_epochs} — AE Loss: {avg_loss:.4f}")

# 6. Save Model
torch.save(ae.state_dict(), "autoencoder_mnist.pth")


