
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader

from am4ip.dataset import TID2013
from am4ip.models import CBDNetwork
from am4ip.trainer import BaselineTrainer
from am4ip.losses import TotalLoss
from am4ip.metrics import nMAE

transform = Compose([PILToTensor(),
                     lambda z: z.to(dtype=torch.float32) / 127.5 - 1  # Normalize between -1 and 1
                     ])
batch_size = 32
lr = 1e-3
epoch = 1

dataset = TID2013(transform=transform, crop_size=(224, 224))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Implement VAE model:
# TODO: complete parameters and implement model forward pass + sampling
model = CBDNetwork()

# Implement loss function:
# TODO: implement the loss function as presented in the course
loss = TotalLoss()

# Choose optimize:
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Implement the trainer
trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer)

# Do the training
trainer.fit(train_loader, epoch=epoch)

# Compute metrics
# TODO: implement evaluation (compute IQ metrics on restaured images similarly to lab1)

print("job's done.")
