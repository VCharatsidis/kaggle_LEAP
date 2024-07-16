import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# Define a simple model
model = nn.Linear(10, 2)

# Define an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Define the scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.01)

# Track learning rate over epochs
lrs = []
for epoch in range(60):  # 30 epochs
    # Simulate a training step
    optimizer.step()

    # Record the current learning rate
    lrs.append(scheduler.get_last_lr()[0])

    # Step the scheduler
    scheduler.step()

# Plot learning rate
plt.plot(lrs)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("CosineAnnealingLR with Single Cycle")
plt.show()
