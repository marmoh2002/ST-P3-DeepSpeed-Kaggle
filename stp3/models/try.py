import torch
from torch.utils.data import DataLoader, TensorDataset

# Sample data (replace with your actual data)
X = torch.randn(100, 3, 224, 224)  # 100 samples, 3 channels, 224x224 images
y = torch.randint(0, 10, (100,))  # 100 labels (0-9)

# Create a Dataset
dataset = TensorDataset(X, y)

# Create a DataLoader with a batch size of 32
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
i=0
for batch_X, batch_y in dataloader:
    i+=1
    print(f"Batch {i}:")
    # Print the shape of the batch
    print(batch_X.shape)  # Output: torch.Size([32, 3, 224, 224]) (except for the last batch)
    print(batch_y.shape)  # Output: torch.Size([32]) (except for the last batch)
    # ... your training logic here ...