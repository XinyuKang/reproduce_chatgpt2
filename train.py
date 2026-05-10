from torch.utils.data import DataLoader
from dataset import GPTDataset
from config import GPTConfig

train_dataset = GPTDataset(GPTConfig.train_data_path, block_size=GPTConfig.block_size)
test_dataset = GPTDataset(GPTConfig.val_data_path, block_size=GPTConfig.block_size)

train_dataloader = DataLoader(train_dataset, batch_size=GPTConfig.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=GPTConfig.batch_size, shuffle=True)

input, labels = next(iter(train_dataloader))
print(input.shape)
print(labels.shape)