from torch.utils.data import DataLoader
from dataset import GPTDataset
from config import GPTConfig
from model import GPT2
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

train_dataset = GPTDataset(GPTConfig.train_data_path, block_size=GPTConfig.block_size)
test_dataset = GPTDataset(GPTConfig.val_data_path, block_size=GPTConfig.block_size)

train_dataloader = DataLoader(train_dataset, batch_size=GPTConfig.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=GPTConfig.batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}")

model = GPT2()
model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=GPTConfig.learning_rate,            # Learning rate
    betas=GPTConfig.betas, # Coefficients for running averages of gradients and its square
    eps=GPTConfig.eps,           # Term added to the denominator to improve numerical stability
    weight_decay=GPTConfig.weight_decay   # Decoupled weight decay coefficient
)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/gpt2_trainer_{timestamp}')

# input, labels = next(iter(train_dataloader))
# print(input.shape)
# print(labels.shape)

def train_one_epoch(epoch_idx):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs. labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % GPTConfig.num_steps == GPTConfig.num_steps-1:
            last_loss = running_loss / GPTConfig.num_steps
            print(f"epoch {epoch_idx} step {i} loss: {last_loss}")
            running_loss = 0
            tb_x = epoch_idx * len(train_dataloader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)

    return last_loss


best_vloss = float("inf")

for epoch in range(GPTConfig.num_epochs):
    model.train(True) # set the module in training mode
    avg_loss = train_one_epoch(epoch)
    
    # set to eval mode
    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for i, val_data in enumerate(test_dataloader):
            val_inputs, val_labels = val_data
            val_inputs.to(device)
            val_data.to(device)
            val_outputs = model(val_inputs)
            val_loss = loss_fn(val_outputs, val_labels)
            val_running_loss += val_loss.item()

    avg_val_loss = val_running_loss / len(test_dataloader)
    print(f"epoch {epoch} validation loss: {avg_val_loss}")
    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_val_loss },
                    epoch + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_val_loss < best_vloss:
        best_vloss = avg_val_loss
        model_path = f'runs/model_{timestamp}_{epoch}'
        torch.save(model.state_dict(), model_path)