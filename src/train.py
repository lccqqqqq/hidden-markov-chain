import torch as t
from src.utils import initialize_transformer_from_yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from torch.utils.data import DataLoader
# Integration with wandb
# Setup optimizer
# Setup checkpointing

CONFIG_PATH = "config/hook_transformer_config.yaml"
device = "cuda" if t.cuda.is_available() else "cpu"

# Initialize the model
model = initialize_transformer_from_yaml(CONFIG_PATH)
model.to(device)
model.train()

# Initialize the optimzer
cfg = yaml.load(open(CONFIG_PATH, "r"))
tcfg = cfg["train"]
ocfg = tcfg["optimizer"]

optimizer = t.optim.AdamW(
    model.parameters(),
    lr=ocfg['learning_rate'],
    betas=(ocfg['adam_beta1'], ocfg['adam_beta2']),
    eps=ocfg["adam_epsilon"],
    weight_decay=ocfg["adam_weight_decay"],
)

# Initialize Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=tcfg["num_epochs"], eta_min=ocfg["learning_rate"] * 0.001)

# Initialize wandb
wandb.init(project=cfg["wandb"]["project_name"], name=cfg["wandb"]["experiment"])

# Initialize & calculate training loop, epochs, steps, etc spec.
num_epochs = tcfg["num_epochs"]
batch_size = tcfg["batch_size"]
seq_length = tcfg["seq_length"]
num_tokens_per_epoch = cfg["data_generator"]["num_tokens"]
tot_tokens = num_epochs * num_tokens_per_epoch
num_optimizer_steps_per_epoch = num_tokens_per_epoch // (batch_size * seq_length)


# Initialize Dataset
train_dataset = t.load(f"data/datasets/cylinder_graph_hmm/train/observations.pt")
test_dataset = t.load(f"data/datasets/cylinder_graph_hmm/test/observations.pt")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Optimizer Loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs} start training...")

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs)
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
