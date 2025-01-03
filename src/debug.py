import torch
import torch.nn as nn
import torch.optim as optim
from util.utils import n_params
from model.model import ChessModel
from dataset.dataset import prepare_data
import torch.nn.functional as F

model = ChessModel(num_embeddings=1900, transformer_width=512, transformer_layers=12, transfromer_nheads=8)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


print(f"Model has {n_params(model)} parameters.")
use_gpu = torch.cuda.is_available()
print(f"Using GPU: {use_gpu}.\n")
if use_gpu:
    model.gpu_mode()
    device = torch.device("cuda:0")
else:
    model.cpu_mode()
    device = torch.device("cpu")



# src = torch.randint(0, 10, (32, 12)).to(model.device)  # Source sequence
# tgt = torch.randint(0, 10, (32, 12)).to(model.device)  # Target sequence

# tgt_input = tgt[:, :-1]  # Remove the last token for input
# tgt_output = tgt[:, 1:]  # Shifted target sequence for output

# output = output.reshape(-1, output.shape[-1])
# tgt_output = tgt_output.reshape(-1)

# output = model(src, tgt_input)

dataset = prepare_data(batch_size=2)
train_dl = dataset["train_dl"]
test_dl = dataset["test_dl"]
eval_dl = dataset["eval_dl"]

batch = next(iter(train_dl))

src_input = batch["src_input"].to(model.device)
tgt_input = batch["tgt_input"].to(model.device)
tgt_output = batch["tgt_output"].to(model.device)

output = model(src_input, tgt_input)
output = output.reshape(-1, output.shape[-1])
tgt_output = tgt_output.reshape(-1)
tgt_encoding = model.encode_moves(tgt_output)

print(output.shape, tgt_encoding.shape)

# normalized features
output = output / output.norm(dim=1, keepdim=True)
tgt_encoding = tgt_encoding / tgt_encoding.norm(dim=1, keepdim=True)

# cosine similarity as logits
logits_per_move = output @ tgt_encoding.t()
logits_per_move_t = logits_per_move.t()
labels = torch.arange(0, logits_per_move.shape[0]).to(model.device)

ce_loss = (criterion(logits_per_move, labels) + criterion(logits_per_move_t, labels)) / 2
sim_loss = 1 - F.cosine_similarity(output, tgt_encoding, dim=-1).mean()

print(ce_loss, sim_loss)



# loss = criterion(output, tgt_output)

# # Backward pass and optimization
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# print(loss)