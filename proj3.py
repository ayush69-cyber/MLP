import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transfrm import EncoderBlock

with open("input.txt","r") as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch : i for i,ch in enumerate(chars)}
itos = {i : ch for i,ch in enumerate(chars)}

def encode(s):
    return[stoi[c] for c in s]

def decode(encoded):
    return "".join([itos[i] for i in encoded])

def get_batch(data,B,T):
    offsets = np.random.randint(0,len(data)-T,B)
    x = []
    y = []
    for i in offsets:
        x.append(data[i:i+T])
        y.append(data[i+1:i+T+1])
    return torch.stack(x),torch.stack(y)

data = encode(text)
test_len = int(len(data)*0.1)
test_idx = test_len // 2
start_idx = test_idx - test_len // 2
end_idx = start_idx + test_len
val_data = torch.tensor(data[start_idx:end_idx],dtype = torch.long)
train_data = torch.tensor(data[:start_idx] + data[end_idx:],dtype = torch.long)

# data = torch.tensor(encode(text),dtype = torch.long)
# train_data,val_data = data.split(int(len(data)*0.9))

class GPT(nn.Module):
    def __init__(self,T,vocab_size,d_model,num_blocks):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,d_model)
        self.position_embedding = nn.Embedding(T,d_model)
        self.num_blks = num_blocks
        self.blks = nn.ModuleList()
        for i in range(num_blocks):
            self.blks.append(EncoderBlock(d_model,8,4*d_model))
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model,vocab_size)
        self.dropout = nn.Dropout(p = 0)
    def causal_mask(self,T):
        mask = torch.full((T,T),-torch.inf)
        return torch.triu(mask,diagonal=1)
    def forward(self,x):
        B,T = x.shape
        mask = self.causal_mask(T)
        x = self.token_embedding(x) + self.position_embedding(torch.arange(T,device = x.device))
        for blk in self.blks:
            x = blk(x,mask)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

T = 64
B = 32

model = GPT(T, vocab_size, 128, 2)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if not os.path.exists("gpt_v0.pth"):

    for step in range(10000):

        model.train()

        x, y = get_batch(train_data, B, T)

        logits = model(x)

        batch_size, seq_len, vocab = logits.shape

        loss = criterion(
            logits.reshape(batch_size * seq_len, vocab),
            y.reshape(batch_size * seq_len)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 500 == 0:

            model.eval()

            with torch.no_grad():
                vx, vy = get_batch(val_data, B, T)

                val_logits = model(vx)

                val_loss = criterion(
                    val_logits.reshape(-1, vocab),
                    vy.reshape(-1)
                )

                predicted = val_logits.argmax(dim=-1)

            print(
                f"Step: {step} | "
                f"Train loss: {loss.item():.4f} | "
                f"Val loss: {val_loss.item():.4f}"
            )

            print("Input:    ", decode(vx[0].tolist()))
            print("Target:   ", decode(vy[0].tolist()))
            print("Predicted:", decode(predicted[0].tolist()))
            print("-" * 60)

        torch.save(model.state_dict(),"gpt_v0.pth")

else:
    state_dict = torch.load("gpt_v0.pth")
    model.load_state_dict(state_dict)
    model.eval()

# model.eval()
# with torch.no_grad():
#     x,y = get_batch(val_data,1,64)
#     logits1 = model(x)

#     x2 = x.clone()
#     x2[:,-1] = (x2[:,-1] + 1) % vocab_size
#     logits2 = model(x2)
#     print(torch.allclose(x[:,:-1],x2[:,:-1]) , torch.allclose(x[:,-1],x2[:,-1]))

# model.eval()

# context = torch.tensor([encode("Speak")], dtype=torch.long)

# with torch.no_grad():
#     for i in range(26):

#         # Keep only the most recent context window
#         context_cond = context[:, -T:]

#         # Forward pass
#         logits = model(context_cond)

#         # Get logits for the final position
#         logits = logits[:, -1, :]

#         # Select the most likely next token
#         predicted = logits.argmax(dim=-1, keepdim=True)

#         # Append prediction
#         context = torch.cat((context, predicted), dim=1)

#         print(decode([predicted.item()]), end="")

# print()

def estimate_loss(model,criterion,data,vocab_size,B,T,n_batches = 50):
    avg_loss = 0
    with torch.no_grad():
        model.eval()
        for i in range(n_batches):
            x,y = get_batch(data,B,T)
            logits = model(x)
            loss =  criterion(logits.reshape(B*T,vocab_size),y.reshape(B*T))
            avg_loss += loss
    avg_loss = avg_loss/n_batches
    return avg_loss
def estimate_loss_over_positions(model,data,B,T,no_of_batches):
    avg_loss = torch.zeros(T)
    with torch.no_grad():
        model.eval()
        for i in range(no_of_batches):
            x,y = get_batch(data,B,T)
            logits = model(x)
            loss = F.cross_entropy(logits.transpose(1,2),y,reduction="none")
            avg_loss = avg_loss + loss.mean(dim = 0)
        avg_loss = avg_loss / no_of_batches
    return avg_loss
def generate_text(model , idx , max_new_tokens, temperature = 1):
    context_window = torch.tensor([idx])
    with torch.no_grad():
        model.eval()
        for i in range(max_new_tokens):
            context_window = context_window[:, -model.position_embedding.num_embeddings:]
            logits = model(context_window)
            logits = logits[:,-1,:]
            logits = logits/temperature
            logits = torch.softmax(logits,-1)
            predicted = torch.multinomial(logits,1)
            context_window = torch.cat((context_window,predicted),dim = 1)
            idx.append(predicted.item())
    return idx
# print(f"Generation Text: {decode(generate_text(model,encode("ROMEO:"),300,1.5))}")

# avg_loss = estimate_loss_over_positions(model,val_data,B,T,50)
# print(f"Average loss on each position: {avg_loss}")
no_of_batches = 50
train_loss = estimate_loss_over_positions(model,train_data,B,T,no_of_batches)
val_loss = estimate_loss_over_positions(model,val_data,B,T,no_of_batches)
gap = abs(train_loss.mean() - val_loss.mean())
print(f"Parameter Count: {total_params}")
print(f"Average loss on train_data: {train_loss.mean():.4f}")
print(f"Average loss on val_data: {val_loss.mean():.4f}")
print(f"Gap: {gap:.4f}")
print(f"curve_base_mean : {val_loss[10:37].mean():.4f} , {val_loss[37:64].mean():.4f}")
# print(f"Mean_loss : {avg_loss.mean()}")