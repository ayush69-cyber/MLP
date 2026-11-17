import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

np.random.seed(42)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis = 1,keepdims=True)

def attention(Q,K,V):
    relevance_scores = np.dot(Q,np.transpose(K))
    softmaxd = softmax(relevance_scores / np.sqrt(Q.shape[1]))
    output = np.dot(softmaxd , V)
    return output,softmaxd

def multi_head_attention(x,num_heads,W_q_list,W_k_list,W_v_list,W_o):
    pre_final = []
    for head_no in range(num_heads):
        Q = np.dot(x,W_q_list[head_no])
        K = np.dot(x,W_k_list[head_no])
        V = np.dot(x,W_v_list[head_no])
        out , weights = attention(Q,K,V)
        pre_final.append(out)
    final = np.dot(np.concatenate(pre_final,axis = 1),W_o)
    return final

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len).reshape(-1,1)
    i = np.arange(0,d_model,2)
    denom = 10000** (i/d_model)
    angles = pos / denom
    encoding = np.zeros((seq_len,d_model))
    encoding[:,0::2] = np.sin(angles)
    encoding[:,1::2] = np.cos(angles)
    return encoding


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


class EncoderBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model,num_heads=num_heads,batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,x,causal_mask = None):
        attn_out,attn_wghts = self.mha(x,x,x,attn_mask = causal_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

def causal_mask(seq_len):
    mask = np.full((seq_len, seq_len), -np.inf)
    return np.triu(mask, k=1)

class DecoderBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,num_heads=num_heads,batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model,num_heads = num_heads,batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(in_features=d_model,out_features=d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self,y,encoder_out,causal_mask):
        self_attn_out,self_attn_wghts = self.self_attn(y,y,y,attn_mask = causal_mask)
        y = self.norm1(y+self_attn_out)
        cross_attn_out,cross_attn_wghts = self.cross_attn(y,encoder_out,encoder_out)
        y = self.norm2(y+cross_attn_out)
        y = self.norm3(y+self.ff(y))
        return y

class PatchEmbedding(nn.Module):
    def __init__(self,img_size,patch_size,in_channels,d_model):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=d_model,stride = patch_size , kernel_size=patch_size)
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self,x):
        # x: (batch, in_channels, img_size, img_size)
        c1 = self.conv(x)
        c1 = self.flatten(c1)
        c1 = c1.transpose(1,2)
        return c1

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model, num_heads, d_ff, num_layers, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch = x.shape[0]
        x = self.patch_embed(x)                                  # (batch, num_patches, d_model)
        cls_tokens = self.cls_token.expand(batch, -1, -1)          # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)                     # (batch, num_patches+1, d_model)
        x = x + self.pos_embed                                    # add learned positional encoding
        for block in self.encoder_blocks:
            x = block(x)
        cls_output = x[:, 0, :]                                    # take only the CLS token's output
        return self.classifier(cls_output)

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

    
if __name__ == "__main__":
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
    # Q = np.array([[1.0,0.0],
    #               [0.0,1.0],
    #               [0.9,0.8]])

    # K = np.array([[0.2,0.1],
    #               [0.7,1.0],
    #               [0.3,0.4]])
    # V = np.array([[1.0,0.0],
    #               [0.0,5.0],
    #               [2.0,2.0]])
    # out , weights = attention(Q , K , V)
    # print('Attention weights (rows=query word, cols=key word):')
    # print(weights)
    # print('Output (word 2, it, blended representation):')
    # print(out[2])

    # n, d_model, num_heads = 3, 8, 4
    # d_k = d_model // num_heads  # 2

    # x = np.random.randn(n, d_model)
    # W_q_list = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
    # W_k_list = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
    # W_v_list = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
    # W_o = np.random.randn(d_model, d_model)

    # output = multi_head_attention(x, num_heads, W_q_list, W_k_list, W_v_list, W_o)
    # print(output.shape)  # predict first: should be (n, d_model) = (3, 8)

    # print(positional_encoding(4,6))

    # model = EncoderBlock(d_model=8, num_heads=4, d_ff=32)
    # x = torch.randn(2, 5, 8)  # batch=2, seq_len=5, d_model=8
    # out = model(x)
    # print(out.shape)  # predict first

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