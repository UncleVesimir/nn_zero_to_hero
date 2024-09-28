import torch
import torch.nn as nn
from torch.nn import functional as F



# HYPERPARAMETERS
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32

# ----------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf8') as f:
  text = f.read()

# Determine vocab size + encode/decode
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch : i for (i, ch) in enumerate(chars)}
itos = {i: ch for (i, ch) in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

# ---------------------

# Train and Test Splits
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.95 * len(data))
train_data, val_data = data[:n], data[n:]


#Data Loader
def get_batch(split):
    batch_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(batch_data) - block_size, (batch_size,))
    x = torch.stack([batch_data[i: i + block_size] for i in ix])
    y = torch.stack([batch_data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(): 
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k  in range(eval_iters):
      X, Y = get_batch(split)
      _, loss = model(X, Y)
      losses[k] = loss.item()

    out[split] = losses.mean()
    model.train()
    return out


# -------

# Define Model

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # we have 64 individual tokens we want to embed (first arg), and the embedding dimension is going to be at least as large as 
        # the entire character space.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # output is (batch, block_size, embed dim)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # (B, T, C)
        position_embeddings  = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embeddings + position_embeddings
        logits = self.lm_head(x)

        # FROM PYTORCH DOCS:
        # INPUT:
        # Shape can be one of the following:
        # A 1D shape of size (C) where C is the number of classes.
        # A 2D shape of size (N, C) where N is the batch size and C is the number of classes.
        # A more complex shape of size (N, C, d1, d2, ..., dK) where N is the batch size, C is the number of classes, 
        # and d1, d2, ..., dK are additional dimensions. This happens when you have a K-dimensional loss, 
        # and K is greater than or equal to 1.
        
        # TARGET:
        # If the target contains class indices, its shape can be:
        # An empty shape (), representing a single value.
        # A 1D shape of size (N) where N is the batch size.
        # A more complex shape of size (N, d1, d2, ..., dK) where N is the batch size and d1, d2, ..., dK are additional dimensions, similar to the input case where you have a K-dimensional loss, and K is greater than or equal to 1.

        #...so, we know we have to view 'targets' as 1D tensor (all the "next characters" in a 'column')
        # --> (4, 8) => (32).
        # AND for the input, we simply need to "combine" all of the word embeddings
        # down into individual rows. *i.e. the embeddings ARE the NN (though this will
        # change later on.
        # --> (4, 8, 64) => (32, 64)
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # combine the batch * and blocks all into one,   
            targets = targets.view(B*T) #squash down to a single column
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss 

    def generate(self, idx, max_new_tokens):
        #idx is a (B,T) array of indices in the current context
        
        for _ in range(max_new_tokens):
            #get predictions
            logits, _ = self(idx)

            logits = logits[:, -1, :] #grab only the last timestep, (B, C)
            
            probs = F.softmax(logits, dim=-1) # (B, C)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx
            

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"Step {iter}: train loss {losses['train']:.4f}, val losses {losses['val']:.4f]}")
  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


context = torch.zeros((1,1), dtype=torch.long, device=device)
generation = model.generate(context, max_new_tokens=20)
print(decode(generation[0].tolist()))