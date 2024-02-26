import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, value, key, query):
        N, query_len = query.shape[0], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = rearrange(value, "b n (h d) -> b h n d", h=self.heads)
        keys = rearrange(key, "b n (h d) -> b h n d", h=self.heads)
        queries = rearrange(query, "b n (h d) -> b h n d", h=self.heads)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other query, then scales, masks, and applies softmax
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerDecoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x, x, x)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, forward_expansion, dropout, max_length, num_layers):
        super(TransformerDecoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).repeat(N, 1).to(x.device)

        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x)

        return x


def generate_data(num_samples, sequence_length, vocabulary_size):
    X = torch.randint(vocabulary_size, (num_samples, sequence_length))
    y = X.clone()  # For a copying task, output is the same as input
    return X, y

if __name__ == "__main__":
    from tqdm import tqdm
    import torch.optim as optim
    from pprint import pprint
    from tensorboardX import SummaryWriter
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    embed_size = 128
    heads = 8
    forward_expansion = 4
    num_layers = 2
    sequence_length = 10
    vocabulary_size = 20
    dropout = 0.1

    # Create the decoder
    decoder = TransformerDecoder(vocabulary_size, embed_size, heads, forward_expansion, dropout, sequence_length, num_layers)
    pprint(decoder)
    
    # Generate some data
    X, y = generate_data(256, sequence_length, vocabulary_size)

    # Optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=1e-6)
    epochs = 100000


    decoder = decoder.to(device)
    X = X.to(device)
    y = y.to(device)

    # training in overfitting mode
    decoder.train()
    
    # Training loop
    update_bar = tqdm(range(epochs), total=epochs, desc="Training")
    writer = SummaryWriter()
    print("Creating tensorboard logs, run tensorboard --logdir=logs in terminal and go to http://localhost:6006/")
    
    for epoch in update_bar:
        optimizer.zero_grad()
        output = decoder(X)
        loss = criterion(output.reshape(-1, output.shape[2]), y.reshape(-1))
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), epoch)
        if epoch % 100 == 0:
            update_bar.set_postfix(loss=loss.item())

    writer.close()
    
     