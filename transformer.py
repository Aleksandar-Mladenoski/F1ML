import torch.nn as nn
import torch

# Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers.
# The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
# We employ a residual connection [ 10 ] around each of the two sub-layers, followed by layer normalization [1].
# That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.
# To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.

class FeedForward(nn.Module):
    def __init__(self, num_features, multiplier):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(num_features, num_features*multiplier)
        self.fc2 = nn.Linear(num_features*multiplier, num_features)
        self.relu = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_features):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_features = num_features
        self.d_k = num_features // num_heads
        self.q_linear = nn.Linear(num_features, num_features)
        self.k_linear = nn.Linear(num_features, num_features)
        self.v_linear = nn.Linear(num_features, num_features)
        self.softmax = nn.Softmax(dim=-1)
        self.out_linear = nn.Linear(num_features, num_features)

    def split_heads(self, x):
        x = x.view(self.batch_size, self.sequence_length, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_k)

    def forward(self, x):
        batch_size, sequence_length, _ = x.size()

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        q_k = self.split_heads(Q)
        k_k = self.split_heads(K)
        v_k = self.split_heads(V)

        attention_scores = torch.matmul(q_k, k_k.transpose(-1, -2)) / torch.sqrt(self.d_k)
        attention = self.softmax(attention_scores)

        multi_head_attention = attention.view(batch_size, sequence_length, self.num_features)
        output = self.out_linear(multi_head_attention)

        return output


class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int = 3, num_features: int = 41):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, num_features)
        self.layer_norm_1 = nn.LayerNorm(num_features)
        self.layer_norm_2 = nn.LayerNorm(num_features)
        self.ffn = FeedForward(num_features, 4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_1 = x
        x = self.attention(x)
        x += residual_1
        x = self.layer_norm_1(x)
        residual_2 = x
        x = self.fnn(x)
        x += residual_2
        x = self.layer_norm_2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_heads: int = 3, num_features: int = 41):
        super(TransformerEncoder, self).__init__()
        self.encoder_1 = EncoderBlock(num_heads, num_features)
        self.encoder_2 = EncoderBlock(num_heads, num_features)
        self.encoder_3 = EncoderBlock(num_heads, num_features)
        self.encoder_4 = EncoderBlock(num_heads, num_features)
        self.encoder_5 = EncoderBlock(num_heads, num_features)
        self.encoder_6 = EncoderBlock(num_heads, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        x = self.encoder_4(x)
        x = self.encoder_5(x)
        x = self.encoder_6(x)
        
        return x

#print(torch.cuda.is_available())
