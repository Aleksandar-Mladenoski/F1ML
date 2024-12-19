from collections import OrderedDict

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
        self.num_drivers = 20
        self.num_heads = num_heads
        self.num_features = num_features
        self.d_k = num_features // num_heads
        self.q_linear = nn.Linear(num_features, num_features)
        self.k_linear = nn.Linear(num_features, num_features)
        self.v_linear = nn.Linear(num_features, num_features)
        self.softmax = nn.Softmax(dim=-1)
        self.out_linear = nn.Linear(num_features, num_features)

    def split_heads(self, x, batch_size, sequence_length):
        x = x.view(batch_size, self.num_drivers, sequence_length, self.num_heads, self.d_k)
        return x.permute(0, 1, 3, 2, 4)  # (batch_size, drivers, num_heads, seq_len, d_k)

    def forward(self, x):
        batch_size, drivers_size, sequence_length, feature_length = x.size()

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        q_k = self.split_heads(Q, batch_size, sequence_length)
        k_k = self.split_heads(Q, batch_size, sequence_length)
        v_k = self.split_heads(Q, batch_size, sequence_length)


        attention_scores = torch.matmul(q_k, k_k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention = self.softmax(attention_scores)
        attention = torch.matmul(attention, v_k)

        multi_head_attention = attention.view(batch_size, self.num_drivers, sequence_length, self.num_features)
        output = self.out_linear(multi_head_attention)

        return output


class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int = 2, num_features_old: int = 106, num_features_new: int = 106, num_multiplier: int = 1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, num_features_new)
        self.linear_conv = nn.Linear(num_features_old, num_features_new)
        self.num_features_old = num_features_old
        self.num_features_new = num_features_new
        self.layer_norm_1 = nn.LayerNorm(num_features_new)
        self.layer_norm_2 = nn.LayerNorm(num_features_new)
        self.fnn = FeedForward(num_features_new, num_multiplier)
        self.relu = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_conv(x)
        x = self.relu(x)
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
    def __init__(self, num_heads: int = 2, num_features_list: list = [30 for _ in range(6)] , num_blocks: int = 6, num_multiplier: int = 1):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.Sequential(OrderedDict([(f'EncoderBlock+{i}', EncoderBlock(num_heads, num_features_list[i], num_features_list[i+1], num_multiplier)) for i in range(num_blocks-1)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x

class F1MLTransformer(nn.Module):
    def __init__(self, num_heads: int = 3, num_embeddings: int = 106, num_features_list: list = [30 for _ in range(6)], num_blocks: int = 6, sequence_length: int = 219, num_multiplier: int = 1, linear_list: list = [512, 128, 32]):
        super(F1MLTransformer, self).__init__()
        self.num_drivers = 20
        self.num_heads = num_heads
        self.num_features_list = num_features_list
        self.num_embeddings = num_embeddings
        self.num_blocks = num_blocks
        self.num_multiplier = num_multiplier
        self.transformer = TransformerEncoder(num_heads, num_features_list, num_blocks, num_multiplier)
        self.linear_input = nn.Linear(num_features_list[-1]*sequence_length, linear_list[0])
        self.linear_layers = nn.Sequential(OrderedDict( [(f'LinearLayer_+{i}', nn.Sequential(nn.Linear(linear_list[i], linear_list[i + 1]), nn.ReLU())) for i in range(len(linear_list) - 1)]))
        self.linear_output = nn.Linear(linear_list[-1], 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, drivers_size, sequence_length, feature_length = x.size()
        x = self.transformer(x)
        x = x.view(batch_size, self.num_drivers, sequence_length * self.num_features_list[-1])
        x = self.linear_input(x)
        x = self.relu(x)
        x = self.linear_layers(x)
        x = self.linear_output(x)

        return x




#print(torch.cuda.is_available())
