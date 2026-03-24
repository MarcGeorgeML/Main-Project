import torch
import torch.nn as nn
import torch.nn.functional as F




'''
Bidirectional cross-attention layers.
'''
class BidirectionalCrossAttention(nn.Module):

    def __init__(self, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.query_matrix = nn.Linear(model_dim, Q_dim)
        self.key_matrix = nn.Linear(model_dim, K_dim)
        self.value_matrix = nn.Linear(model_dim, V_dim)
    

    def bidirectional_scaled_dot_product_attention(self, Q, K, V):
        score = torch.bmm(Q, K.transpose(-1, -2))
        scaled_score = score / (K.shape[-1]**0.5)
        attn = F.softmax(scaled_score, dim=-1)
        attn = torch.clamp(attn, min=1e-6)
        attention = torch.bmm(attn, V)

        return attention
    

    def forward(self, query, key, value):
        Q = self.query_matrix(query)
        K = self.key_matrix(key)
        V = self.value_matrix(value)
        attention = self.bidirectional_scaled_dot_product_attention(Q, K, V)

        return attention



'''
Multi-head bidirectional cross-attention layers.
'''
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList(
            [BidirectionalCrossAttention(model_dim, Q_dim, K_dim, V_dim) for _ in range(self.num_heads)]
        )
        self.projection_matrix = nn.Linear(num_heads * V_dim, model_dim)

    def forward(self, query, key, value):
        heads = torch.cat(
            [head(query, key, value) for head in self.attention_heads],
            dim=-1
        )
        multihead_attention = self.projection_matrix(heads)

        return multihead_attention



'''
A feed-forward network, which operates as a key-value memory.
'''
class Feedforward(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.linear_W1 = nn.Linear(model_dim, hidden_dim)
        self.linear_W2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    

    def forward(self, x):
        return self.dropout(self.linear_W2(self.relu(self.linear_W1(x))))



'''
Residual connection to smooth the learning process.
'''
class AddNorm(nn.Module):

    def __init__(self, model_dim, dropout_rate):
        super().__init__()

        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    
    def forward(self, x, sublayer):
        out = x + self.dropout(sublayer(x))
        output = self.layer_norm(out)
        return output



'''
MultiAttn is a multimodal fusion model which aims to capture the complicated interactions and 
dependencies across textual, audio and visual modalities through bidirectional cross-attention layers.
MultiAttn is made up of three sub-components:
1. MultiAttn_text: integrate the textual modality with audio and visual information;
2. MultiAttn_audio: incorporate the audio modality with textual and visual information;
3. MultiAttn_visual: fuse the visual modality with textual and visual cues.
'''
class MultiAttnLayer(nn.Module):

    def __init__(self, num_heads, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        Q_dim = K_dim = V_dim = model_dim // num_heads
        self.attn_1 = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim)
        self.add_norm_1 = AddNorm(model_dim, dropout_rate)
        self.attn_2 = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim)
        self.add_norm_2 = AddNorm(model_dim, dropout_rate)
        self.ff = Feedforward(model_dim, hidden_dim, dropout_rate)
        self.add_norm_3 = AddNorm(model_dim, dropout_rate)


    def forward(self, query_modality, modality_A, modality_B):
        attn1 = self.attn_1(query_modality, modality_A, modality_A)
        attn_output_1 = self.add_norm_1(query_modality, lambda _: attn1)
        attn_output_2 = self.add_norm_2(attn_output_1, lambda attn_output_1: self.attn_2(attn_output_1, modality_B, modality_B))
        ff_output = self.add_norm_3(attn_output_2, self.ff)

        return ff_output



'''
Stacks of MultiAttn layers.
'''
class MultiAttn(nn.Module):

    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()

        self.multiattn_layers = nn.ModuleList([
            MultiAttnLayer(num_heads, model_dim, hidden_dim, dropout_rate) for _ in range(num_layers)])


    def forward(self, query_modality, modality_A, modality_B):
        for multiattn_layer in self.multiattn_layers:
            query_modality = multiattn_layer(query_modality, modality_A, modality_B)
        
        return query_modality



class MultiAttnModel(nn.Module):

    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()

        self.multiattn_text = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_audio = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_visual = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        
        self.modality_gate = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # text weightage
        # self.text_weight = 2.0


    
    def forward(self, text_features, audio_features, visual_features):
        # modality gating
        combined = torch.cat(
            (text_features, audio_features, visual_features),
            dim=-1
        )

        gates = self.modality_gate(combined)

        g_t = gates[...,0].unsqueeze(-1)
        g_a = gates[...,1].unsqueeze(-1)
        g_v = gates[...,2].unsqueeze(-1)

        text_features = text_features * g_t
        audio_features = audio_features * g_a
        visual_features = visual_features * g_v
        f_t = self.multiattn_text(text_features, audio_features, visual_features)
        f_a = self.multiattn_audio(audio_features, text_features, visual_features)
        f_v = self.multiattn_visual(visual_features, text_features, audio_features)

        return f_t, f_a, f_v