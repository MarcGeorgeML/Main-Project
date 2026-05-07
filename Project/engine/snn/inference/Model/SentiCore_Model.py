from .MultiAttn import MultiAttnModel
from .MLP import MLP
import torch
import torch.nn as nn

class SentiCore(nn.Module):

    def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout,
                 num_layers, model_dim, num_heads, D_m_audio, D_m_visual,
                 n_classes, spikformer_model):

        super().__init__()

        self.dataset = dataset
        self.multi_attn_flag = multi_attn_flag
        self.spikformer_model = spikformer_model

        # modality projections
        self.text_fc = nn.Linear(roberta_dim, model_dim)
        self.audio_fc = nn.Linear(D_m_audio, model_dim)
        self.visual_fc = nn.Linear(D_m_visual, model_dim)

        # multimodal fusion
        self.multiattn = MultiAttnModel(num_layers, model_dim, num_heads, hidden_dim, dropout)

        # classifier
        self.fc = nn.Linear(model_dim * 3, model_dim)

        if dataset == "MELD":
            self.mlp = MLP(model_dim, model_dim * 2, n_classes, dropout)
        else:
            self.mlp = MLP(model_dim, model_dim, n_classes, dropout)


    def spiking_modulation(self, x, s):
        s = torch.softmax(s, dim=-1)
        return x * (1 + s)


    def forward(self, texts, audios, visuals):

        # projection → [1, B, D] = [L, B, D], spikformer-ready
        text_features  = self.text_fc(texts)
        audio_features = self.audio_fc(audios)
        visual_features = self.visual_fc(visuals)

        # spikformer expects [L, B, D], returns [L, B, D]
        text_s   = self.spikformer_model(text_features)
        audio_s  = self.spikformer_model(audio_features)
        visual_s = self.spikformer_model(visual_features)

        # spiking modulation — all [1, B, D]
        text_features   = self.spiking_modulation(text_features,   text_s)
        audio_features  = self.spiking_modulation(audio_features,  audio_s)
        visual_features = self.spiking_modulation(visual_features, visual_s)

        # permute [1, B, D] → [B, 1, D] for MultiAttn (bmm needs 3D)
        text_features   = text_features.permute(1, 0, 2)
        audio_features  = audio_features.permute(1, 0, 2)
        visual_features = visual_features.permute(1, 0, 2)

        # multimodal fusion
        if self.multi_attn_flag:
            f_t, f_a, f_v = self.multiattn(text_features, audio_features, visual_features)
        else:
            f_t, f_a, f_v = text_features, audio_features, visual_features

        # squeeze [B, 1, D] → [B, D]
        f_t = f_t.squeeze(1)
        f_a = f_a.squeeze(1)
        f_v = f_v.squeeze(1)

        fused       = torch.cat((f_t, f_a, f_v), dim=-1)
        fc_outputs  = self.fc(fused)
        mlp_outputs = self.mlp(fc_outputs)

        return f_t, f_a, f_v, fc_outputs, mlp_outputs



