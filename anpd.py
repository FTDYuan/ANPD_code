import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared_encoder import NodeGNNHead
from .type_embedding import TypeEmbeddingMLP, load_roi2net_M
from .reconstructor import MAE_Reconstructor
from .gate_fusion import GateNet
from utils.mask_utils import mask_node_features


class DilatedTCN(nn.Module):
    def __init__(self, in_ch=1, hid=64, layers=3, k=9):
        super().__init__()
        channels = [hid] * layers
        modules = []
        dilation = 1
        in_channels = in_ch

        for out_channels in channels:
            padding = (k - 1) // 2 * dilation
            modules.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.ReLU(),
                nn.GroupNorm(1, out_channels),
            ])
            in_channels = out_channels
            dilation *= 2

        self.net = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.contiguous()
        x = self.net(x)
        return self.pool(x).squeeze(-1)


class AddictionGCNModel(nn.Module):
    def __init__(
        self,
        node_dim,
        hidden_dim,
        out_dim,
        num_types,
        fusion,
        prior_json="roi2net.json",
        film_d=32,
        film_lambda=0.1,
        gated_vector=False,
        proj2_drop=0.2,
    ):
        super().__init__()
        self.num_nodes = 116
        self.dropout = 0.5
        self.num_classes = num_types
        self.hidden_dim = hidden_dim
        self.t_len = 2500
        self.fusion = fusion
        self.num_features = out_dim * 2

        self.tcn = DilatedTCN(in_ch=1, hid=64, layers=3, k=9)
        self.shared_head = NodeGNNHead(
            in_ch=64,
            gnn_hid=64,
            heads=4,
            gnn_layers=2,
            out_dim=out_dim,
            edge_dim=1,
            dropout=0.2,
            use_jk=True,
            use_edge_dropout=True,
            edge_drop_p=0.1,
        )
        self.spec_head = NodeGNNHead(
            in_ch=64,
            gnn_hid=64,
            heads=4,
            gnn_layers=2,
            out_dim=out_dim,
            edge_dim=1,
            dropout=0.2,
            use_jk=True,
            use_edge_dropout=True,
            edge_drop_p=0.1,
        )

        if prior_json is not None and os.path.isfile(prior_json):
            prior_matrix, _ = load_roi2net_M(
                prior_json,
                N_expected=self.num_nodes,
            )
            self.type_emb = TypeEmbeddingMLP(
                input_dim=self.num_nodes * self.t_len,
                hidden_dim=64,
                embedding_dim=out_dim,
                num_types=num_types,
                prior_M=prior_matrix,
                N=self.num_nodes,
                T=self.t_len,
                film_d=film_d,
                film_lambda=film_lambda,
                gated_vector=gated_vector,
                proj2_drop=proj2_drop,
            )
        else:
            self.type_emb = TypeEmbeddingMLP(
                input_dim=self.num_nodes * self.t_len,
                hidden_dim=64,
                embedding_dim=out_dim,
                num_types=num_types,
            )

        self.mae = MAE_Reconstructor(
            in_dim=2 * out_dim,
            out_dim=node_dim,
        )

        if fusion == "gate":
            self.gate = GateNet(
                in_dim=out_dim * 2,
                hidden_dim=hidden_dim,
            )
        elif fusion != "concat":
            raise ValueError("Fusion must be 'concat' or 'gate'")

        self.fcs = nn.ModuleList([
            nn.Linear(self.num_features, self.hidden_dim),
            nn.Linear(self.num_nodes * self.hidden_dim, self.num_classes),
        ])

    def forward(self, data, apply_mask=True):
        batch_size = int(data.batch.max().item()) + 1
        num_nodes = self.num_nodes
        time_length = data.x.size(-1)

        signal = data.x.view(batch_size, num_nodes, time_length)
        flat_signal = signal.view(batch_size, -1)
        pred_type_emb, logits = self.type_emb(flat_signal, signal=signal)

        x = F.layer_norm(data.x, (data.x.size(-1),))
        x_orig = x.clone()

        if apply_mask:
            x_masked, node_mask = mask_node_features(
                x_orig,
                mask_ratio=0.1,
            )
        else:
            x_masked = x_orig
            node_mask = torch.zeros(
                x_orig.size(0),
                dtype=torch.bool,
                device=x_orig.device,
            )

        node_input = x_masked.view(-1, 1, time_length).contiguous()
        node_feat = self.tcn(node_input)

        z_shared_node = self.shared_head(
            node_feat,
            data.edge_index,
            data.edge_attr,
        )
        z_spec_node_temp = self.spec_head(
            node_feat,
            data.edge_index,
            data.edge_attr,
        )

        z_shared = z_shared_node.view(batch_size, num_nodes, -1)
        z_spec_temp = z_spec_node_temp.view(batch_size, num_nodes, -1)

        alpha = 0.3
        pred_type_dir = F.normalize(
            pred_type_emb.unsqueeze(1)
            .expand(-1, num_nodes, -1)
            .reshape(batch_size * num_nodes, -1),
            dim=-1,
        )
        gamma = 1.0 + alpha * pred_type_dir
        z_spec = (z_spec_node_temp * gamma).view(batch_size, num_nodes, -1)

        if self.fusion == "concat":
            z_used = torch.cat([z_shared, z_spec], dim=-1)
        else:
            z_used = self.gate(z_shared, z_spec)

        z_used = F.relu(self.fcs[0](z_used))
        z_used = F.dropout(
            z_used,
            p=self.dropout,
            training=self.training,
        )
        y_pred = self.fcs[-1](z_used.reshape(batch_size, -1))

        z_concat_node = torch.cat(
            [z_shared_node, z_spec_node_temp],
            dim=-1,
        )
        recon_x = self.mae(z_concat_node)
        recon_x_masked = recon_x[node_mask]
        x_true_masked = x_orig[node_mask]

        return (
            y_pred,
            recon_x_masked,
            x_true_masked,
            z_shared,
            z_spec,
            pred_type_emb,
            node_mask,
            z_spec_temp,
            logits,
        )

    def save_model(self, path="checkpoints/model.pth"):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)
