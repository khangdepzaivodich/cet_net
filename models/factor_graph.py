"""
Causal AU Factor Graph with Belief Transport
A GNN that updates AU beliefs using uncertainty-gated message passing.

The key idea: edges are not just learned similarity — they are belief constraints.
Rules fire more strongly when the model is uncertain or inconsistent.

Input:
  - au_features:  [B, K, 256]  per-AU feature vectors
  - beliefs:      [B, K]       initial AU beliefs
  - uncertainty:  [B, K]       uncertainty per AU

Output:
  - beliefs_final: [B, K]     updated AU beliefs after message passing
"""

import torch
import torch.nn as nn


class BeliefTransportLayer(nn.Module):
    """One round of message passing with uncertainty-gated rule activation."""

    def __init__(self, hidden_dim=256, num_aus=12):
        super().__init__()
        self.num_aus = num_aus

        # Message function: takes source AU features + source belief -> message
        # Input: [hidden_dim + 1] (feature vec + belief scalar)
        self.message_fn = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gate parameters: learnable scalars a and b
        # gamma = sigmoid(a * violation + b * uncertainty)
        self.gate_a = nn.Parameter(torch.tensor(1.0))
        self.gate_b = nn.Parameter(torch.tensor(1.0))

        # Update function: takes current belief + aggregated message -> new belief
        # Input: [hidden_dim + 1] (aggregated msg + current belief)
        self.update_fn = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, au_features, beliefs, uncertainty, adjacency):
        """
        Args:
            au_features: [B, K, 256]
            beliefs:     [B, K]
            uncertainty: [B, K]
            adjacency:   [K, K] with values in {-1, 0, +1}

        Returns:
            new_beliefs: [B, K] updated beliefs
        """
        B, K, D = au_features.shape
        device = au_features.device

        # Prepare source features: concatenate features with beliefs
        # beliefs: [B, K] -> [B, K, 1]
        beliefs_expanded = beliefs.unsqueeze(-1)
        src_input = torch.cat([au_features, beliefs_expanded], dim=-1)  # [B, K, D+1]

        # Compute messages from all sources
        # [B, K, D+1] -> [B, K, D]
        messages = self.message_fn(src_input)  # [B, K, D]

        # For each target AU j, aggregate messages from connected source AUs
        # We need to iterate through edges defined by adjacency
        aggregated = torch.zeros(B, K, D, device=device)

        # Get non-zero edges
        edge_src, edge_dst = adjacency.nonzero(as_tuple=True)

        for idx in range(len(edge_src)):
            i = edge_src[idx].item()  # source AU
            j = edge_dst[idx].item()  # target AU
            edge_weight = adjacency[i, j]  # +1 or -1

            # Message from AU i
            msg = messages[:, i, :]  # [B, D]

            # Rule violation: how much does the current state violate expectations?
            # If edge_weight is +1 (should co-occur), violation = |z_i - z_j|
            # If edge_weight is -1 (should not co-occur), violation = z_i * z_j
            if edge_weight > 0:
                violation = torch.abs(beliefs[:, i] - beliefs[:, j])  # [B]
            else:
                violation = beliefs[:, i] * beliefs[:, j]  # [B]

            # Confidence gate: opens wider when uncertain or violating
            gamma = torch.sigmoid(
                self.gate_a * violation + self.gate_b * uncertainty[:, j]
            )  # [B]

            # Apply gate and edge weight to message
            gated_msg = msg * gamma.unsqueeze(-1) * edge_weight  # [B, D]

            aggregated[:, j, :] = aggregated[:, j, :] + gated_msg

        # Update beliefs
        update_input = torch.cat([aggregated, beliefs_expanded], dim=-1)  # [B, K, D+1]
        update_flat = update_input.view(B * K, -1)  # [B*K, D+1]
        delta = self.update_fn(update_flat)  # [B*K, 1]
        delta = delta.view(B, K)  # [B, K]

        new_beliefs = torch.sigmoid(
            torch.logit(beliefs.clamp(1e-6, 1 - 1e-6)) + delta
        )

        return new_beliefs


class CausalFactorGraph(nn.Module):
    """
    Multi-layer belief transport over the AU factor graph.
    Stacks multiple BeliefTransportLayers.
    """

    def __init__(self, hidden_dim=256, num_aus=12, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            BeliefTransportLayer(hidden_dim, num_aus)
            for _ in range(num_layers)
        ])

    def forward(self, au_features, beliefs_init, uncertainty, adjacency):
        """
        Args:
            au_features:  [B, K, 256]
            beliefs_init: [B, K]
            uncertainty:  [B, K]
            adjacency:    [K, K]

        Returns:
            beliefs_final: [B, K]
        """
        beliefs = beliefs_init
        for layer in self.layers:
            beliefs = layer(au_features, beliefs, uncertainty, adjacency)
        return beliefs
