"""
Expression Composition Head
Maps AU beliefs to expression probabilities.

Expressions are NOT predicted independently from pixels.
They are computed as a consequence of the AU belief vector.

Two modes:

Standard mode (seen classes):
  Input:  [B, K]  final AU beliefs
  Output: [B, C_seen]  expression probabilities

Zero-shot mode (unseen classes):
  Unseen expressions are constructed as compositions of AU basis vectors.
  Each unseen class is defined by an AU descriptor (which AUs make up that expression).
  The model builds a prototype for the unseen class from learned AU basis vectors,
  adds a non-linear interaction correction, then classifies by cosine similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpressionHead(nn.Module):
    def __init__(self, num_aus=12, num_expressions=7, zsl_temperature=10.0):
        super().__init__()
        self.num_aus = num_aus
        self.num_expressions = num_expressions
        self.zsl_temperature = zsl_temperature

        # --- Standard mode ---
        # Composition layer: maps AU belief vector to expression logits.
        # The weight matrix W[expr, au] learns which AUs define which expression.
        self.compose = nn.Linear(num_aus, num_expressions)

        # --- Zero-shot mode ---
        # Learned AU basis vectors: each AU gets a D-dimensional embedding
        # that captures its semantic meaning in "expression space"
        self.basis_dim = 64
        self.au_basis = nn.Parameter(torch.randn(num_aus, self.basis_dim))
        nn.init.xavier_uniform_(self.au_basis)

        # Interaction correction network: models non-linear AU combinations
        # Because expressions are NOT linear sums of AUs (e.g., AU6+AU12 together
        # means something different than AU6 alone + AU12 alone)
        self.interaction_correction = nn.Sequential(
            nn.Linear(num_aus, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.basis_dim),
        )

        # Belief projector: maps AU belief vector into the same embedding space
        # as the prototypes, so we can compare them with cosine similarity
        self.belief_projector = nn.Sequential(
            nn.Linear(num_aus, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.basis_dim),
        )

    def forward(self, beliefs, mode="standard", au_descriptors=None):
        """
        Args:
            beliefs: [B, K] final AU beliefs from factor graph
            mode: "standard" for seen classes, "zsl" for zero-shot,
                  "generalized" for generalized zero-shot (seen + unseen)
            au_descriptors: [C_unseen, K] binary AU descriptors for unseen classes.
                            Required when mode is "zsl" or "generalized".

        Returns:
            expr_probs: [B, C] expression probabilities
            expr_logits: [B, C] raw logits
        """
        if mode == "standard":
            logits = self.compose(beliefs)  # [B, C_seen]
            probs = F.softmax(logits, dim=-1)
            return probs, logits

        elif mode == "zsl":
            # Pure zero-shot: only classify into unseen classes
            assert au_descriptors is not None, \
                "au_descriptors required for ZSL mode"
            logits = self._zsl_classify(beliefs, au_descriptors)
            probs = F.softmax(logits, dim=-1)
            return probs, logits

        elif mode == "generalized":
            # Generalized zero-shot: classify into both seen AND unseen classes
            assert au_descriptors is not None, \
                "au_descriptors required for generalized ZSL mode"
            # Get seen class logits from the standard head
            seen_logits = self.compose(beliefs)  # [B, C_seen]
            # Get unseen class logits from ZSL prototypes
            unseen_logits = self._zsl_classify(beliefs, au_descriptors)  # [B, C_unseen]
            # Concatenate: [B, C_seen + C_unseen]
            all_logits = torch.cat([seen_logits, unseen_logits], dim=-1)
            all_probs = F.softmax(all_logits, dim=-1)
            return all_probs, all_logits

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _zsl_classify(self, beliefs, au_descriptors):
        """
        Classify by cosine similarity between projected beliefs and class prototypes.

        Args:
            beliefs: [B, K] AU belief vector
            au_descriptors: [C_unseen, K] binary AU descriptors for unseen classes

        Returns:
            logits: [B, C_unseen] similarity scores (scaled by temperature)
        """
        # Step 1: Build prototypes for each unseen class
        # prototypes: [C_unseen, basis_dim]
        prototypes = self.build_prototypes(au_descriptors)

        # Step 2: Project the AU belief vector into the same embedding space
        # belief_embedding: [B, basis_dim]
        belief_embedding = self.belief_projector(beliefs)

        # Step 3: Cosine similarity between each sample and each prototype
        # Normalize both to unit vectors
        belief_norm = F.normalize(belief_embedding, p=2, dim=-1)  # [B, D]
        proto_norm = F.normalize(prototypes, p=2, dim=-1)  # [C_unseen, D]

        # Cosine similarity: [B, C_unseen]
        similarity = belief_norm @ proto_norm.t()

        # Scale by temperature to sharpen the distribution
        logits = similarity * self.zsl_temperature

        return logits

    def build_prototypes(self, au_descriptors):
        """
        Build prototypes for multiple expression classes from their AU descriptors.

        For each class, the prototype is:
            p = sum(a_k * basis_k) + correction(a)

        where a is the binary AU descriptor and basis_k is the learned
        embedding for AU k.

        Args:
            au_descriptors: [C, K] binary vectors, one per class

        Returns:
            prototypes: [C, basis_dim]
        """
        # Weighted sum of AU basis vectors
        # au_descriptors: [C, K], au_basis: [K, basis_dim]
        weighted = au_descriptors @ self.au_basis  # [C, basis_dim]

        # Interaction correction: captures non-linear AU combinations
        correction = self.interaction_correction(au_descriptors)  # [C, basis_dim]

        prototypes = weighted + correction  # [C, basis_dim]
        return prototypes

    def build_single_prototype(self, au_descriptor):
        """
        Build a prototype for a single unseen expression class.

        Args:
            au_descriptor: [K] binary vector indicating which AUs define this class

        Returns:
            prototype: [basis_dim] the synthesized prototype vector
        """
        return self.build_prototypes(au_descriptor.unsqueeze(0)).squeeze(0)
