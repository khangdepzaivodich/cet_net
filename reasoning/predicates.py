"""
Predicate Collection

Collects all predicate truth values from the three branches into
a unified structure for the symbolic reasoning engine.

Three categories:
  - 8 geometry predicates (from GeometryBranch)
  - 5 texture predicates (from TextureBranch)
  - 18 muscle predicates (from MuscleActivationHead)
"""


# Geometry predicate names (indices into geometry_predicates tensor)
GEOMETRY_PREDICATES = {
    "CornerUp": 0,
    "CornerDown": 1,
    "BrowRaised": 2,
    "BrowsTogether": 3,
    "LipSeparated": 4,
    "EyeOpenness": 5,
    "CheekRaised": 6,
    "ChinRaised": 7,
}

# Texture predicate names
TEXTURE_PREDICATES = {
    "CrowFeet": 0,
    "ForeheadWrinkle": 1,
    "GlabellarWrinkle": 2,
    "NasolabialFold": 3,
    "ChinWrinkle": 4,
}

# Muscle predicate names (indices match knowledge/muscle_anatomy.py ordering)
MUSCLE_PREDICATES = {
    "Frontalis_Medialis": 0,
    "Frontalis_Lateralis": 1,
    "Corrugator": 2,
    "Depressor_Supercilii": 3,
    "Orbicularis_Oculi_Orbital": 4,
    "Orbicularis_Oculi_Palpebral": 5,
    "Levator_Palpebrae": 6,
    "Levator_Labii_Superioris": 7,
    "LLSAN": 8,
    "Zygomaticus_Major": 9,
    "Zygomaticus_Minor": 10,
    "Risorius": 11,
    "Buccinator": 12,
    "Depressor_Anguli_Oris": 13,
    "Depressor_Labii_Inferioris": 14,
    "Mentalis": 15,
    "Orbicularis_Oris": 16,
    "Nasalis": 17,
}


class PredicateStore:
    """
    Container for all predicate truth values from a forward pass.
    Provides named access to individual predicates.
    """

    def __init__(self, geometry_preds, texture_preds, muscle_preds):
        """
        Args:
            geometry_preds: [B, 8] tensor
            texture_preds:  [B, 5] tensor
            muscle_preds:   [B, 18] tensor
        """
        self.geometry = geometry_preds
        self.texture = texture_preds
        self.muscle = muscle_preds

    def geo(self, name):
        """Get geometry predicate by name. Returns [B] tensor."""
        return self.geometry[:, GEOMETRY_PREDICATES[name]]

    def tex(self, name):
        """Get texture predicate by name. Returns [B] tensor."""
        return self.texture[:, TEXTURE_PREDICATES[name]]

    def mus(self, name):
        """Get muscle predicate by name. Returns [B] tensor."""
        return self.muscle[:, MUSCLE_PREDICATES[name]]
