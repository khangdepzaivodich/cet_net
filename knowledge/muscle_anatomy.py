"""
Facial Muscle Anatomy Knowledge Base

Defines:
  1. The 18 facial muscles used as latent causal variables
  2. Anatomical adjacency matrix (which muscles are physically connected)
  3. Muscle → AU mapping (which muscles produce which Action Units)

References:
  - Ekman & Friesen (1978) FACS Manual
  - Standring, Gray's Anatomy (41st ed.)
"""

import torch

# ============================================================================
# 18 Facial Muscles (ordered by facial region: forehead → eyes → nose → mouth → chin)
# ============================================================================

MUSCLE_NAMES = [
    "Frontalis_Medialis",          # 0  - raises inner eyebrows
    "Frontalis_Lateralis",         # 1  - raises outer eyebrows
    "Corrugator_Supercilii",       # 2  - pulls brows together (frown)
    "Depressor_Supercilii",        # 3  - pulls brows down
    "Orbicularis_Oculi_Orbital",   # 4  - squeezes eye shut (cheek raise)
    "Orbicularis_Oculi_Palpebral", # 5  - gentle eye closure (blink)
    "Levator_Palpebrae",           # 6  - raises upper eyelid
    "Levator_Labii_Superioris",    # 7  - raises upper lip
    "LLSAN",                       # 8  - levator labii superioris alaeque nasi (nose wrinkle)
    "Zygomaticus_Major",           # 9  - pulls mouth corner up and back (smile)
    "Zygomaticus_Minor",           # 10 - pulls upper lip up
    "Risorius",                    # 11 - pulls mouth corner sideways
    "Buccinator",                  # 12 - compresses cheek against teeth
    "Depressor_Anguli_Oris",       # 13 - pulls mouth corner down (frown)
    "Depressor_Labii_Inferioris",  # 14 - pulls lower lip down
    "Mentalis",                    # 15 - raises chin (chin wrinkle)
    "Orbicularis_Oris",            # 16 - purses/tightens lips
    "Nasalis",                     # 17 - flares/compresses nostrils
]

NUM_MUSCLES = len(MUSCLE_NAMES)  # 18

# Short name → index lookup
MUSCLE_IDX = {name: i for i, name in enumerate(MUSCLE_NAMES)}


def get_muscle_adjacency(num_muscles=NUM_MUSCLES):
    """
    Returns a [num_muscles, num_muscles] binary adjacency matrix.
    Entry [i, j] = 1 if muscles i and j are anatomically connected/adjacent.
    Symmetric.
    """
    adj = torch.zeros(num_muscles, num_muscles)

    def _connect(a, b):
        i, j = MUSCLE_IDX[a], MUSCLE_IDX[b]
        adj[i, j] = 1
        adj[j, i] = 1

    # Forehead region
    _connect("Frontalis_Medialis", "Frontalis_Lateralis")
    _connect("Frontalis_Medialis", "Corrugator_Supercilii")
    _connect("Frontalis_Lateralis", "Corrugator_Supercilii")
    _connect("Corrugator_Supercilii", "Depressor_Supercilii")

    # Eye region
    _connect("Orbicularis_Oculi_Orbital", "Orbicularis_Oculi_Palpebral")
    _connect("Orbicularis_Oculi_Palpebral", "Levator_Palpebrae")
    _connect("Orbicularis_Oculi_Orbital", "Frontalis_Lateralis")   # orbital part near brow
    _connect("Orbicularis_Oculi_Orbital", "Zygomaticus_Major")     # orbital part near cheek

    # Nose region
    _connect("LLSAN", "Levator_Labii_Superioris")
    _connect("LLSAN", "Nasalis")
    _connect("Nasalis", "Levator_Labii_Superioris")

    # Mouth region (upper)
    _connect("Zygomaticus_Major", "Zygomaticus_Minor")
    _connect("Zygomaticus_Major", "Risorius")
    _connect("Zygomaticus_Major", "Orbicularis_Oris")
    _connect("Zygomaticus_Minor", "Levator_Labii_Superioris")
    _connect("Risorius", "Buccinator")

    # Mouth region (lower)
    _connect("Depressor_Anguli_Oris", "Depressor_Labii_Inferioris")
    _connect("Depressor_Anguli_Oris", "Orbicularis_Oris")
    _connect("Depressor_Anguli_Oris", "Risorius")
    _connect("Depressor_Labii_Inferioris", "Mentalis")
    _connect("Orbicularis_Oris", "Mentalis")
    _connect("Orbicularis_Oris", "Buccinator")

    return adj


def get_muscle_to_au_map(num_muscles=NUM_MUSCLES, au_list=None):
    """
    Returns a [num_muscles, num_aus] binary matrix.
    Entry [m, a] = 1 if muscle m contributes to AU a.

    au_list: list of AU numbers (e.g., [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26])
             Defaults to DISFA's 12 AUs.
    """
    if au_list is None:
        au_list = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

    num_aus = len(au_list)
    au_idx = {au: i for i, au in enumerate(au_list)}

    mapping = torch.zeros(num_muscles, num_aus)

    def _map(muscle_name, au_num):
        if au_num in au_idx:
            mapping[MUSCLE_IDX[muscle_name], au_idx[au_num]] = 1.0

    # Muscle → AU (from FACS manual)
    _map("Frontalis_Medialis", 1)           # AU1: Inner Brow Raiser
    _map("Frontalis_Lateralis", 2)          # AU2: Outer Brow Raiser
    _map("Corrugator_Supercilii", 4)        # AU4: Brow Lowerer
    _map("Depressor_Supercilii", 4)         # AU4: Brow Lowerer (secondary)
    _map("Levator_Palpebrae", 5)            # AU5: Upper Lid Raiser
    _map("Orbicularis_Oculi_Orbital", 6)    # AU6: Cheek Raiser
    _map("Orbicularis_Oculi_Palpebral", 6)  # AU6: Cheek Raiser (secondary)
    _map("LLSAN", 9)                        # AU9: Nose Wrinkler
    _map("Levator_Labii_Superioris", 9)     # AU9: (secondary contributor)
    _map("Levator_Labii_Superioris", 12)    # AU10→ mapped to closest available
    _map("Zygomaticus_Major", 12)           # AU12: Lip Corner Puller
    _map("Depressor_Anguli_Oris", 15)       # AU15: Lip Corner Depressor
    _map("Mentalis", 17)                    # AU17: Chin Raiser
    _map("Risorius", 20)                    # AU20: Lip Stretcher
    _map("Orbicularis_Oris", 25)            # AU25: Lips Part (relaxation)
    _map("Orbicularis_Oris", 26)            # AU26: Jaw Drop (relaxation)
    _map("Depressor_Labii_Inferioris", 25)  # AU25: secondary
    _map("Depressor_Labii_Inferioris", 26)  # AU26: secondary

    return mapping


def get_muscle_compatibility_pairs():
    """
    Returns list of (muscle_a, muscle_b, relation) tuples.
    relation: "antagonist" = muscles oppose each other (penalty if both high)
              "synergist"  = muscles tend to co-activate (reward if both active)
    """
    pairs = [
        # Antagonistic pairs (physically opposing)
        ("Zygomaticus_Major", "Depressor_Anguli_Oris", "antagonist"),
        ("Frontalis_Medialis", "Corrugator_Supercilii", "weak_antagonist"),
        ("Levator_Palpebrae", "Orbicularis_Oculi_Palpebral", "antagonist"),

        # Synergistic pairs (tend to co-activate)
        ("Zygomaticus_Major", "Orbicularis_Oculi_Orbital", "synergist"),  # Duchenne smile
        ("Corrugator_Supercilii", "Depressor_Supercilii", "synergist"),
        ("LLSAN", "Levator_Labii_Superioris", "synergist"),
        ("Orbicularis_Oris", "Mentalis", "synergist"),
        ("Depressor_Anguli_Oris", "Depressor_Labii_Inferioris", "synergist"),
    ]
    return [(MUSCLE_IDX[a], MUSCLE_IDX[b], rel) for a, b, rel in pairs]
