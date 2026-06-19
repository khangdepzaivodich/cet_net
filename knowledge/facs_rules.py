"""
FACS Knowledge Base
Hardcoded domain knowledge about Action Unit relationships.

This defines:
1. An adjacency matrix encoding which AUs co-occur or are mutually exclusive.
2. AU-to-expression mappings (which AUs compose which expression).
3. Rule tuples for the rule violation loss.
"""

import torch

# AU ordering (must match config.py):
# 0:AU1, 1:AU2, 2:AU4, 3:AU6, 4:AU7, 5:AU10,
# 6:AU12, 7:AU14, 8:AU15, 9:AU17, 10:AU23, 11:AU24


def get_au_adjacency(num_aus=12):
    """
    Returns a [num_aus, num_aus] tensor with values in {-1, 0, +1}.
      +1 = these AUs tend to co-occur (reinforce each other)
      -1 = these AUs tend to be mutually exclusive (suppress each other)
       0 = no strong known relationship
    """
    adj = torch.zeros(num_aus, num_aus)

    # --- Co-occurring pairs (+1) ---
    # AU1 + AU2: inner + outer brow raise (often appear together in surprise)
    adj[0, 1] = 1; adj[1, 0] = 1
    # AU1 + AU4: brow raise + brow lowerer (can co-occur in fear)
    adj[0, 2] = 1; adj[2, 0] = 1
    # AU6 + AU12: cheek raiser + lip corner puller (Duchenne smile / happy)
    adj[3, 6] = 1; adj[6, 3] = 1
    # AU6 + AU7: cheek raiser + lid tightener (often co-occur)
    adj[3, 4] = 1; adj[4, 3] = 1
    # AU10 + AU17: upper lip raiser + chin raiser (disgust)
    adj[5, 9] = 1; adj[9, 5] = 1
    # AU23 + AU24: lip tightener + lip pressor (often co-occur in anger)
    adj[10, 11] = 1; adj[11, 10] = 1
    # AU4 + AU7: brow lowerer + lid tightener (anger)
    adj[2, 4] = 1; adj[4, 2] = 1

    # --- Mutually exclusive pairs (-1) ---
    # AU12 (lip corner puller / smile) vs AU15 (lip corner depressor / frown)
    adj[6, 8] = -1; adj[8, 6] = -1
    # AU12 (smile) vs AU23 (lip tightener)
    adj[6, 10] = -1; adj[10, 6] = -1
    # AU12 (smile) vs AU24 (lip pressor)
    adj[6, 11] = -1; adj[11, 6] = -1
    # AU1 (inner brow raise) vs AU4 tends to suppress AU2 (outer brow raise)
    # in sadness context, but this is weaker so we leave it as 0

    return adj


def get_au_expression_map(num_aus=12, num_expressions=7):
    """
    Returns a [num_expressions, num_aus] binary matrix.
    Entry [expr, au] = 1 if that AU is a defining component of that expression.

    Expressions: 0:Happy, 1:Sad, 2:Fear, 3:Disgust, 4:Anger, 5:Surprise, 6:Neutral
    """
    mapping = torch.zeros(num_expressions, num_aus)

    # Happy: AU6 (cheek raiser) + AU12 (lip corner puller)
    mapping[0, 3] = 1; mapping[0, 6] = 1

    # Sad: AU1 (inner brow raise) + AU4 (brow lowerer) + AU15 (lip corner depressor)
    mapping[1, 0] = 1; mapping[1, 2] = 1; mapping[1, 8] = 1

    # Fear: AU1 (inner brow raise) + AU2 (outer brow raise) + AU4 (brow lowerer) + AU7 (lid tightener)
    mapping[2, 0] = 1; mapping[2, 1] = 1; mapping[2, 2] = 1; mapping[2, 4] = 1

    # Disgust: AU4 (brow lowerer) + AU10 (upper lip raiser) + AU17 (chin raiser)
    mapping[3, 2] = 1; mapping[3, 5] = 1; mapping[3, 9] = 1

    # Anger: AU4 (brow lowerer) + AU7 (lid tightener) + AU23 (lip tightener) + AU24 (lip pressor)
    mapping[4, 2] = 1; mapping[4, 4] = 1; mapping[4, 10] = 1; mapping[4, 11] = 1

    # Surprise: AU1 (inner brow raise) + AU2 (outer brow raise)
    mapping[5, 0] = 1; mapping[5, 1] = 1

    # Neutral: no AUs strongly active (all zeros)

    return mapping


def get_rules():
    """
    Returns a list of rule tuples for the rule violation loss.
    Each rule is: (condition_au_index, expected_au_index, direction)
      direction = +1: if condition AU is active, expected AU should also be active
      direction = -1: if condition AU is active, expected AU should NOT be active

    These are used to compute per-sample rule violations.
    """
    rules = [
        # If AU6 active -> AU12 should be active (Duchenne smile)
        (3, 6, +1),
        # If AU12 active -> AU6 should be active
        (6, 3, +1),
        # If AU12 active -> AU15 should NOT be active (can't smile and frown)
        (6, 8, -1),
        # If AU12 active -> AU23 should NOT be active
        (6, 10, -1),
        # If AU1 active and AU2 active -> surprise context (reinforcement)
        (0, 1, +1),
        (1, 0, +1),
        # If AU4 active -> AU7 tends to co-occur (anger/fear)
        (2, 4, +1),
        # If AU23 active -> AU24 should co-occur
        (10, 11, +1),
        (11, 10, +1),
        # If AU10 active -> AU17 should co-occur (disgust)
        (5, 9, +1),
        (9, 5, +1),
    ]
    return rules
