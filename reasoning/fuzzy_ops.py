"""
Differentiable Fuzzy Logic Operations

Implements product t-norm fuzzy logic, which is fully differentiable
and suitable for gradient-based learning.

Product T-norm:
  AND(a, b) = a * b
  OR(a, b)  = a + b - a * b
  NOT(a)    = 1 - a
  IMPLY(a, b) = 1 - a * (1 - b)    (equivalent to NOT(a AND NOT(b)))

All inputs and outputs are in [0, 1].
All operations are differentiable w.r.t. inputs.
"""

import torch


def fuzzy_and(a, b):
    """Product t-norm AND: a * b"""
    return a * b


def fuzzy_or(a, b):
    """Product t-conorm OR: a + b - a*b"""
    return a + b - a * b


def fuzzy_not(a):
    """Standard fuzzy negation: 1 - a"""
    return 1.0 - a


def fuzzy_imply(a, b):
    """
    Fuzzy implication (Reichenbach): 1 - a*(1-b)
    If a is true and b is false, satisfaction is low.
    If a is false, rule is trivially satisfied.
    """
    return 1.0 - a * (1.0 - b)


def fuzzy_and_multi(*args):
    """AND over multiple inputs: product of all."""
    result = args[0]
    for a in args[1:]:
        result = fuzzy_and(result, a)
    return result


def fuzzy_or_multi(*args):
    """OR over multiple inputs."""
    result = args[0]
    for a in args[1:]:
        result = fuzzy_or(result, a)
    return result


def fuzzy_rule_satisfaction(condition, consequent):
    """
    Compute satisfaction of rule: condition → consequent
    Returns a scalar satisfaction score in [0, 1].
    Higher = rule is more satisfied.

    Args:
        condition: tensor of truth values [B] or [B, N]
        consequent: tensor of truth values [B] or [B, N]

    Returns:
        satisfaction score, same shape as inputs
    """
    return fuzzy_imply(condition, consequent)


def fuzzy_antagonist_penalty(a, b):
    """
    Penalty for two antagonistic muscles being active simultaneously.
    High when both a and b are high. Returns a*b (should be minimized).

    Args:
        a, b: muscle activation values in [0, 1]

    Returns:
        penalty score (higher = worse)
    """
    return fuzzy_and(a, b)


def fuzzy_synergist_reward(a, b):
    """
    Reward for two synergistic muscles co-activating.
    Returns how well a and b agree (both high or both low).

    Uses: 1 - |a - b| (agreement score)

    Args:
        a, b: muscle activation values in [0, 1]

    Returns:
        reward score (higher = better agreement)
    """
    return 1.0 - torch.abs(a - b)
