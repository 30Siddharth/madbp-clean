"""Sampling utilities (DDIM with hard endpoint constraints).

The sampling code lives on the model itself (``MultiAgentDiffusionPlanner.sample``
and ``.sample_moving_window``); this submodule is reserved for future
guidance modules (e.g., CBF-as-classifier-guidance).
"""
