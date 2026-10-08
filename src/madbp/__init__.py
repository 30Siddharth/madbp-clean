"""MA-DBP: Multi-Agent Diffusion-Based Planner.

A single shared diffusion model for continuous multi-robot navigation that
generalizes from small training teams to large deployment teams via
inter-agent axial attention and a moving-window rollout.

Paper: arXiv:2604.06598 (Train-Small Deploy-Large)
"""

__version__ = "0.1.0"

__all__ = ["MultiAgentDiffusionPlanner", "__version__"]


def __getattr__(name):  # lazy: keep `madbp.sim` importable without torch
    if name == "MultiAgentDiffusionPlanner":
        from madbp.models.diffusion import MultiAgentDiffusionPlanner

        return MultiAgentDiffusionPlanner
    raise AttributeError(name)
