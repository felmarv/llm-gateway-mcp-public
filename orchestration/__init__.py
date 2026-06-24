"""Multi-role orchestration inspired by two Sakana AI papers (ICLR 2026):

  - TRINITY: An Evolved LLM Coordinator (arXiv:2512.04695)
  - Learning to Orchestrate Agents in Natural Language with the Conductor
    (arXiv:2512.04388)

What this package implements (provider-agnostic, fully functional):

  independence.py  - access_list / visibility certification (Conductor T-02):
                     for parallel multi-model flows, prove each member saw ONLY
                     the original prompt (independence_certified).
  roles.py         - Thinker / Worker / Verifier instruction templates and a
                     configurable role -> model aptitude table (Trinity T-03).
  compose.py       - plan -> execute -> verify pipeline with per-step visibility
                     (the verifier does NOT see the plan) and failure-gated
                     re-planning (cap 1 round); depth graduates the step count.

The papers optimize SYNERGY (workers reading each other to converge). Parallel
review here optimizes the OPPOSITE — INDEPENDENCE — so that agreement between
models is *evidence*. The two planes are kept separate on purpose:
  - blind parallel panels  -> independence enforced (independence.py)
  - sequential compose     -> controlled visibility per step (compose.py)
"""
