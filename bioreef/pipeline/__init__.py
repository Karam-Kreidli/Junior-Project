"""
BioReef.ai pipeline primitives.

Stage-level inference functions that scripts orchestrate. Keeping them in the
library (rather than in the CLI scripts) means scripts never import from each
other — they all import from here.
"""
