"""
BioReef.ai — Marine Biodiversity Monitoring System
===================================================
A three-stage deep learning pipeline for high-precision species detection,
tracking, and hierarchical classification in the Gulf of Oman.

Architecture:
    Stage 1: Context-Aware Spatial Detection (DINOv2 + MCEAM)
    Stage 2: Hybrid Spatiotemporal Tracking (ByteTrack + EMA Re-ID)
    Stage 3: Hierarchical Behavioral Classification (ViT-LSTM + HSLM)
"""

__version__ = "0.1.0"
__project__ = "BioReef.ai"
