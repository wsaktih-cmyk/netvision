"""
AI & XAI Layer package untuk NetVision.

Menyediakan model deteksi intrusi berbasis AI (NetVisionAI) dan
komponen explainability/XAI (NetVisionExplainer) untuk menjelaskan
hasil deteksi tersebut secara transparan.
"""

from .models import NetVisionAI
from .explainer import NetVisionExplainer, FEATURE_NAMES

__all__ = [
    "NetVisionAI",
    "NetVisionExplainer",
    "FEATURE_NAMES",
]