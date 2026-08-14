"""
Package `capture` - Capture Layer untuk project NetVision.

Menyediakan fungsi-fungsi untuk membaca interface jaringan, menangkap raw
packet, mengekstraksi fitur jaringan, dan merakit feature vector yang siap
dipakai untuk analisis atau model Machine Learning. Terintegrasi secara
dinamis dengan modul entropy.py untuk menghitung Shannon entropy tiap
payload yang tertangkap.

Import langsung dari package ini, misalnya:
    from capture import capture_packets, list_interfaces
"""

from .packet_capture import (
    FEATURE_ORDER,
    calculate_packet_rate,
    calculate_port_diversity,
    capture_packets,
    extract_features,
    extract_payload,
    list_interfaces,
    normalize_features,
    parse_packet,
)

__all__ = [
    "FEATURE_ORDER",
    "list_interfaces",
    "extract_payload",
    "parse_packet",
    "calculate_packet_rate",
    "calculate_port_diversity",
    "extract_features",
    "normalize_features",
    "capture_packets",
]