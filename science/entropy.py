"""
Analisis Payload - Shannon Entropy & Deteksi Anomali via Z-score
==================================================================
Bagian dari proyek NIDS (Network Intrusion Detection System).

Payload dengan entropy tinggi (mendekati 8 bit/byte) sering menandakan
data terenkripsi, terkompresi, atau ter-obfuscate - pola umum pada
traffic berbahaya (malware packing, C2 channel, data exfiltration).

Z-score dipakai untuk melihat seberapa jauh entropy sebuah payload
menyimpang dari baseline (kumpulan entropy payload normal yang sudah
diamati sebelumnya, misalnya dari histori traffic yang disimpan di DB).
"""

import math
import statistics
from collections import Counter
from pathlib import Path


def read_payload(source: str | bytes) -> bytes:
    """
    Membaca payload dan mengembalikannya sebagai bytes.

    - Jika `source` sudah berupa bytes -> dikembalikan langsung.
    - Jika `source` adalah path ke file yang ada -> baca isi file.
    - Jika `source` berupa string biasa (mis. payload hasil capture
      yang sudah didekode) -> di-encode ke UTF-8.
    """
    if isinstance(source, bytes):
        return source

    path = Path(source)
    if path.is_file():
        return path.read_bytes()

    return source.encode("utf-8", errors="ignore")


def calculate_shannon_entropy(payload: bytes) -> float:
    """
    Menghitung Shannon Entropy payload (satuan bit/byte, rentang 0-8).

        H(X) = -sum( p(x) * log2(p(x)) )   untuk tiap nilai byte unik x

    Entropy mendekati 8   -> data terlihat acak (terenkripsi/terkompresi)
    Entropy rendah        -> data berpola/repetitif (mis. teks biasa)
    """
    if not payload:
        return 0.0

    length = len(payload)
    frequency = Counter(payload)

    entropy = 0.0
    for count in frequency.values():
        p_x = count / length
        entropy -= p_x * math.log2(p_x)

    return entropy


def calculate_z_score(value: float, baseline: list[float]) -> float:
    """
    Menghitung Z-score sebuah nilai terhadap baseline.

        Z = (value - mean(baseline)) / stdev(baseline)

    `baseline` adalah kumpulan nilai entropy dari payload-payload
    normal/historis yang sudah diamati (baseline traffic).
    """
    if len(baseline) < 2:
        return 0.0

    mean = statistics.mean(baseline)
    stdev = statistics.stdev(baseline)

    if stdev == 0:
        return 0.0

    return (value - mean) / stdev


def get_status(z_score: float, threshold: float = 2.5) -> str:
    """
    Memberi status berdasarkan Z-score: ANOMALI jika z_score > threshold.

    (Ganti jadi `abs(z_score) > threshold` kalau entropy yang jauh LEBIH
    RENDAH dari baseline juga ingin ditandai sebagai anomali.)
    """
    return "ANOMALI" if z_score > threshold else "NORMAL"


def analyze_payload(source: str | bytes, baseline: list[float], threshold: float = 2.5) -> dict:
    """Pipeline lengkap: baca payload -> hitung entropy -> Z-score -> status."""
    payload = read_payload(source)
    entropy = calculate_shannon_entropy(payload)
    z_score = calculate_z_score(entropy, baseline)
    status = get_status(z_score, threshold)

    return {
        "panjang_payload": len(payload),
        "entropy": round(entropy, 4),
        "z_score": round(z_score, 4),
        "status": status,
    }


if __name__ == "__main__":
    # Baseline: entropy dari 6 payload HTTP "normal" yang sudah diamati
    # sebelumnya (dalam praktiknya bisa diambil dari tabel histori di DB)
    baseline_entropies = [4.549, 4.490, 4.727, 4.493, 4.531, 4.532]

    contoh = {
        "Payload normal (HTTP request baru)":
            b"GET /contact.html HTTP/1.1\r\nHost: example.com\r\n\r\n",
        "Payload mencurigakan (byte acak/terenkripsi)":
            bytes(range(256)) * 4,
    }

    for nama, payload in contoh.items():
        hasil = analyze_payload(payload, baseline_entropies)
        print(f"\n{nama}")
        print(f"  Panjang : {hasil['panjang_payload']} bytes")
        print(f"  Entropy : {hasil['entropy']} bit/byte")
        print(f"  Z-score : {hasil['z_score']}")
        print(f"  Status  : {hasil['status']}")