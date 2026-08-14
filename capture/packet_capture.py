"""
Capture Layer - project NetVision
===================================================================
Menangani pembacaan interface jaringan, penangkapan raw packet, ekstraksi
fitur-fitur jaringan (IP, port, ukuran paket, entropy payload), serta
perakitan feature vector yang siap dipakai untuk analisis atau model ML.

Terintegrasi secara dinamis dengan modul entropy.py (lihat root project):
tiap payload yang berhasil ditangkap dikirim ke fungsi entropy (default:
calculate_shannon_entropy) untuk dihitung Shannon entropy-nya - entropy
tinggi biasanya jadi indikator payload terenkripsi/mencurigakan.
`entropy_fn` sengaja dibuat sebagai parameter, bukan hardcode, supaya
implementasi entropy yang dipakai bisa diganti tanpa mengubah kode di sini.

Dependensi: scapy (`pip install scapy`), psutil (`pip install psutil`).
Menangkap paket (capture_packets) biasanya butuh privilege root/administrator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import psutil
from scapy.all import IP, TCP, UDP, Packet, sniff

# --- Integrasi dinamis dengan modul entropy.py di root project ---
# capture/packet_capture.py ada satu level di dalam folder capture/, jadi
# root project (tempat entropy.py berada) adalah satu level di atasnya.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from entropy import calculate_shannon_entropy  # noqa: E402


# Urutan fitur ini HARUS konsisten setiap kali dipanggil, supaya feature
# vector yang dihasilkan cocok dengan urutan yang dipakai saat training model.
FEATURE_ORDER: list[str] = [
    "total_paket",
    "unique_src_ip",
    "unique_dst_ip",
    "packet_rate",
    "port_diversity",
    "avg_packet_size",
    "tcp_ratio",
    "udp_ratio",
    "avg_entropy",
]


def list_interfaces() -> list[str]:
    """
    Mengembalikan daftar nama antarmuka jaringan yang tersedia di OS saat
    ini (cross-platform: Linux, Windows, maupun macOS), lewat psutil.
    """
    try:
        return list(psutil.net_if_addrs().keys())
    except Exception as e:
        print(f"[list_interfaces] Gagal membaca daftar interface: {e}")
        return []


def extract_payload(packet: Packet) -> bytes:
    """
    Mengisolasi raw byte payload dari layer TCP atau UDP sebuah paket,
    secara aman. Kalau paket tidak punya layer TCP/UDP, atau tidak ada
    data setelah header transport, mengembalikan bytes kosong (b"") -
    tidak pernah melempar exception ke pemanggilnya.
    """
    try:
        if packet.haslayer(TCP):
            return bytes(packet[TCP].payload)
        if packet.haslayer(UDP):
            return bytes(packet[UDP].payload)
    except Exception:
        pass
    return b""


def parse_packet(packet: Packet) -> dict[str, Any] | None:
    """
    Mengekstrak metadata (IP asal/tujuan, port asal/tujuan, protokol,
    ukuran paket) dari satu objek paket.

    Paket tak terstruktur - tidak punya layer IP (mis. ARP), atau gagal
    dibaca field-nya - ditangani secara graceful dengan mengembalikan
    None, supaya satu paket rusak tidak menghentikan seluruh capture.

    Catatan: versi ini hanya menangani IPv4. Traffic IPv6 (layer scapy
    `IPv6`) belum dicakup dan bisa ditambahkan sebagai pengembangan.
    """
    if not packet.haslayer(IP):
        return None

    try:
        ip_layer = packet[IP]
        info: dict[str, Any] = {
            "src_ip": ip_layer.src,
            "dst_ip": ip_layer.dst,
            "size": len(packet),
            "protocol": "OTHER",
            "src_port": None,
            "dst_port": None,
        }

        if packet.haslayer(TCP):
            info["protocol"] = "TCP"
            info["src_port"] = packet[TCP].sport
            info["dst_port"] = packet[TCP].dport
        elif packet.haslayer(UDP):
            info["protocol"] = "UDP"
            info["src_port"] = packet[UDP].sport
            info["dst_port"] = packet[UDP].dport

        return info
    except Exception:
        return None


def calculate_packet_rate(jumlah_paket: int, durasi_detik: float) -> float:
    """Mengkalkulasi jumlah paket per detik dalam jendela waktu penangkapan."""
    if durasi_detik <= 0:
        return 0.0
    return jumlah_paket / durasi_detik


def calculate_port_diversity(parsed_packets: list[dict[str, Any]]) -> float:
    """
    Menghitung persentase port tujuan unik dari total paket:

        port_diversity = Jumlah Port Tujuan Unik / Total Paket

    Rentang hasil 0-1, konsisten untuk kebutuhan fitur Machine Learning.
    """
    if not parsed_packets:
        return 0.0

    dst_ports = {p["dst_port"] for p in parsed_packets if p.get("dst_port") is not None}
    return len(dst_ports) / len(parsed_packets)


def extract_features(
    parsed_packets: list[dict[str, Any]],
    durasi_detik: float,
    entropies: list[float] | None = None,
) -> dict[str, float]:
    """
    Mengagregasi data dari seluruh paket yang berhasil di-parse menjadi
    kamus statistik jaringan: total paket, IP unik, packet rate, port
    diversity, ukuran paket rata-rata, rasio protokol, dan rata-rata
    entropy payload (dari `entropies`, hasil kiriman ke modul entropy.py).
    """
    total_paket = len(parsed_packets)

    if total_paket == 0:
        return {nama: 0.0 for nama in FEATURE_ORDER}

    unique_src_ip = len({p["src_ip"] for p in parsed_packets})
    unique_dst_ip = len({p["dst_ip"] for p in parsed_packets})
    avg_size = sum(p["size"] for p in parsed_packets) / total_paket
    tcp_count = sum(1 for p in parsed_packets if p["protocol"] == "TCP")
    udp_count = sum(1 for p in parsed_packets if p["protocol"] == "UDP")

    return {
        "total_paket": float(total_paket),
        "unique_src_ip": float(unique_src_ip),
        "unique_dst_ip": float(unique_dst_ip),
        "packet_rate": calculate_packet_rate(total_paket, durasi_detik),
        "port_diversity": calculate_port_diversity(parsed_packets),
        "avg_packet_size": avg_size,
        "tcp_ratio": tcp_count / total_paket,
        "udp_ratio": udp_count / total_paket,
        "avg_entropy": (sum(entropies) / len(entropies)) if entropies else 0.0,
    }


def normalize_features(features: dict[str, float]) -> list[float]:
    """
    Mengonversi dictionary statistik jaringan menjadi 1D list (feature
    vector) dengan urutan tetap (FEATURE_ORDER), siap diproses library ML
    seperti Scikit-learn atau River.
    """
    return [float(features.get(nama, 0.0)) for nama in FEATURE_ORDER]


def capture_packets(
    interface: str,
    durasi_detik: float = 10.0,
    entropy_fn: Callable[[bytes], float] = calculate_shannon_entropy,
) -> dict[str, Any]:
    """
    Mengorkestrasi seluruh capture layer: menangkap paket dari `interface`
    selama `durasi_detik` detik, mem-parsing tiap paket lewat parse_packet(),
    mengekstrak payload lewat extract_payload() dan mengirimnya ke
    `entropy_fn` untuk dihitung entropy-nya, lalu mengagregasi semuanya
    jadi feature dict dan feature vector siap pakai untuk ML.

    Paket yang gagal di-parse (parse_packet mengembalikan None) dilewati
    dengan aman, begitu juga kegagalan hitung entropy pada satu payload -
    keduanya tidak menghentikan proses capture secara keseluruhan.

    Mengembalikan dict:
        {
            "raw_packets": jumlah paket yang berhasil di-parse,
            "parsed_packets": list metadata tiap paket,
            "features": dict statistik jaringan,
            "feature_vector": list 1D siap dipakai model ML,
        }
    """
    parsed_packets: list[dict[str, Any]] = []
    entropies: list[float] = []

    def _proses_paket(packet: Packet) -> None:
        info = parse_packet(packet)
        if info is None:
            return  # paket tak terstruktur (mis. bukan IP) -> lewati dengan aman

        payload = extract_payload(packet)
        if payload:
            try:
                entropies.append(entropy_fn(payload))
            except Exception:
                pass  # kegagalan hitung entropy tidak boleh menghentikan capture

        parsed_packets.append(info)

    try:
        sniff(iface=interface, timeout=durasi_detik, prn=_proses_paket, store=False)
    except PermissionError as e:
        raise PermissionError(
            "Butuh privilege root/administrator untuk menangkap paket jaringan "
            "(coba jalankan ulang skrip dengan sudo)."
        ) from e

    fitur = extract_features(parsed_packets, durasi_detik, entropies)
    feature_vector = normalize_features(fitur)

    return {
        "raw_packets": len(parsed_packets),
        "parsed_packets": parsed_packets,
        "features": fitur,
        "feature_vector": feature_vector,
    }