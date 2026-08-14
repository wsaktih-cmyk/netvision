"""
=============================================================
  Network Packet Sniffer & Feature Extractor (Terintegrasi)
  Langkah 3 - Data Collection Layer untuk AI/ML Pipeline
=============================================================
  Fitur:
  - Deteksi network interface secara dinamis
  - Sniffing paket selama 10 detik
  - Ekstraksi 23 fitur (termasuk avg_entropy payload)
  - Integrasi dengan modul science.entropy (Langkah 2)
  - Output feature vector (array 1D) siap untuk ML model
=============================================================
"""

import sys
import time
import ctypes
import platform
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ─────────────────────────────────────────────
#  Import Scapy
# ─────────────────────────────────────────────
try:
    from scapy.all import sniff, get_if_list, conf
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.packet import Raw
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# ─────────────────────────────────────────────
#  Integrasi Modul Entropy (Langkah 2)
# ─────────────────────────────────────────────
# Menambahkan root project ke sys.path agar bisa import modul 'science'
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Asumsi fungsi bernama calculate_entropy di dalam science/entropy.py
    # Sesuaikan nama fungsi import ini dengan kodemu yang sebenarnya
    from science.entropy import calculate_entropy
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False
    print("\n  [WARN] Modul 'science.entropy' tidak ditemukan!")
    print("  Pastikan script dijalankan dari root direktori 'netvision'.")
    print("  (Kalkulasi entropy akan di-bypass dengan nilai 0.0)\n")

    # Fallback dummy fungsi jika gagal import
    def calculate_entropy(payload: bytes) -> float:
        return 0.0


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — PRIVILEGE CHECK
# ══════════════════════════════════════════════════════════════

def check_privileges() -> bool:
    current_os = platform.system()
    try:
        if current_os == "Windows":
            if not ctypes.windll.shell32.IsUserAnAdmin():
                raise PermissionError("  [!] AKSES DITOLAK - Jalankan CMD/PowerShell as Administrator.")
        else:
            import os
            if os.geteuid() != 0:
                raise PermissionError("  [!] AKSES DITOLAK - Gunakan 'sudo' untuk menjalankan script.")
        print("  [OK] Hak akses mencukupi.")
        return True
    except PermissionError as e:
        print(e)
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — DETEKSI INTERFACE JARINGAN
# ══════════════════════════════════════════════════════════════

def list_interfaces() -> list:
    print("\n" + "=" * 60)
    print("  [2] DETEKSI NETWORK INTERFACE")
    print("=" * 60)

    if not SCAPY_AVAILABLE:
        print("  [ERROR] Scapy tidak terinstall. Jalankan: pip install scapy")
        sys.exit(1)

    try:
        interfaces = get_if_list()
        if not interfaces:
            print("  [WARN] Tidak ada interface yang terdeteksi.")
            sys.exit(1)

        print(f"  Ditemukan {len(interfaces)} interface jaringan:\n")
        for idx, iface in enumerate(interfaces, start=1):
            tag = " <-- [AKAN DIGUNAKAN]" if idx == 1 else ""
            print(f"    [{idx}] {iface}{tag}")

        print("\n" + "-" * 60)
        return interfaces
    except Exception as e:
        print(f"  [ERROR] Gagal membaca interface: {e}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — CAPTURE & SNIFFING PAKET
# ══════════════════════════════════════════════════════════════

class PacketCollector:
    def __init__(self):
        self.packets     = []
        self.start_time  = None
        self.total_bytes = 0

    def packet_handler(self, packet) -> None:
        self.packets.append(packet)
        self.total_bytes += len(packet)

        count = len(self.packets)
        if count % 10 == 0 or count <= 5:
            elapsed = time.time() - self.start_time
            print(f"  [>>] Paket tertangkap: {count:>5} | Elapsed: {elapsed:.1f}s      ", end="\r")

def capture_packets(interface: str, duration: int = 10) -> PacketCollector:
    print(f"\n{'=' * 60}")
    print(f"  [3] MEMULAI SNIFFING PAKET")
    print(f"{'=' * 60}")
    print(f"  Interface : {interface}")
    print(f"  Durasi    : {duration} detik")
    print(f"  Mulai     : {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'-' * 60}")
    print(f"  Menangkap paket... (Ctrl+C untuk berhenti lebih awal)\n")

    collector = PacketCollector()
    collector.start_time = time.time()

    try:
        sniff(iface=interface, prn=collector.packet_handler, timeout=duration, store=False)
    except OSError as e:
        print(f"\n\n  [ERROR] Gagal membuka interface '{interface}': {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n  [INFO] Sniffing dihentikan lebih awal oleh pengguna.")

    elapsed = time.time() - collector.start_time
    print(f"\n{'-' * 60}")
    print(f"  [OK] Sniffing selesai!")
    print(f"  Total paket : {len(collector.packets):,}")
    print(f"  Total bytes : {collector.total_bytes:,} bytes")
    print(f"  Durasi real : {elapsed:.2f} detik")
    print(f"{'=' * 60}")

    return collector


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — EKSTRAKSI FITUR & ENTROPY PAYLOAD
# ══════════════════════════════════════════════════════════════

def extract_payload(pkt) -> bytes:
    """Mengisolasi raw payload dari paket jaringan secara aman."""
    if pkt.haslayer(Raw):
        return bytes(pkt[Raw].load)
    return b""

def extract_features(collector: PacketCollector, duration: int = 10) -> dict:
    packets = collector.packets
    n       = len(packets)

    if n == 0:
        print("\n  [WARN] Tidak ada paket tertangkap. Feature vector diisi nol.")
        return defaultdict(float) # Return zeros for all keys

    src_ips, dst_ips       = set(), set()
    src_ports, dst_ports   = set(), set()
    tcp_count, udp_count   = 0, 0
    icmp_count, other      = 0, 0
    syn_count, fin_count   = 0, 0
    rst_count              = 0
    pkt_sizes              = []
    entropies              = [] # List untuk menyimpan nilai entropy payload

    for pkt in packets:
        pkt_sizes.append(len(pkt))
        
        # --- Ekstraksi & Kalkulasi Entropy ---
        payload = extract_payload(pkt)
        if payload:
            try:
                # Menggunakan fungsi dari science/entropy.py
                ent_val = calculate_entropy(payload)
                entropies.append(ent_val)
            except Exception:
                pass

        # --- Parsing IP & Protocol ---
        if IP in pkt:
            src_ips.add(pkt[IP].src)
            dst_ips.add(pkt[IP].dst)

            if TCP in pkt:
                tcp_count += 1
                src_ports.add(pkt[TCP].sport)
                dst_ports.add(pkt[TCP].dport)
                flags = pkt[TCP].flags
                if flags & 0x02: syn_count += 1
                if flags & 0x01: fin_count += 1
                if flags & 0x04: rst_count += 1

            elif UDP in pkt:
                udp_count += 1
                src_ports.add(pkt[UDP].sport)
                dst_ports.add(pkt[UDP].dport)

            elif ICMP in pkt:
                icmp_count += 1
            else:
                other += 1
        else:
            other += 1

    # Kalkulasi Metrik Akhir
    total_bytes    = collector.total_bytes
    packet_rate    = n / duration
    avg_pkt_size   = sum(pkt_sizes) / n if n > 0 else 0.0
    bytes_per_sec  = total_bytes / duration
    tcp_udp_total  = tcp_count + udp_count
    
    port_diversity = (len(src_ports) + len(dst_ports)) / (2 * tcp_udp_total) if tcp_udp_total > 0 else 0.0
    ip_diversity   = (len(src_ips) + len(dst_ips)) / (2 * n)
    
    # Kalkulasi rata-rata entropy (jika tidak ada payload terdeteksi, set 0)
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

    features = {
        "packet_count"       : n,
        "packet_rate"        : round(packet_rate, 4),
        "total_bytes"        : total_bytes,
        "avg_packet_size"    : round(avg_pkt_size, 4),
        "unique_src_ips"     : len(src_ips),
        "unique_dst_ips"     : len(dst_ips),
        "unique_src_ports"   : len(src_ports),
        "unique_dst_ports"   : len(dst_ports),
        "port_diversity"     : round(port_diversity, 4),
        "tcp_count"          : tcp_count,
        "udp_count"          : udp_count,
        "icmp_count"         : icmp_count,
        "other_count"        : other,
        "tcp_ratio"          : round(tcp_count / n, 4),
        "udp_ratio"          : round(udp_count / n, 4),
        "icmp_ratio"         : round(icmp_count / n, 4),
        "syn_count"          : syn_count,
        "fin_count"          : fin_count,
        "rst_count"          : rst_count,
        "syn_ratio"          : round(syn_count / n, 4),
        "bytes_per_second"   : round(bytes_per_sec, 4),
        "ip_diversity_ratio" : round(ip_diversity, 4),
        "avg_entropy"        : round(avg_entropy, 4) # FITUR BARU HASIL INTEGRASI
    }

    return features


def print_features(features: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  [4] VALIDASI EKSTRAKSI FITUR (features dictionary)")
    print(f"{'=' * 60}")

    for key, value in features.items():
        if isinstance(value, float):
            print(f"    {key:<25} : {value:>12.4f}")
        else:
            print(f"    {key:<25} : {value:>12,}")
    print(f"\n{'=' * 60}")


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — FEATURE VECTOR (OUTPUT UNTUK ML MODEL)
# ══════════════════════════════════════════════════════════════

def build_feature_vector(features: dict) -> np.ndarray:
    FEATURE_ORDER = [
        "packet_count", "packet_rate", "total_bytes", "avg_packet_size",
        "unique_src_ips", "unique_dst_ips", "unique_src_ports", "unique_dst_ports",
        "port_diversity", "tcp_count", "udp_count", "icmp_count", "other_count",
        "tcp_ratio", "udp_ratio", "icmp_ratio", "syn_count", "fin_count", "rst_count",
        "syn_ratio", "bytes_per_second", "ip_diversity_ratio", 
        "avg_entropy" # [22] Rata-rata Shannon Entropy (Integrasi Langkah 2)
    ]

    vector = np.array([features.get(key, 0.0) for key in FEATURE_ORDER], dtype=np.float64)

    print(f"\n{'=' * 60}")
    print(f"  [5] FEATURE VECTOR - SIAP UNTUK ML MODEL (Langkah 4)")
    print(f"{'=' * 60}")
    print(f"  Shape : {vector.shape}  ({len(vector)} dimensi fitur)")
    print(f"  dtype : {vector.dtype}")
    print(f"\n  [Raw Array Output]:")
    print(f"  {np.array2string(vector, precision=4, separator=', ')}")
    print(f"{'=' * 60}")

    return vector


# ══════════════════════════════════════════════════════════════
#  MAIN — ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("   NETWORK PACKET SNIFFER & FEATURE EXTRACTOR")
    print("   AI/ML Pipeline --- Data Collection Layer (Langkah 3)")
    print("=" * 60)
    
    check_privileges()

    interfaces = list_interfaces()
    target_interface = conf.iface.name
    
    SNIFF_DURATION = 10
    collector = capture_packets(interface=target_interface, duration=SNIFF_DURATION)

    features = extract_features(collector, duration=SNIFF_DURATION)
    print_features(features)
    feature_vector = build_feature_vector(features)

    print(f"\n{'=' * 60}")
    print(f"  [DONE] Pipeline Langkah 3 BERHASIL TERINTEGRASI!")
    print(f"  Status Entropy : {'Aktif (science.entropy)' if ENTROPY_AVAILABLE else 'Bypass (0.0)'}")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    main()