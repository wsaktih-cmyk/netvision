"""
Simulasi Penyebaran Malware - Model SIR (Susceptible-Infected-Recovered)
===========================================================================
Modul simulasi penyebaran malware untuk project NetVision.

    S (Susceptible) -> perangkat rentan: belum terinfeksi, tapi bisa tertular
    I (Infected)     -> perangkat yang sudah terinfeksi malware
    R (Recovered)    -> perangkat yang sudah dipulihkan atau diisolasi dari
                         jaringan (tidak lagi bisa menulari/tertulari)

Model SIR diskrit (per hari), memakai metode Euler dengan langkah waktu 1 hari:

    dS = -beta * S * I / N
    dI =  beta * S * I / N  -  gamma * I
    dR =  gamma * I

    N     -> total perangkat (S + I + R), diasumsikan konstan sepanjang
             simulasi (tidak ada perangkat baru masuk/keluar jaringan)
    beta  -> tingkat penyebaran: seberapa mudah malware menular antar
             perangkat per hari
    gamma -> tingkat pemulihan/isolasi: proporsi perangkat terinfeksi yang
             dipulihkan atau diisolasi dari jaringan per hari
"""


def calculate_change(S: float, I: float, R: float, beta: float, gamma: float) -> tuple[float, float, float]:
    """
    Menghitung perubahan (delta) jumlah perangkat rentan (S), terinfeksi (I),
    dan dipulihkan/diisolasi (R) untuk satu langkah waktu (1 hari), mengikuti
    model SIR:

        dS = -beta * S * I / N
        dI =  beta * S * I / N - gamma * I
        dR =  gamma * I

    N (total perangkat) dihitung sebagai S + I + R.
    Mengembalikan tuple (dS, dI, dR).
    """
    N = S + I + R
    if N == 0:
        return 0.0, 0.0, 0.0

    infeksi_baru = beta * S * I / N
    pemulihan_baru = gamma * I

    dS = -infeksi_baru
    dI = infeksi_baru - pemulihan_baru
    dR = pemulihan_baru

    return dS, dI, dR


def simulate_sir(S0: float, I0: float, R0: float, beta: float, gamma: float, hari: int) -> dict[str, list[float]]:
    """
    Mensimulasikan penyebaran malware dari hari ke hari dengan model SIR.

    Parameter:
        S0, I0, R0 -> jumlah perangkat rentan/terinfeksi/dipulihkan di hari ke-0
        beta       -> tingkat penyebaran
        gamma      -> tingkat pemulihan/isolasi
        hari       -> jumlah hari yang disimulasikan

    Tiap hari, delta dihitung lewat calculate_change() lalu ditambahkan ke
    S, I, R (metode Euler). Hasil di-clamp ke >= 0 untuk jaga-jaga dari
    error pembulatan.

    Mengembalikan dict berisi list "hari", "S", "I", "R" (satu nilai per
    hari, termasuk hari ke-0) - siap dipakai untuk analisis lanjutan atau
    digambar jadi grafik (mis. dengan matplotlib).
    """
    S, I, R = float(S0), float(I0), float(R0)
    hasil = {"hari": [0], "S": [S], "I": [I], "R": [R]}

    for t in range(1, hari + 1):
        dS, dI, dR = calculate_change(S, I, R, beta, gamma)

        S = max(S + dS, 0.0)
        I = max(I + dI, 0.0)
        R = max(R + dR, 0.0)

        hasil["hari"].append(t)
        hasil["S"].append(S)
        hasil["I"].append(I)
        hasil["R"].append(R)

    return hasil


if __name__ == "__main__":
    # --- Contoh pemakaian ---
    total_perangkat = 1000

    hasil = simulate_sir(
        S0=total_perangkat - 1,   # semua rentan kecuali 1 perangkat awal terinfeksi
        I0=1,
        R0=0,
        beta=0.4,     # tingkat penyebaran
        gamma=0.1,    # tingkat pemulihan/isolasi
        hari=60,
    )

    print(f"{'Hari':>4} | {'S':>8} | {'I':>8} | {'R':>8}")
    for t, S, I, R in zip(hasil["hari"], hasil["S"], hasil["I"], hasil["R"]):
        if t % 5 == 0:  # tampilkan tiap 5 hari biar ringkas
            print(f"{t:>4} | {S:8.1f} | {I:8.1f} | {R:8.1f}")

    puncak_idx = hasil["I"].index(max(hasil["I"]))
    print(f"\nPuncak infeksi: hari ke-{hasil['hari'][puncak_idx]}, "
            f"{hasil['I'][puncak_idx]:.1f} perangkat terinfeksi")