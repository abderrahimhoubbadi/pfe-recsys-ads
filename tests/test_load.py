"""
Load Test — Measures API throughput and latency under increasing load.

Tests the FastAPI recommendation endpoint at different concurrency levels
(10, 25, 50, 100 requests/second) and produces a latency vs concurrency plot.
"""

import time
import statistics
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

# ── Configure matplotlib for headless ──
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "figure.dpi": 150,
    }
)


@dataclass
class LoadTestResult:
    """Result of a single load test tier."""

    concurrency: int
    total_requests: int
    successful: int
    failed: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float


def run_single_request(service, ads, user_texts):
    """Execute one recommendation request and return latency."""
    import random

    user = random.choice(user_texts)
    t0 = time.perf_counter()
    try:
        ad_id, eng, rev, lat = service.recommend(user_text=user, available_ads=ads)
        return (time.perf_counter() - t0) * 1000, True
    except Exception as e:
        return (time.perf_counter() - t0) * 1000, False


def run_load_test(concurrency_levels: List[int] = None, requests_per_level: int = 100):
    """Run load tests at increasing concurrency levels."""
    from src.api.recommendation_service import RecommendationService
    from src.api.schemas import AdInfo

    if concurrency_levels is None:
        concurrency_levels = [1, 5, 10, 25, 50]

    print("=" * 65)
    print(" Load Test — API Performance Under Concurrent Load")
    print("=" * 65)

    # ── Initialize ──
    print("\n[Init] Starting RecommendationService...")
    service = RecommendationService(
        embedding_model="all-MiniLM-L6-v2",
        alpha=1.0,
        n_ensemble=5,
        epsilon=0.3,
    )

    ads = [
        AdInfo(
            ad_id=1,
            title="RTX 5090 GPU",
            description="Carte graphique NVIDIA gaming",
            category="tech",
        ),
        AdInfo(
            ad_id=2,
            title="Cours Python",
            description="Formation deep learning et IA",
            category="éducation",
        ),
        AdInfo(
            ad_id=3,
            title="Voyage Marrakech",
            description="Séjour tout inclus riad luxe",
            category="voyage",
        ),
        AdInfo(
            ad_id=4,
            title="MacBook Pro M4",
            description="Portable Apple développeurs",
            category="tech",
        ),
        AdInfo(
            ad_id=5,
            title="Abonnement Netflix",
            description="Films et séries streaming",
            category="divertissement",
        ),
        AdInfo(
            ad_id=6,
            title="PS5 Pro",
            description="Console dernière génération",
            category="gaming",
        ),
    ]
    service.register_ads(ads)

    user_texts = [
        "homme 28 ans développeur passionné de technologie",
        "femme 35 ans directrice marketing lifestyle",
        "étudiant 22 ans informatique formations en ligne",
        "couple 40 ans vacances au Maroc",
        "homme 50 ans chef d'entreprise solutions B2B",
    ]

    print(f"   Service ready, {len(ads)} ads registered\n")

    # ── Run tests ──
    results: List[LoadTestResult] = []

    for conc in concurrency_levels:
        print(
            f"[Load] Concurrency={conc:>3d} | {requests_per_level} requests ...",
            end=" ",
            flush=True,
        )

        latencies = []
        successes = 0
        failures = 0

        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=conc) as pool:
            futures = [
                pool.submit(run_single_request, service, ads, user_texts)
                for _ in range(requests_per_level)
            ]
            for f in as_completed(futures):
                lat_ms, ok = f.result()
                latencies.append(lat_ms)
                if ok:
                    successes += 1
                else:
                    failures += 1

        total_time = time.perf_counter() - t0
        latencies.sort()

        result = LoadTestResult(
            concurrency=conc,
            total_requests=requests_per_level,
            successful=successes,
            failed=failures,
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=latencies[len(latencies) // 2],
            p95_latency_ms=latencies[int(len(latencies) * 0.95)],
            p99_latency_ms=latencies[int(len(latencies) * 0.99)],
            requests_per_second=requests_per_level / total_time,
        )
        results.append(result)

        print(
            f"avg={result.avg_latency_ms:>6.1f}ms | "
            f"P95={result.p95_latency_ms:>6.1f}ms | "
            f"RPS={result.requests_per_second:>5.1f} | "
            f"errors={result.failed}"
        )

    return results


def generate_load_plot(results: List[LoadTestResult], output_dir: str = "metrics"):
    """Generate load test visualization."""
    os.makedirs(output_dir, exist_ok=True)

    concs = [r.concurrency for r in results]
    avgs = [r.avg_latency_ms for r in results]
    p95s = [r.p95_latency_ms for r in results]
    p99s = [r.p99_latency_ms for r in results]
    rps = [r.requests_per_second for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Load Test — H-DeepBandit API Performance", fontweight="bold", fontsize=13
    )

    # Latency vs Concurrency
    ax1.plot(
        concs, avgs, "o-", color="#2c3e50", linewidth=2, markersize=6, label="Moyenne"
    )
    ax1.plot(
        concs, p95s, "s--", color="#e67e22", linewidth=1.5, markersize=5, label="P95"
    )
    ax1.plot(
        concs, p99s, "^:", color="#e74c3c", linewidth=1.5, markersize=5, label="P99"
    )
    ax1.axhline(y=100, color="#95a5a6", linestyle=":", label="SLA (100ms)")
    ax1.set_xlabel("Concurrence (threads)")
    ax1.set_ylabel("Latence (ms)")
    ax1.set_title("Latence vs Concurrence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Throughput
    ax2.bar(range(len(concs)), rps, color="#27ae60", alpha=0.8, width=0.6)
    ax2.set_xticks(range(len(concs)))
    ax2.set_xticklabels([str(c) for c in concs])
    ax2.set_xlabel("Concurrence (threads)")
    ax2.set_ylabel("Requêtes/seconde")
    ax2.set_title("Débit (RPS)")
    ax2.grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(rps):
        ax2.text(
            i,
            v + max(rps) * 0.02,
            f"{v:.0f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "load_test_latency.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   📊 Saved: {path}")


if __name__ == "__main__":
    results = run_load_test(
        concurrency_levels=[1, 5, 10, 25, 50],
        requests_per_level=100,
    )

    generate_load_plot(results)

    print("\n" + "=" * 65)
    print(" ✅ LOAD TEST COMPLETE")
    print("=" * 65)

    # Summary table
    print(
        f"\n   {'Conc':>6s} | {'Avg(ms)':>8s} | {'P95(ms)':>8s} | {'P99(ms)':>8s} | {'RPS':>6s} | {'Errors':>6s}"
    )
    print("   " + "-" * 55)
    for r in results:
        print(
            f"   {r.concurrency:>6d} | {r.avg_latency_ms:>8.1f} | "
            f"{r.p95_latency_ms:>8.1f} | {r.p99_latency_ms:>8.1f} | "
            f"{r.requests_per_second:>6.1f} | {r.failed:>6d}"
        )
    print()
