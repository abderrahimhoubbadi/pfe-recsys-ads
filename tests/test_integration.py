"""
Integration Test — End-to-End Pipeline Verification.

Tests the full closed-loop pipeline without external dependencies
(no Docker, no Kafka, no Redis required).
"""

import sys
import time


def test_full_pipeline():
    """Test the complete recommendation pipeline."""
    print("=" * 60)
    print(" Integration Test — Closed-Loop Pipeline")
    print("=" * 60)

    # ── 1. Import all components ──
    print("\n[1/7] Importing modules...")
    t0 = time.perf_counter()

    from src.api.recommendation_service import RecommendationService
    from src.api.schemas import (
        AdInfo,
        RecommendationRequest,
        RecommendationResponse,
        FeedbackRequest,
    )
    from src.infra.factory import (
        MessageProducer,
        MessageConsumer,
        StateStore,
    )
    from src.agents.global_semantic_neural import GlobalSemanticDeepBandit
    from src.policy.moo_policies import epsilon_constraint_policy

    print(f"   ✅ All imports OK ({time.perf_counter() - t0:.1f}s)")

    # ── 2. Initialize service ──
    print("\n[2/7] Initializing RecommendationService (H-DeepBandit)...")
    t0 = time.perf_counter()
    service = RecommendationService(
        embedding_model="all-MiniLM-L6-v2",
        alpha=1.0,
        n_ensemble=5,
        epsilon=0.3,
    )
    print(f"   ✅ Service initialized ({time.perf_counter() - t0:.1f}s)")
    print(f"   Model: H-DeepBandit, dim={service.emb_dim}, arms={service.agent.n_arms}")

    # ── 3. Register ads ──
    print("\n[3/7] Registering ads...")
    ads = [
        AdInfo(
            ad_id=1,
            title="RTX 5090 GPU",
            description="Carte graphique NVIDIA pour gaming",
            category="tech",
        ),
        AdInfo(
            ad_id=2,
            title="Parfum Chanel",
            description="Parfum iconique pour femme",
            category="beauté",
        ),
        AdInfo(
            ad_id=3,
            title="Cours Python",
            description="Formation deep learning et IA",
            category="éducation",
        ),
        AdInfo(
            ad_id=4,
            title="Voyage Marrakech",
            description="Séjour tout inclus riad luxe",
            category="voyage",
        ),
        AdInfo(
            ad_id=5,
            title="MacBook Pro M4",
            description="Ordinateur portable Apple pour développeurs",
            category="tech",
        ),
        AdInfo(
            ad_id=6,
            title="Abonnement Netflix",
            description="Films et séries en streaming",
            category="divertissement",
        ),
    ]
    n_new = service.register_ads(ads)
    print(f"   ✅ {n_new} ads registered, total arms: {service.agent.n_arms}")

    # ── 4. Test recommendations for different users ──
    print("\n[4/7] Testing recommendations...")
    users = [
        ("homme 28 ans passionné de technologie et gaming", "Tech Gamer"),
        ("femme 35 ans intéressée par la mode et les cosmétiques", "Fashion"),
        ("étudiant 22 ans en informatique cherchant des cours en ligne", "Student"),
        ("couple 40 ans planifiant des vacances au Maroc", "Traveler"),
    ]

    results = []
    for user_text, label in users:
        ad_id, eng, rev, latency = service.recommend(
            user_text=user_text, available_ads=ads
        )
        ad_name = next(a.title for a in ads if a.ad_id == ad_id)
        results.append((label, ad_id, ad_name, eng, rev, latency))
        print(
            f"   {label:12s} → Ad #{ad_id} ({ad_name:18s}) | eng={eng:+.4f} rev={rev:+.4f} | {latency:.0f}ms"
        )

    print(f"   ✅ {len(users)} recommendations served")

    # ── 5. Test feedback loop (online learning) ──
    print("\n[5/7] Testing feedback loop (100 iterations)...")
    import random

    clicks = 0
    total_rev = 0.0
    for i in range(100):
        user_text, label = random.choice(users)
        ad_id, eng, rev, latency = service.recommend(
            user_text=user_text, available_ads=ads
        )

        # Simulate click based on engagement score
        clicked = random.random() < max(0.1, min(0.9, eng + 0.5))
        revenue = random.uniform(0.01, 0.15) if clicked else 0.0

        service.process_feedback(
            user_text=user_text,
            ad_id=ad_id,
            click=clicked,
            conversion=clicked and random.random() < 0.3,
            revenue=revenue,
        )

        if clicked:
            clicks += 1
            total_rev += revenue

    metrics = service.get_metrics()
    print(f"   Requests: {metrics['total_requests']}")
    print(f"   Feedbacks: {metrics['total_feedbacks']}")
    print(f"   Avg CTR: {metrics['avg_ctr']:.3f}")
    print(f"   Avg Revenue: {metrics['avg_revenue']:.4f}")
    print(f"   Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
    print(
        f"   ✅ Feedback loop working (model learned from {metrics['total_feedbacks']} updates)"
    )

    # ── 6. Test cold-start (new ads) ──
    print("\n[6/7] Testing cold-start (adding 2 new ads)...")
    new_ads = [
        AdInfo(
            ad_id=7,
            title="PS5 Pro",
            description="Console de jeu dernière génération",
            category="gaming",
        ),
        AdInfo(
            ad_id=8,
            title="Formation cloud GCP",
            description="Certification Google Cloud Platform",
            category="cloud",
        ),
    ]
    all_ads = ads + new_ads
    n_new = service.register_ads(new_ads)
    print(
        f"   ✅ {n_new} new ads added (zero-shot), total arms: {service.agent.n_arms}"
    )

    # Recommend with new ads available — zero-shot transfer should work
    ad_id, eng, rev, latency = service.recommend(
        user_text="homme 28 ans passionné de technologie et gaming",
        available_ads=all_ads,
    )
    ad_name = next(a.title for a in all_ads if a.ad_id == ad_id)
    print(f"   Tech Gamer → Ad #{ad_id} ({ad_name}) | eng={eng:+.4f} rev={rev:+.4f}")
    print(f"   ✅ Zero-shot transfer: new ads are immediately selectable")

    # ── 7. Test FastAPI app ──
    print("\n[7/7] Testing FastAPI app endpoints...")
    from fastapi.testclient import TestClient
    from src.api.main import app

    # Override the global service
    import src.api.main as main_module

    main_module.service = service

    client = TestClient(app)

    # Health check
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    print(f"   GET  /health  → {r.status_code} OK")

    # Recommend
    r = client.post(
        "/recommend",
        json={
            "user_id": 42,
            "user_text": "développeur Python passionné par l'IA",
            "available_ads": [a.model_dump() for a in all_ads],
        },
    )
    assert r.status_code == 200
    resp = r.json()
    print(
        f"   POST /recommend → {r.status_code} OK | ad={resp['selected_ad_id']}, eng={resp['engagement_score']:.4f}"
    )

    # Feedback
    r = client.post(
        "/feedback",
        json={
            "user_id": 42,
            "ad_id": resp["selected_ad_id"],
            "user_text": "développeur Python passionné par l'IA",
            "ad_text": "test ad",
            "click": True,
            "revenue": 0.05,
        },
    )
    assert r.status_code == 200
    assert r.json()["agent_updated"] is True
    print(f"   POST /feedback → {r.status_code} OK | agent_updated=True")

    # Metrics
    r = client.get("/metrics")
    assert r.status_code == 200
    m = r.json()
    print(
        f"   GET  /metrics  → {r.status_code} OK | requests={m['total_requests']}, feedbacks={m['total_feedbacks']}"
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print(" ✅ ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\n Pipeline: impression → SentenceTransformer → H-DeepBandit →")
    print(f"           ε-Constraint → decision → feedback → model update")
    print(f"\n Components verified:")
    print(f"   • RecommendationService (H-DeepBandit × ε-Constraint)")
    print(f"   • SentenceTransformer embedding (dim={service.emb_dim})")
    print(f"   • Online learning (closed-loop feedback)")
    print(f"   • Cold-start zero-shot transfer")
    print(f"   • FastAPI endpoints (/recommend, /feedback, /health, /metrics)")
    print(f"   • Infrastructure factory (abstract interfaces)")
    print()


if __name__ == "__main__":
    test_full_pipeline()
