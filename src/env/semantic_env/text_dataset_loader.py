"""
Text Dataset Loader — Generates a realistic catalog of ads with textual descriptions.

Instead of anonymous numerical IDs, each ad/arm has a real category, title,
and description that can be meaningfully embedded by a language model.
This is designed for validating the Cold-Start advantage of semantic embeddings.
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================
# A curated catalog of 60 realistic online ads across 6 categories.
# The key insight: ads WITHIN the same category should have similar
# embeddings, enabling "Zero-Shot Transfer" when new ads appear.
# ============================================================

AD_CATALOG = [
    # ── Category 1: Smartphones & Tech (IDs 0-9) ──
    {
        "id": 0,
        "category": "tech",
        "title": "iPhone 16 Pro Max",
        "desc": "Latest Apple smartphone with A18 chip, 48MP camera, and titanium design.",
    },
    {
        "id": 1,
        "category": "tech",
        "title": "Samsung Galaxy S25 Ultra",
        "desc": "Samsung flagship with Snapdragon processor, S-Pen, and 200MP camera.",
    },
    {
        "id": 2,
        "category": "tech",
        "title": "Google Pixel 9 Pro",
        "desc": "Google phone with Tensor G4 chip, best-in-class AI photo features.",
    },
    {
        "id": 3,
        "category": "tech",
        "title": "OnePlus 13",
        "desc": "Premium Android phone with Hasselblad camera and 100W fast charging.",
    },
    {
        "id": 4,
        "category": "tech",
        "title": "Sony Xperia 1 VI",
        "desc": "Professional-grade smartphone with 4K OLED display and Alpha camera tech.",
    },
    {
        "id": 5,
        "category": "tech",
        "title": "Xiaomi 15 Ultra",
        "desc": "Chinese flagship with Leica optics and 6000mAh silicon-carbon battery.",
    },
    {
        "id": 6,
        "category": "tech",
        "title": "MacBook Pro M4",
        "desc": "Apple laptop with M4 Pro chip, 18-hour battery, and Liquid Retina XDR display.",
    },
    {
        "id": 7,
        "category": "tech",
        "title": "iPad Air M3",
        "desc": "Thin and powerful Apple tablet with M3 chip for creative professionals.",
    },
    {
        "id": 8,
        "category": "tech",
        "title": "Samsung Galaxy Tab S10",
        "desc": "Android tablet with AMOLED display, DeX mode, and S-Pen productivity.",
    },
    {
        "id": 9,
        "category": "tech",
        "title": "AirPods Pro 3",
        "desc": "Apple wireless earbuds with adaptive audio and hearing health features.",
    },
    # ── Category 2: Fashion & Clothing (IDs 10-19) ──
    {
        "id": 10,
        "category": "fashion",
        "title": "Nike Air Max 2025",
        "desc": "Iconic running sneakers with visible Air cushioning and recycled mesh upper.",
    },
    {
        "id": 11,
        "category": "fashion",
        "title": "Adidas Ultraboost Light",
        "desc": "Lightweight performance running shoes with responsive Boost midsole.",
    },
    {
        "id": 12,
        "category": "fashion",
        "title": "Zara Summer Collection",
        "desc": "Trendy summer dresses and linen shirts for warm weather casual style.",
    },
    {
        "id": 13,
        "category": "fashion",
        "title": "Levi's 501 Original Jeans",
        "desc": "Classic straight-fit denim jeans, an American fashion staple since 1873.",
    },
    {
        "id": 14,
        "category": "fashion",
        "title": "H&M Conscious Line",
        "desc": "Sustainable fashion collection made from organic cotton and recycled polyester.",
    },
    {
        "id": 15,
        "category": "fashion",
        "title": "Gucci Horsebit Loafers",
        "desc": "Luxury Italian leather loafers with signature Gucci horsebit hardware.",
    },
    {
        "id": 16,
        "category": "fashion",
        "title": "Ray-Ban Wayfarer",
        "desc": "Timeless sunglasses with polarized lenses and classic acetate frame.",
    },
    {
        "id": 17,
        "category": "fashion",
        "title": "Uniqlo Heattech Thermal",
        "desc": "Japanese thermal underwear with heat-retaining technology for cold winters.",
    },
    {
        "id": 18,
        "category": "fashion",
        "title": "Patagonia Down Jacket",
        "desc": "Eco-friendly insulated jacket made with fair-trade recycled down feathers.",
    },
    {
        "id": 19,
        "category": "fashion",
        "title": "New Balance 990v6",
        "desc": "Made-in-USA premium lifestyle sneakers with superior arch support.",
    },
    # ── Category 3: Food & Beverages (IDs 20-29) ──
    {
        "id": 20,
        "category": "food",
        "title": "Uber Eats Promo -30%",
        "desc": "Order food delivery from local restaurants with 30% discount on first order.",
    },
    {
        "id": 21,
        "category": "food",
        "title": "HelloFresh Meal Kit",
        "desc": "Weekly meal kit subscription with pre-portioned ingredients and easy recipes.",
    },
    {
        "id": 22,
        "category": "food",
        "title": "Starbucks Iced Latte",
        "desc": "Refreshing espresso-based iced coffee with your choice of milk and syrup.",
    },
    {
        "id": 23,
        "category": "food",
        "title": "Coca-Cola Zero Sugar",
        "desc": "Zero-calorie carbonated soft drink with original Coca-Cola taste.",
    },
    {
        "id": 24,
        "category": "food",
        "title": "McDonald's Big Mac Meal",
        "desc": "Classic double-patty burger with special sauce, fries, and a drink.",
    },
    {
        "id": 25,
        "category": "food",
        "title": "Domino's Pizza Deal",
        "desc": "Large pepperoni pizza with stuffed crust, two-for-one Tuesday special.",
    },
    {
        "id": 26,
        "category": "food",
        "title": "Nespresso Vertuo Pods",
        "desc": "Premium coffee capsules for Nespresso machines, various blend intensities.",
    },
    {
        "id": 27,
        "category": "food",
        "title": "Whole Foods Organic Box",
        "desc": "Weekly organic fruit and vegetable delivery box from local farms.",
    },
    {
        "id": 28,
        "category": "food",
        "title": "Red Bull Energy Drink",
        "desc": "Caffeine and taurine energy drink for mental and physical performance.",
    },
    {
        "id": 29,
        "category": "food",
        "title": "Lindt Swiss Chocolate",
        "desc": "Premium Swiss milk chocolate truffles, luxury gift box assortment.",
    },
    # ── Category 4: Travel & Transport (IDs 30-39) ──
    {
        "id": 30,
        "category": "travel",
        "title": "Booking.com Flash Sale",
        "desc": "Hotel deals worldwide with up to 50% off, free cancellation available.",
    },
    {
        "id": 31,
        "category": "travel",
        "title": "Ryanair Weekend Flights",
        "desc": "Budget airlines flights across Europe starting from 15 euros one way.",
    },
    {
        "id": 32,
        "category": "travel",
        "title": "Airbnb Unique Stays",
        "desc": "Book unique vacation homes, treehouses, and castles worldwide.",
    },
    {
        "id": 33,
        "category": "travel",
        "title": "Hertz Car Rental",
        "desc": "Rent a car for your road trip with GPS and insurance included.",
    },
    {
        "id": 34,
        "category": "travel",
        "title": "Emirates Business Class",
        "desc": "Luxury air travel with lie-flat seats, gourmet meals, and lounge access.",
    },
    {
        "id": 35,
        "category": "travel",
        "title": "InterRail Europe Pass",
        "desc": "Unlimited train travel across 33 European countries for youth travelers.",
    },
    {
        "id": 36,
        "category": "travel",
        "title": "TUI All-Inclusive Resort",
        "desc": "Beach resort packages in Mediterranean with flights, meals, and activities.",
    },
    {
        "id": 37,
        "category": "travel",
        "title": "Lime Electric Scooter",
        "desc": "Rent an e-scooter in your city for short commutes and urban exploration.",
    },
    {
        "id": 38,
        "category": "travel",
        "title": "Lonely Planet Travel Guide",
        "desc": "Best-selling travel guidebook with insider tips and cultural insights.",
    },
    {
        "id": 39,
        "category": "travel",
        "title": "Samsonite Carry-On",
        "desc": "Lightweight hardshell cabin luggage with TSA lock and spinner wheels.",
    },
    # ── Category 5: Health & Fitness (IDs 40-49) ──
    {
        "id": 40,
        "category": "health",
        "title": "Apple Watch Ultra 3",
        "desc": "Rugged smartwatch with advanced health sensors, GPS, and dive computer.",
    },
    {
        "id": 41,
        "category": "health",
        "title": "Peloton Bike+",
        "desc": "Indoor cycling bike with live and on-demand fitness classes streaming.",
    },
    {
        "id": 42,
        "category": "health",
        "title": "Whoop 5.0 Fitness Band",
        "desc": "Wearable fitness tracker monitoring sleep, strain, and recovery metrics.",
    },
    {
        "id": 43,
        "category": "health",
        "title": "MyProtein Whey Isolate",
        "desc": "High-protein powder supplement for muscle recovery after workouts.",
    },
    {
        "id": 44,
        "category": "health",
        "title": "Headspace Meditation App",
        "desc": "Guided meditation and mindfulness app for stress relief and better sleep.",
    },
    {
        "id": 45,
        "category": "health",
        "title": "Theragun Pro Massager",
        "desc": "Percussive therapy device for deep muscle treatment and pain relief.",
    },
    {
        "id": 46,
        "category": "health",
        "title": "Garmin Forerunner 965",
        "desc": "Advanced GPS running watch with training load analytics and race predictor.",
    },
    {
        "id": 47,
        "category": "health",
        "title": "Vitamix Blender A3500",
        "desc": "Professional-grade blender for smoothies, soups, and healthy meal prep.",
    },
    {
        "id": 48,
        "category": "health",
        "title": "Oura Ring Gen 4",
        "desc": "Smart ring tracking sleep stages, heart rate, and body temperature trends.",
    },
    {
        "id": 49,
        "category": "health",
        "title": "Nike Training Club Premium",
        "desc": "Workout app with personalized training plans and expert-led video sessions.",
    },
    # ── Category 6: Entertainment & Gaming (IDs 50-59) ──
    {
        "id": 50,
        "category": "gaming",
        "title": "PlayStation 6 Console",
        "desc": "Next-gen Sony gaming console with ray tracing, SSD, and exclusive titles.",
    },
    {
        "id": 51,
        "category": "gaming",
        "title": "Nintendo Switch 2",
        "desc": "Hybrid portable gaming console with 1080p OLED screen and Joy-Con 2.",
    },
    {
        "id": 52,
        "category": "gaming",
        "title": "Xbox Game Pass Ultimate",
        "desc": "Cloud gaming subscription with hundreds of games on console, PC, and mobile.",
    },
    {
        "id": 53,
        "category": "gaming",
        "title": "Steam Deck 2",
        "desc": "Handheld PC gaming device running SteamOS with custom AMD APU.",
    },
    {
        "id": 54,
        "category": "gaming",
        "title": "Netflix Premium Plan",
        "desc": "Ad-free streaming with 4K HDR video and spatial audio on 4 devices.",
    },
    {
        "id": 55,
        "category": "gaming",
        "title": "Spotify Premium Family",
        "desc": "Music streaming for up to 6 accounts with offline downloads and no ads.",
    },
    {
        "id": 56,
        "category": "gaming",
        "title": "Razer BlackWidow Keyboard",
        "desc": "Mechanical gaming keyboard with RGB lighting and tactile green switches.",
    },
    {
        "id": 57,
        "category": "gaming",
        "title": "Meta Quest 4 VR Headset",
        "desc": "Standalone VR headset with mixed reality, eye tracking, and wireless play.",
    },
    {
        "id": 58,
        "category": "gaming",
        "title": "Disney+ Annual Plan",
        "desc": "Streaming service with Marvel, Star Wars, Pixar, and National Geographic.",
    },
    {
        "id": 59,
        "category": "gaming",
        "title": "Logitech G Pro Mouse",
        "desc": "Lightweight wireless gaming mouse with HERO 25K sensor and LIGHTSPEED.",
    },
]

# ── User Profiles (Textual) ──
USER_PROFILES = [
    {
        "id": 0,
        "desc": "Young male tech enthusiast who loves gadgets, smartphones, and PC gaming.",
        "preferences": ["tech", "gaming"],
    },
    {
        "id": 1,
        "desc": "Fashion-conscious woman who follows luxury brands and sustainable fashion trends.",
        "preferences": ["fashion"],
    },
    {
        "id": 2,
        "desc": "Health-focused runner who tracks fitness metrics and follows a high-protein diet.",
        "preferences": ["health", "food"],
    },
    {
        "id": 3,
        "desc": "Frequent traveler who books budget flights and explores new cities every month.",
        "preferences": ["travel"],
    },
    {
        "id": 4,
        "desc": "Casual gamer and movie fan who subscribes to multiple streaming platforms.",
        "preferences": ["gaming"],
    },
    {
        "id": 5,
        "desc": "Busy office worker who orders food delivery daily and drinks lots of coffee.",
        "preferences": ["food"],
    },
    {
        "id": 6,
        "desc": "University student interested in affordable tech, fast food, and entertainment.",
        "preferences": ["tech", "food", "gaming"],
    },
    {
        "id": 7,
        "desc": "Fitness influencer and outdoor enthusiast who reviews sports gear and supplements.",
        "preferences": ["health", "fashion"],
    },
    {
        "id": 8,
        "desc": "Retired couple who enjoy luxury travel, fine dining, and cultural experiences.",
        "preferences": ["travel", "food"],
    },
    {
        "id": 9,
        "desc": "DIY maker and early adopter who buys every new Apple product on launch day.",
        "preferences": ["tech"],
    },
]


class TextDatasetLoader:
    """
    Provides a realistic text-based ad catalog and user profiles.

    Key feature: Supports a 'cold-start split' where a percentage of items
    are initially hidden and revealed later during the simulation.
    """

    def __init__(self, cold_start_ratio: float = 0.2, seed: int = 42):
        """
        Args:
            cold_start_ratio: Fraction of ads to withhold for the Cold-Start shock.
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.cold_start_ratio = cold_start_ratio
        self.all_ads = AD_CATALOG.copy()
        self.user_profiles = USER_PROFILES.copy()

        # Shuffle and split items for cold-start
        n_total = len(self.all_ads)
        n_hidden = int(n_total * cold_start_ratio)
        indices = self.rng.permutation(n_total)

        self._known_indices = sorted(indices[n_hidden:].tolist())
        self._hidden_indices = sorted(indices[:n_hidden].tolist())

        self.known_ads = [self.all_ads[i] for i in self._known_indices]
        self.hidden_ads = [self.all_ads[i] for i in self._hidden_indices]

        logger.info(
            f"TextDatasetLoader: {len(self.known_ads)} known ads, "
            f"{len(self.hidden_ads)} hidden for cold-start."
        )

    def get_known_ads(self) -> List[Dict]:
        """Return ads available from the start."""
        return self.known_ads

    def get_hidden_ads(self) -> List[Dict]:
        """Return ads hidden for cold-start injection."""
        return self.hidden_ads

    def get_all_ads(self) -> List[Dict]:
        """Return the full catalog (known + hidden)."""
        return self.known_ads + self.hidden_ads

    def get_ad_texts(self, ads: List[Dict]) -> List[str]:
        """Convert ads to their textual representation for embedding."""
        return [f"{ad['title']}. {ad['desc']}" for ad in ads]

    def get_random_user(self) -> Dict:
        """Sample a random user profile."""
        idx = self.rng.integers(0, len(self.user_profiles))
        return self.user_profiles[idx]

    def get_user_text(self, user: Dict) -> str:
        """Convert user profile to text for embedding."""
        return user["desc"]

    def get_n_known_arms(self) -> int:
        return len(self.known_ads)

    def get_n_total_arms(self) -> int:
        return len(self.all_ads)

    def get_categories(self) -> List[str]:
        return list(set(ad["category"] for ad in self.all_ads))
