# Expériences et Simulations (`experiments`)

Ce dossier contient les scripts exécutables pour lancer les différentes phases du projet.

## Scripts Disponibles

### 1. Phase Offline
- **`offline_simulation.py`**
    - **But :** Vérifier que l'algo LinUCB apprend (Regret décroissant).
    - **Mode :** Tout en RAM, pas de base de données.
    - **Commande :** `python3 experiments/offline_simulation.py`

### 2. Phase Temps Réel
- **`realtime_simulation.py`**
    - **But :** Valider la boucle complète Kafka + Redis.
    - **Pré-requis :** Docker lancé (`docker compose up -d`).
    - **Fonctionnement :** Lance un thread Générateur (Producer) et un thread Agent (Consumer).

### 3. Phase Off-Policy (IPS)
- **`run_ips_benchmark.py`**
    - **But :** Comparer LinUCB vs Random vs Ancien Système sur des logs.
    - **Pré-requis :** Avoir généré des logs (`src/data_gen/log_synthetic.py`).
- **`plot_ips_convergence.py`**
    - **But :** Générer le graphique de convergence (`metrics/ips_convergence.png`).
