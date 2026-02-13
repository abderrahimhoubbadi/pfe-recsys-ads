import numpy as np
import logging

class IPSEvaluator:
    """
    Implémentation de l'évaluation Off-Policy via Inverse Propensity Scoring (IPS).
    Utilise la méthode du "Replay" (Match/Reject).
    """
    def __init__(self, agent, cap_M=10.0):
        """
        Args:
            agent: L'agent à évaluer (doit avoir select_arm et update).
            cap_M (float): Valeur de plafonnement pour le Clipped IPS.
        """
        self.agent = agent
        self.cap_M = cap_M
        self.logger = logging.getLogger(__name__)

    def evaluate(self, dataset):
        """
        Exécute le replay sur le dataset fourni.
        
        Args:
            dataset: Liste d'événements (dicts ou objets) contenant:
                     - context (list/array)
                     - action (int)
                     - reward (float)
                     - propensity (float)
                     
        Returns:
            float: Le score IPS estimé (Performance).
            dict: Métriques détaillées (count, matches, etc.)
        """
        score_cumulative = 0.0
        count = 0
        matches = 0
        history = []
        
        self.logger.info(f"Starting IPS Evaluation on {len(dataset)} events with M={self.cap_M}")
        
        for i, event in enumerate(dataset):
            context = np.array(event['context'])
            log_action = event['action']
            reward = event['reward']
            propensity = event['propensity']
            
            # 1. Prediction (Counterfactual)
            # L'agent choisit une action pour ce contexte
            predicted_action = self.agent.select_arm(context)
            
            # 2. Matching
            if predicted_action == log_action:
                matches += 1
                
                # 3. Calcul du Poids (Clipped IPS)
                # w_t = min(1/p_t, M)
                weight = min(1.0 / propensity, self.cap_M)
                
                # Weighted Reward
                weighted_reward = weight * reward
                
                score_cumulative += weighted_reward
                count += 1
                
                # 4. Learning (Online Update)
                # On ne met à jour l'agent QUE si on a un match (car on a le feedback réel)
                self.agent.update(context, log_action, reward)
                
                if count > 0:
                    current_avg = score_cumulative / count
                    history.append(current_avg)
                
            else:
                # Rejet
                if count > 0:
                    # On répète la dernière valeur connue pour avoir une courbe continue
                    # Ou on ne fait rien (courbe par "step valide").
                    # Option 2 : On append la dernière valeur pour que la courbe ait la même taille que 'matches'
                    history.append(score_cumulative / count)
                else:
                    history.append(0.0)
                
        if count == 0:
            return 0.0, {"matches": 0, "total": len(dataset), "history": []}
            
        final_score = score_cumulative / count
        
        metrics = {
            "ips_score": final_score,
            "total_events": len(dataset),
            "matched_events": matches,
            "match_rate": matches / len(dataset),
            "history": history
        }
        
        return final_score, metrics
