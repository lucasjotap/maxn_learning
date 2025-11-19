"""Unified Recommendation Engine combining multiple models."""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml

from data_warehouse import DataWarehouse
from models.knn_recommender import KNNRecommender
from models.collaborative_filtering import CollaborativeFilteringRecommender
from models.embedding_recommender import EmbeddingRecommender


class RecommendationEngine:
    """Unified recommendation engine combining KNN, collaborative filtering, and embeddings."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize recommendation engine."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.warehouse = DataWarehouse(self.config['data_warehouse']['path'])
        self.models: Dict[str, Dict[str, Any]] = {}
        self.models_dir = Path(self.config['models']['base_path'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load domain configurations
        self.domains = self.config.get('domains', {})
    
    def _get_model_path(self, domain: str, model_type: str) -> str:
        """Get path for saved model."""
        return str(self.models_dir / f"{domain}_{model_type}.pkl")
    
    def train_knn(
        self,
        domain: str,
        user_item_matrix: np.ndarray,
        item_ids: List[str]
    ):
        """Train KNN model for domain."""
        if domain not in self.domains:
            raise ValueError(f"Domain {domain} not configured")
        
        config = self.domains[domain]
        model = KNNRecommender(n_neighbors=config.get('knn_neighbors', 20))
        model.fit(user_item_matrix, item_ids)
        
        path = self._get_model_path(domain, 'knn')
        model.save(path)
        
        if domain not in self.models:
            self.models[domain] = {}
        self.models[domain]['knn'] = model
        
        # Log training
        self.warehouse.log_training(
            domain=domain,
            model_type='knn',
            metrics={'n_items': len(item_ids), 'n_neighbors': config.get('knn_neighbors', 20)},
            model_path=path,
            training_samples=len(item_ids)
        )
    
    def train_collaborative_filtering(
        self,
        domain: str,
        user_ids: List[str],
        item_ids: List[str],
        scores: List[float]
    ):
        """Train collaborative filtering model for domain."""
        if domain not in self.domains:
            raise ValueError(f"Domain {domain} not configured")
        
        model = CollaborativeFilteringRecommender(
            factors=50,
            regularization=0.01,
            iterations=15
        )
        model.fit(user_ids, item_ids, scores)
        
        path = self._get_model_path(domain, 'collab')
        model.save(path)
        
        if domain not in self.models:
            self.models[domain] = {}
        self.models[domain]['collab'] = model
        
        # Log training
        self.warehouse.log_training(
            domain=domain,
            model_type='collab',
            metrics={'n_users': len(set(user_ids)), 'n_items': len(set(item_ids))},
            model_path=path,
            training_samples=len(scores)
        )
    
    def train_embedding(
        self,
        domain: str,
        item_ids: List[str],
        item_texts: List[str],
        embeddings: Optional[np.ndarray] = None
    ):
        """Train embedding model for domain."""
        if domain not in self.domains:
            raise ValueError(f"Domain {domain} not configured")
        
        config = self.domains[domain]
        model_name = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        model = EmbeddingRecommender(model_name=model_name)
        model.fit(item_ids, item_texts, embeddings)
        
        path = self._get_model_path(domain, 'embedding')
        model.save(path)
        
        if domain not in self.models:
            self.models[domain] = {}
        self.models[domain]['embedding'] = model
        
        # Log training
        self.warehouse.log_training(
            domain=domain,
            model_type='embedding',
            metrics={'n_items': len(item_ids), 'model': model_name},
            model_path=path,
            training_samples=len(item_ids)
        )
    
    def load_models(self, domain: str):
        """Load trained models for domain."""
        if domain not in self.domains:
            raise ValueError(f"Domain {domain} not configured")
        
        config = self.domains[domain]
        
        # Load KNN
        knn_path = self._get_model_path(domain, 'knn')
        if Path(knn_path).exists():
            self.models.setdefault(domain, {})['knn'] = KNNRecommender.load(knn_path)
        
        # Load Collaborative Filtering
        collab_path = self._get_model_path(domain, 'collab')
        if Path(collab_path).exists():
            self.models.setdefault(domain, {})['collab'] = CollaborativeFilteringRecommender.load(collab_path)
        
        # Load Embedding
        embedding_path = self._get_model_path(domain, 'embedding')
        if Path(embedding_path).exists():
            model_name = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            self.models.setdefault(domain, {})['embedding'] = EmbeddingRecommender.load(embedding_path, device=None)
    
    def recommend(
        self,
        user_id: str,
        domain: str,
        model_types: Optional[List[str]] = None,
        top_k: int = 10,
        exclude_interacted: bool = True,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recommendations using ensemble of models.
        
        Args:
            user_id: User ID
            domain: Domain (movies, music, tasks, notes)
            model_types: Which models to use ['knn', 'collab', 'embedding'], None = all
            top_k: Number of recommendations
            exclude_interacted: Whether to exclude already interacted items
            query_text: Optional text query for embedding model
        
        Returns:
            List of recommendations with item_id, score, and model_type
        """
        if domain not in self.models:
            self.load_models(domain)
        
        if domain not in self.models or not self.models[domain]:
            return []
        
        if model_types is None:
            model_types = list(self.models[domain].keys())
        
        # Get user's interacted items
        exclude_items = []
        if exclude_interacted:
            interactions = self.warehouse.get_user_interactions(user_id=user_id, domain=domain)
            exclude_items = interactions['item_id'].tolist() if not interactions.empty else []
        
        all_recommendations = []
        
        # Get recommendations from each model
        for model_type in model_types:
            if model_type not in self.models[domain]:
                continue
            
            model = self.models[domain][model_type]
            
            try:
                if model_type == 'collab':
                    recs = model.recommend(
                        user_id=user_id,
                        exclude_items=exclude_items,
                        top_k=top_k * 2  # Get more for ensemble
                    )
                elif model_type == 'embedding':
                    if query_text:
                        recs = model.recommend(
                            query_text=query_text,
                            exclude_items=exclude_items,
                            top_k=top_k * 2
                        )
                    else:
                        # Use user's recent interactions to build query
                        interactions = self.warehouse.get_user_interactions(
                            user_id=user_id,
                            domain=domain,
                            limit=10
                        )
                        if not interactions.empty:
                            # Get text descriptions of liked items
                            items_df = self.warehouse.get_items(domain)
                            liked_items = interactions[interactions['rating'] >= 4.0] if 'rating' in interactions.columns else interactions
                            
                            if not liked_items.empty:
                                texts = []
                                for _, row in liked_items.iterrows():
                                    item_df = items_df[items_df['item_id'] == row['item_id']]
                                    if not item_df.empty:
                                        desc = f"{item_df.iloc[0].get('title', '')} {item_df.iloc[0].get('description', '')}"
                                        texts.append(desc)
                                
                                if texts:
                                    query_text = " ".join(texts[:3])  # Use top 3 liked items
                        
                        if query_text:
                            recs = model.recommend(
                                query_text=query_text,
                                exclude_items=exclude_items,
                                top_k=top_k * 2
                            )
                        else:
                            continue
                elif model_type == 'knn':
                    # For KNN, use user's most recent interaction
                    interactions = self.warehouse.get_user_interactions(
                        user_id=user_id,
                        domain=domain,
                        limit=1
                    )
                    if not interactions.empty:
                        recent_item = interactions.iloc[0]['item_id']
                        recs = model.recommend(
                            item_id=recent_item,
                            exclude_items=exclude_items,
                            top_k=top_k * 2
                        )
                    else:
                        continue
                else:
                    continue
                
                # Tag with model type
                for rec in recs:
                    rec['model_type'] = model_type
                
                all_recommendations.extend(recs)
            
            except Exception as e:
                print(f"Error getting recommendations from {model_type}: {e}")
                continue
        
        # Ensemble: aggregate scores from different models
        item_scores: Dict[str, Dict[str, float]] = {}
        for rec in all_recommendations:
            item_id = rec['item_id']
            if item_id not in item_scores:
                item_scores[item_id] = {'scores': [], 'models': []}
            item_scores[item_id]['scores'].append(rec['score'])
            item_scores[item_id]['models'].append(rec['model_type'])
        
        # Combine scores (weighted average)
        ensemble_recs = []
        for item_id, data in item_scores.items():
            if item_id in exclude_items:
                continue
            
            # Weighted average (can be customized)
            avg_score = np.mean(data['scores'])
            ensemble_recs.append({
                'item_id': item_id,
                'score': float(avg_score),
                'model_types': list(set(data['models'])),
                'n_models': len(set(data['models']))
            })
        
        # Sort by score
        ensemble_recs.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top-k
        recommendations = ensemble_recs[:top_k]
        
        # Save to warehouse
        if recommendations:
            self.warehouse.save_recommendations(
                user_id=user_id,
                domain=domain,
                recommendations=recommendations,
                model_type='ensemble'
            )
        
        return recommendations
    
    def train_all_models(self, domain: str, min_samples: int = 10):
        """Train all models for a domain from warehouse data."""
        if domain not in self.domains:
            raise ValueError(f"Domain {domain} not configured")
        
        # Get data from warehouse
        interactions_df = self.warehouse.get_user_interactions(domain=domain)
        items_df = self.warehouse.get_items(domain)
        
        if interactions_df.empty or items_df.empty:
            print(f"Insufficient data for domain {domain}")
            return
        
        if len(interactions_df) < min_samples:
            print(f"Not enough samples for {domain}: {len(interactions_df)} < {min_samples}")
            return
        
        # Train Collaborative Filtering
        try:
            user_item_df = self.warehouse.get_user_item_matrix(domain)
            if not user_item_df.empty and len(user_item_df) >= min_samples:
                self.train_collaborative_filtering(
                    domain=domain,
                    user_ids=user_item_df['user_id'].tolist(),
                    item_ids=user_item_df['item_id'].tolist(),
                    scores=user_item_df['score'].tolist()
                )
        except Exception as e:
            print(f"Error training collaborative filtering for {domain}: {e}")
        
        # Train KNN
        try:
            if not user_item_df.empty:
                # Build user-item matrix
                user_ids = sorted(user_item_df['user_id'].unique())
                item_ids = sorted(user_item_df['item_id'].unique())
                
                matrix = np.zeros((len(item_ids), len(user_ids)))
                for _, row in user_item_df.iterrows():
                    u_idx = user_ids.index(row['user_id'])
                    i_idx = item_ids.index(row['item_id'])
                    matrix[i_idx, u_idx] = row['score']
                
                self.train_knn(domain=domain, user_item_matrix=matrix, item_ids=item_ids)
        except Exception as e:
            print(f"Error training KNN for {domain}: {e}")
        
        # Train Embedding
        try:
            if not items_df.empty:
                item_ids = items_df['item_id'].tolist()
                item_texts = []
                
                for _, row in items_df.iterrows():
                    title = row.get('title', '') or ''
                    desc = row.get('description', '') or ''
                    text = f"{title} {desc}".strip()
                    item_texts.append(text if text else str(row['item_id']))
                
                self.train_embedding(domain=domain, item_ids=item_ids, item_texts=item_texts)
        except Exception as e:
            print(f"Error training embedding for {domain}: {e}")
    
    def auto_retrain(self, domain: Optional[str] = None):
        """Auto-retrain models based on new data."""
        domains = [domain] if domain else list(self.domains.keys())
        min_samples = self.config['models'].get('min_samples_for_training', 10)
        
        for dom in domains:
            if not self.domains[dom].get('enabled', True):
                continue
            
            print(f"Auto-retraining models for {dom}...")
            self.train_all_models(dom, min_samples=min_samples)
            print(f"Completed auto-retraining for {dom}")

