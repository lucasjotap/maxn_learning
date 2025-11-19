"""Collaborative Filtering Recommendation Model using Matrix Factorization."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix
try:
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False


class CollaborativeFilteringRecommender:
    """Matrix factorization-based collaborative filtering."""
    
    def __init__(
        self,
        factors: int = 50,
        regularization: float = 0.01,
        iterations: int = 15,
        algorithm: str = 'als'  # 'als' or 'bpr'
    ):
        """Initialize collaborative filtering model.
        
        Args:
            factors: Number of latent factors
            regularization: Regularization parameter
            iterations: Number of training iterations
            algorithm: 'als' (Alternating Least Squares) or 'bpr' (Bayesian Personalized Ranking)
        """
        if not IMPLICIT_AVAILABLE:
            raise ImportError("implicit library required for collaborative filtering")
        
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.algorithm = algorithm
        
        if algorithm == 'als':
            self.model = AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                random_state=42
            )
        elif algorithm == 'bpr':
            self.model = BayesianPersonalizedRanking(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.user_ids: Optional[List[str]] = None
        self.item_ids: Optional[List[str]] = None
        self.user_id_map: Dict[str, int] = {}
        self.item_id_map: Dict[str, int] = {}
        self.reverse_user_map: Dict[int, str] = {}
        self.reverse_item_map: Dict[int, str] = {}
        self.is_fitted = False
    
    def _build_id_maps(self, user_ids: List[str], item_ids: List[str]):
        """Build mapping between string IDs and integer indices."""
        self.user_ids = sorted(set(user_ids))
        self.item_ids = sorted(set(item_ids))
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
    
    def _build_sparse_matrix(
        self,
        user_ids: List[str],
        item_ids: List[str],
        scores: List[float]
    ) -> csr_matrix:
        """Build sparse user-item matrix."""
        rows = [self.user_id_map[uid] for uid in user_ids]
        cols = [self.item_id_map[iid] for iid in item_ids]
        values = scores
        
        matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        )
        return matrix
    
    def fit(
        self,
        user_ids: List[str],
        item_ids: List[str],
        scores: List[float]
    ):
        """Fit the collaborative filtering model.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            scores: Interaction scores/ratings
        """
        if len(user_ids) != len(item_ids) or len(item_ids) != len(scores):
            raise ValueError("user_ids, item_ids, and scores must have same length")
        
        self._build_id_maps(user_ids, item_ids)
        matrix = self._build_sparse_matrix(user_ids, item_ids, scores)
        
        # Train model (implicit expects item-user matrix for ALS)
        self.model.fit(matrix.T)
        self.is_fitted = True
    
    def recommend(
        self,
        user_id: str,
        exclude_items: Optional[List[str]] = None,
        top_k: int = 10,
        filter_already_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """Get recommendations for a user.
        
        Args:
            user_id: User ID
            exclude_items: Item IDs to exclude
            top_k: Number of recommendations
            filter_already_interacted: Whether to filter items user already interacted with
        
        Returns:
            List of recommendations with item_id and score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommending")
        
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        
        # Get user recommendations
        item_indices, scores = self.model.recommend(
            user_idx,
            self.model.item_factors,
            N=top_k * 2,  # Get more than needed for filtering
            filter_already_interacted_items=filter_already_interacted
        )
        
        exclude_set = set(exclude_items or [])
        recommendations = []
        
        for idx, score in zip(item_indices, scores):
            item_id = self.reverse_item_map[idx]
            
            if item_id not in exclude_set:
                recommendations.append({
                    'item_id': item_id,
                    'score': float(score)
                })
            
            if len(recommendations) >= top_k:
                break
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def similar_items(
        self,
        item_id: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find items similar to given item.
        
        Args:
            item_id: Item ID
            top_k: Number of similar items
        
        Returns:
            List of similar items with item_id and score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        if item_id not in self.item_id_map:
            return []
        
        item_idx = self.item_id_map[item_id]
        similar_indices, scores = self.model.similar_items(item_idx, N=top_k + 1)
        
        recommendations = []
        for idx, score in zip(similar_indices, scores):
            similar_item_id = self.reverse_item_map[idx]
            if similar_item_id != item_id:  # Exclude self
                recommendations.append({
                    'item_id': similar_item_id,
                    'score': float(score)
                })
        
        return recommendations
    
    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'user_ids': self.user_ids,
                'item_ids': self.item_ids,
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'reverse_user_map': self.reverse_user_map,
                'reverse_item_map': self.reverse_item_map,
                'factors': self.factors,
                'regularization': self.regularization,
                'iterations': self.iterations,
                'algorithm': self.algorithm,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'CollaborativeFilteringRecommender':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        recommender = cls(
            factors=data['factors'],
            regularization=data['regularization'],
            iterations=data['iterations'],
            algorithm=data['algorithm']
        )
        recommender.model = data['model']
        recommender.user_ids = data['user_ids']
        recommender.item_ids = data['item_ids']
        recommender.user_id_map = data['user_id_map']
        recommender.item_id_map = data['item_id_map']
        recommender.reverse_user_map = data['reverse_user_map']
        recommender.reverse_item_map = data['reverse_item_map']
        recommender.is_fitted = data['is_fitted']
        
        return recommender

