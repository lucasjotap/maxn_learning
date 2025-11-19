"""KNN-based Recommendation Model."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path


class KNNRecommender:
    """KNN-based collaborative filtering recommender."""
    
    def __init__(self, n_neighbors: int = 20, metric: str = 'cosine'):
        """Initialize KNN recommender.
        
        Args:
            n_neighbors: Number of neighbors to consider
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model: Optional[NearestNeighbors] = None
        self.item_ids: Optional[np.ndarray] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, user_item_matrix: np.ndarray, item_ids: List[str]):
        """Fit the KNN model on user-item matrix.
        
        Args:
            user_item_matrix: 2D array of shape (n_items, n_users) or (n_items, n_features)
            item_ids: List of item IDs corresponding to rows
        """
        if user_item_matrix.shape[0] != len(item_ids):
            raise ValueError("Item IDs length must match matrix rows")
        
        self.item_ids = np.array(item_ids)
        
        # Normalize features if needed
        if user_item_matrix.shape[1] > 10:  # Treat as feature matrix
            user_item_matrix = self.scaler.fit_transform(user_item_matrix)
        
        self.model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(item_ids)),
            metric=self.metric,
            algorithm='auto'
        )
        self.model.fit(user_item_matrix)
        self.is_fitted = True
    
    def recommend(
        self,
        item_id: str,
        exclude_items: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on item similarity.
        
        Args:
            item_id: Reference item ID
            exclude_items: Item IDs to exclude from recommendations
            top_k: Number of recommendations to return
        
        Returns:
            List of recommendations with item_id and score
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before recommending")
        
        if item_id not in self.item_ids:
            return []
        
        item_idx = np.where(self.item_ids == item_id)[0][0]
        
        # Find neighbors (include the item itself, which we'll exclude)
        distances, indices = self.model.kneighbors(
            self.model._fit_X[item_idx:item_idx+1],
            n_neighbors=min(self.n_neighbors + 1, len(self.item_ids))
        )
        
        recommendations = []
        exclude_set = set(exclude_items or [])
        exclude_set.add(item_id)  # Always exclude the reference item
        
        for dist, idx in zip(distances[0], indices[0]):
            neighbor_item_id = self.item_ids[idx]
            
            if neighbor_item_id not in exclude_set:
                # Convert distance to similarity score (higher = more similar)
                score = 1.0 / (1.0 + dist) if self.metric != 'cosine' else 1.0 - dist
                recommendations.append({
                    'item_id': neighbor_item_id,
                    'score': float(score)
                })
            
            if len(recommendations) >= top_k:
                break
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def recommend_from_user_profile(
        self,
        user_profile: np.ndarray,
        exclude_items: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recommendations from user profile vector.
        
        Args:
            user_profile: User preference vector
            exclude_items: Item IDs to exclude
            top_k: Number of recommendations
        
        Returns:
            List of recommendations
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before recommending")
        
        # Normalize if needed
        if hasattr(self.scaler, 'mean_'):
            user_profile = self.scaler.transform([user_profile])[0]
        
        distances, indices = self.model.kneighbors(
            [user_profile],
            n_neighbors=min(self.n_neighbors + 1, len(self.item_ids))
        )
        
        recommendations = []
        exclude_set = set(exclude_items or [])
        
        for dist, idx in zip(distances[0], indices[0]):
            item_id = self.item_ids[idx]
            
            if item_id not in exclude_set:
                score = 1.0 / (1.0 + dist) if self.metric != 'cosine' else 1.0 - dist
                recommendations.append({
                    'item_id': item_id,
                    'score': float(score)
                })
            
            if len(recommendations) >= top_k:
                break
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'item_ids': self.item_ids,
                'scaler': self.scaler,
                'n_neighbors': self.n_neighbors,
                'metric': self.metric,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'KNNRecommender':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        recommender = cls(
            n_neighbors=data['n_neighbors'],
            metric=data['metric']
        )
        recommender.model = data['model']
        recommender.item_ids = data['item_ids']
        recommender.scaler = data['scaler']
        recommender.is_fitted = data['is_fitted']
        
        return recommender

