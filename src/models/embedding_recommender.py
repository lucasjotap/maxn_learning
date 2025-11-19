"""Embedding-based Recommendation Model using Sentence Transformers."""

import numpy as np
from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch


class EmbeddingRecommender:
    """Semantic embedding-based recommender using transformer models."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """Initialize embedding recommender.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_name = model_name
        self.device = device
        self.encoder = SentenceTransformer(model_name, device=device)
        self.item_ids: Optional[List[str]] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.item_texts: Optional[List[str]] = None
        self.is_fitted = False
    
    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text to embeddings."""
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def fit(
        self,
        item_ids: List[str],
        item_texts: List[str],
        embeddings: Optional[np.ndarray] = None
    ):
        """Fit the embedding model.
        
        Args:
            item_ids: List of item IDs
            item_texts: List of text descriptions for items (title + description)
            embeddings: Pre-computed embeddings (optional, if None will compute)
        """
        if len(item_ids) != len(item_texts):
            raise ValueError("item_ids and item_texts must have same length")
        
        self.item_ids = item_ids
        self.item_texts = item_texts
        
        if embeddings is not None:
            if embeddings.shape[0] != len(item_ids):
                raise ValueError("embeddings must match item_ids length")
            self.item_embeddings = embeddings
        else:
            # Compute embeddings from texts
            self.item_embeddings = self._encode_text(item_texts)
        
        self.is_fitted = True
    
    def recommend(
        self,
        query_text: str,
        exclude_items: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on text query.
        
        Args:
            query_text: Query text describing what user wants
            exclude_items: Item IDs to exclude
            top_k: Number of recommendations
        
        Returns:
            List of recommendations with item_id and score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommending")
        
        # Encode query
        query_embedding = self._encode_text([query_text])[0]
        query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.item_embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1]
        
        exclude_set = set(exclude_items or [])
        recommendations = []
        
        for idx in top_indices:
            item_id = self.item_ids[idx]
            
            if item_id not in exclude_set:
                recommendations.append({
                    'item_id': item_id,
                    'score': float(similarities[idx])
                })
            
            if len(recommendations) >= top_k:
                break
        
        return recommendations
    
    def recommend_similar(
        self,
        item_id: str,
        exclude_items: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get items similar to given item based on embeddings.
        
        Args:
            item_id: Reference item ID
            exclude_items: Item IDs to exclude
            top_k: Number of recommendations
        
        Returns:
            List of similar items with item_id and score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        if item_id not in self.item_ids:
            return []
        
        item_idx = self.item_ids.index(item_id)
        item_embedding = self.item_embeddings[item_idx:item_idx+1]
        
        # Compute similarities
        similarities = cosine_similarity(item_embedding, self.item_embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1]
        
        exclude_set = set(exclude_items or [])
        exclude_set.add(item_id)  # Exclude self
        recommendations = []
        
        for idx in top_indices:
            similar_item_id = self.item_ids[idx]
            
            if similar_item_id not in exclude_set:
                recommendations.append({
                    'item_id': similar_item_id,
                    'score': float(similarities[idx])
                })
            
            if len(recommendations) >= top_k:
                break
        
        return recommendations
    
    def update_item(self, item_id: str, item_text: str):
        """Update or add a single item."""
        if not self.is_fitted:
            self.item_ids = []
            self.item_texts = []
            self.item_embeddings = np.array([]).reshape(0, self.encoder.get_sentence_embedding_dimension())
            self.is_fitted = True
        
        embedding = self._encode_text([item_text])[0]
        
        if item_id in self.item_ids:
            # Update existing
            idx = self.item_ids.index(item_id)
            self.item_texts[idx] = item_text
            self.item_embeddings[idx] = embedding
        else:
            # Add new
            self.item_ids.append(item_id)
            self.item_texts.append(item_text)
            self.item_embeddings = np.vstack([self.item_embeddings, embedding])
    
    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific item."""
        if not self.is_fitted or item_id not in self.item_ids:
            return None
        
        idx = self.item_ids.index(item_id)
        return self.item_embeddings[idx]
    
    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model_name': self.model_name,
                'item_ids': self.item_ids,
                'item_embeddings': self.item_embeddings,
                'item_texts': self.item_texts,
                'is_fitted': self.is_fitted
            }, f)
        
        # Note: SentenceTransformer model is not saved, will be reloaded from model_name
        # This keeps the file size manageable
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'EmbeddingRecommender':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        recommender = cls(
            model_name=data['model_name'],
            device=device
        )
        recommender.item_ids = data['item_ids']
        recommender.item_embeddings = data['item_embeddings']
        recommender.item_texts = data['item_texts']
        recommender.is_fitted = data['is_fitted']
        
        return recommender

