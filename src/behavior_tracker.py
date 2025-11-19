"""Behavior Tracking Layer for User Interactions."""

from typing import Dict, Optional, Any, List
from datetime import datetime
from data_warehouse import DataWarehouse


class BehaviorTracker:
    """Track user behavior and interactions."""
    
    def __init__(self, warehouse: DataWarehouse):
        """Initialize behavior tracker."""
        self.warehouse = warehouse
    
    def track_view(
        self,
        user_id: str,
        domain: str,
        item_id: str,
        item_metadata: Optional[Dict] = None,
        context: Optional[Dict] = None
    ):
        """Track item view."""
        self.warehouse.log_interaction(
            user_id=user_id,
            domain=domain,
            item_id=item_id,
            interaction_type='view',
            item_metadata=item_metadata,
            context=context
        )
        self.warehouse.add_user(user_id)
    
    def track_like(
        self,
        user_id: str,
        domain: str,
        item_id: str,
        rating: float = 5.0,
        item_metadata: Optional[Dict] = None,
        context: Optional[Dict] = None
    ):
        """Track item like/favorite."""
        self.warehouse.log_interaction(
            user_id=user_id,
            domain=domain,
            item_id=item_id,
            interaction_type='like',
            rating=rating,
            item_metadata=item_metadata,
            context=context
        )
        self.warehouse.add_user(user_id)
    
    def track_complete(
        self,
        user_id: str,
        domain: str,
        item_id: str,
        rating: Optional[float] = None,
        item_metadata: Optional[Dict] = None,
        context: Optional[Dict] = None
    ):
        """Track task/item completion."""
        self.warehouse.log_interaction(
            user_id=user_id,
            domain=domain,
            item_id=item_id,
            interaction_type='complete',
            rating=rating,
            item_metadata=item_metadata,
            context=context
        )
        self.warehouse.add_user(user_id)
    
    def track_skip(
        self,
        user_id: str,
        domain: str,
        item_id: str,
        item_metadata: Optional[Dict] = None,
        context: Optional[Dict] = None
    ):
        """Track item skip."""
        self.warehouse.log_interaction(
            user_id=user_id,
            domain=domain,
            item_id=item_id,
            interaction_type='skip',
            rating=1.0,
            item_metadata=item_metadata,
            context=context
        )
        self.warehouse.add_user(user_id)
    
    def track_rating(
        self,
        user_id: str,
        domain: str,
        item_id: str,
        rating: float,
        item_metadata: Optional[Dict] = None,
        context: Optional[Dict] = None
    ):
        """Track explicit rating."""
        if not (1.0 <= rating <= 5.0):
            raise ValueError("Rating must be between 1.0 and 5.0")
        
        self.warehouse.log_interaction(
            user_id=user_id,
            domain=domain,
            item_id=item_id,
            interaction_type='rating',
            rating=rating,
            item_metadata=item_metadata,
            context=context
        )
        self.warehouse.add_user(user_id)
    
    def bulk_track(
        self,
        interactions: List[Dict[str, Any]]
    ):
        """Bulk track multiple interactions."""
        for interaction in interactions:
            self.warehouse.log_interaction(
                user_id=interaction['user_id'],
                domain=interaction['domain'],
                item_id=interaction['item_id'],
                interaction_type=interaction.get('interaction_type', 'view'),
                rating=interaction.get('rating'),
                item_metadata=interaction.get('item_metadata'),
                context=interaction.get('context')
            )
            self.warehouse.add_user(interaction['user_id'])
    
    def get_user_history(
        self,
        user_id: str,
        domain: Optional[str] = None,
        limit: Optional[int] = None
    ):
        """Get user's interaction history."""
        return self.warehouse.get_user_interactions(
            user_id=user_id,
            domain=domain,
            limit=limit
        )
    
    def get_user_stats(
        self,
        user_id: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user statistics."""
        history = self.get_user_history(user_id, domain)
        
        if history.empty:
            return {
                'total_interactions': 0,
                'domains': {},
                'avg_rating': None
            }
        
        stats = {
            'total_interactions': len(history),
            'domains': {},
            'avg_rating': None
        }
        
        # Per-domain stats
        for dom in history['domain'].unique():
            dom_history = history[history['domain'] == dom]
            stats['domains'][dom] = {
                'interactions': len(dom_history),
                'avg_rating': dom_history['rating'].mean() if 'rating' in dom_history.columns else None,
                'likes': len(dom_history[dom_history['interaction_type'] == 'like']),
                'completes': len(dom_history[dom_history['interaction_type'] == 'complete'])
            }
        
        # Overall average rating
        if 'rating' in history.columns:
            stats['avg_rating'] = history['rating'].mean()
        
        return stats

