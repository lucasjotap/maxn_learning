"""Data Warehouse Layer for Recommendation Engine."""

import duckdb
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from pathlib import Path


class DataWarehouse:
    """DuckDB-based data warehouse for storing user behavior and recommendations."""
    
    def __init__(self, db_path: str = "data/warehouse.duckdb"):
        """Initialize data warehouse."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Create database schema if not exists."""
        # User interactions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                domain VARCHAR NOT NULL,
                item_id VARCHAR NOT NULL,
                item_metadata JSON,
                interaction_type VARCHAR NOT NULL,  -- 'view', 'like', 'complete', 'skip', etc.
                rating FLOAT,
                timestamp TIMESTAMP NOT NULL,
                context JSON
            )
        """)
        
        # Users table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR PRIMARY KEY,
                metadata JSON,
                created_at TIMESTAMP NOT NULL,
                last_active TIMESTAMP
            )
        """)
        
        # Items catalog table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS items_catalog (
                item_id VARCHAR NOT NULL,
                domain VARCHAR NOT NULL,
                title VARCHAR,
                description TEXT,
                metadata JSON,
                embedding BLOB,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP,
                PRIMARY KEY (item_id, domain)
            )
        """)
        
        # Recommendations cache
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                domain VARCHAR NOT NULL,
                item_id VARCHAR NOT NULL,
                score FLOAT NOT NULL,
                model_type VARCHAR NOT NULL,  -- 'knn', 'collab', 'embedding'
                generated_at TIMESTAMP NOT NULL,
                delivered BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Training history
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY,
                domain VARCHAR NOT NULL,
                model_type VARCHAR NOT NULL,
                training_date TIMESTAMP NOT NULL,
                metrics JSON,
                model_path VARCHAR,
                training_samples INTEGER
            )
        """)
        
        self.conn.commit()
    
    def log_interaction(
        self,
        user_id: str,
        domain: str,
        item_id: str,
        interaction_type: str,
        rating: Optional[float] = None,
        item_metadata: Optional[Dict] = None,
        context: Optional[Dict] = None
    ):
        """Log user interaction with an item."""
        self.conn.execute("""
            INSERT INTO user_interactions 
            (user_id, domain, item_id, item_metadata, interaction_type, rating, timestamp, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            user_id, domain, item_id,
            str(item_metadata) if item_metadata else None,
            interaction_type,
            rating,
            datetime.now(),
            str(context) if context else None
        ])
        self.conn.commit()
    
    def add_user(self, user_id: str, metadata: Optional[Dict] = None):
        """Add or update user."""
        self.conn.execute("""
            INSERT INTO users (user_id, metadata, created_at, last_active)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                metadata = EXCLUDED.metadata,
                last_active = EXCLUDED.last_active
        """, [
            user_id,
            str(metadata) if metadata else None,
            datetime.now(),
            datetime.now()
        ])
        self.conn.commit()
    
    def upsert_item(
        self,
        item_id: str,
        domain: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[bytes] = None
    ):
        """Add or update item in catalog."""
        self.conn.execute("""
            INSERT INTO items_catalog 
            (item_id, domain, title, description, metadata, embedding, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (item_id, domain) DO UPDATE SET
                title = COALESCE(EXCLUDED.title, items_catalog.title),
                description = COALESCE(EXCLUDED.description, items_catalog.description),
                metadata = COALESCE(EXCLUDED.metadata, items_catalog.metadata),
                embedding = COALESCE(EXCLUDED.embedding, items_catalog.embedding),
                updated_at = EXCLUDED.updated_at
        """, [
            item_id, domain, title, description,
            str(metadata) if metadata else None,
            embedding,
            datetime.now(),
            datetime.now()
        ])
        self.conn.commit()
    
    def get_user_interactions(
        self,
        user_id: Optional[str] = None,
        domain: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get user interactions as DataFrame."""
        query = "SELECT * FROM user_interactions WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        
        query += " ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        return self.conn.execute(query, params).df()
    
    def get_user_item_matrix(
        self,
        domain: str,
        interaction_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Get user-item interaction matrix for collaborative filtering."""
        query = """
            SELECT 
                user_id,
                item_id,
                COALESCE(AVG(rating), 1.0) as score
            FROM user_interactions
            WHERE domain = ?
        """
        params = [domain]
        
        if interaction_type:
            query += " AND interaction_type = ?"
            params.append(interaction_type)
        
        query += " GROUP BY user_id, item_id"
        
        return self.conn.execute(query, params).df()
    
    def get_items(self, domain: str) -> pd.DataFrame:
        """Get all items for a domain."""
        return self.conn.execute("""
            SELECT * FROM items_catalog WHERE domain = ?
        """, [domain]).df()
    
    def save_recommendations(
        self,
        user_id: str,
        domain: str,
        recommendations: List[Dict[str, Any]],
        model_type: str
    ):
        """Save generated recommendations."""
        for rec in recommendations:
            self.conn.execute("""
                INSERT INTO recommendations 
                (user_id, domain, item_id, score, model_type, generated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                user_id,
                domain,
                rec['item_id'],
                rec['score'],
                model_type,
                datetime.now()
            ])
        self.conn.commit()
    
    def get_recent_recommendations(
        self,
        user_id: str,
        domain: str,
        limit: int = 10
    ) -> pd.DataFrame:
        """Get recent recommendations for user."""
        return self.conn.execute("""
            SELECT * FROM recommendations
            WHERE user_id = ? AND domain = ?
            ORDER BY generated_at DESC
            LIMIT ?
        """, [user_id, domain, limit]).df()
    
    def log_training(
        self,
        domain: str,
        model_type: str,
        metrics: Dict,
        model_path: str,
        training_samples: int
    ):
        """Log training event."""
        self.conn.execute("""
            INSERT INTO training_history
            (domain, model_type, training_date, metrics, model_path, training_samples)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            domain,
            model_type,
            datetime.now(),
            str(metrics),
            model_path,
            training_samples
        ])
        self.conn.commit()
    
    def get_training_stats(self, domain: Optional[str] = None) -> pd.DataFrame:
        """Get training statistics."""
        query = "SELECT * FROM training_history WHERE 1=1"
        params = []
        
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        
        query += " ORDER BY training_date DESC"
        return self.conn.execute(query, params).df()
    
    def close(self):
        """Close database connection."""
        self.conn.close()

