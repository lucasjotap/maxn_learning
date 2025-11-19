"""FastAPI Server for n8n Integration."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import yaml
from datetime import datetime

from recommendation_engine import RecommendationEngine
from behavior_tracker import BehaviorTracker
from training_pipeline import TrainingPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Recommendation Engine API",
    description="API for personalized recommendation engine with n8n integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config_path = "config/config.yaml"
engine = RecommendationEngine(config_path)
tracker = BehaviorTracker(engine.warehouse)
pipeline = TrainingPipeline(config_path)


# Pydantic models
class InteractionRequest(BaseModel):
    user_id: str
    domain: str
    item_id: str
    interaction_type: str = Field(..., pattern="^(view|like|complete|skip|rating)$")
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    item_metadata: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class ItemRequest(BaseModel):
    item_id: str
    domain: str
    title: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RecommendationRequest(BaseModel):
    user_id: str
    domain: str
    top_k: int = Field(10, ge=1, le=100)
    model_types: Optional[List[str]] = None
    query_text: Optional[str] = None
    exclude_interacted: bool = True


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    user_id: str
    domain: str
    generated_at: str


class BulkInteractionRequest(BaseModel):
    interactions: List[InteractionRequest]


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "Recommendation Engine API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/interactions", response_model=Dict[str, str])
def log_interaction(interaction: InteractionRequest):
    """Log user interaction."""
    try:
        if interaction.interaction_type == 'view':
            tracker.track_view(
                user_id=interaction.user_id,
                domain=interaction.domain,
                item_id=interaction.item_id,
                item_metadata=interaction.item_metadata,
                context=interaction.context
            )
        elif interaction.interaction_type == 'like':
            tracker.track_like(
                user_id=interaction.user_id,
                domain=interaction.domain,
                item_id=interaction.item_id,
                rating=interaction.rating or 5.0,
                item_metadata=interaction.item_metadata,
                context=interaction.context
            )
        elif interaction.interaction_type == 'complete':
            tracker.track_complete(
                user_id=interaction.user_id,
                domain=interaction.domain,
                item_id=interaction.item_id,
                rating=interaction.rating,
                item_metadata=interaction.item_metadata,
                context=interaction.context
            )
        elif interaction.interaction_type == 'skip':
            tracker.track_skip(
                user_id=interaction.user_id,
                domain=interaction.domain,
                item_id=interaction.item_id,
                item_metadata=interaction.item_metadata,
                context=interaction.context
            )
        elif interaction.interaction_type == 'rating':
            if not interaction.rating:
                raise HTTPException(status_code=400, detail="Rating required for rating interaction")
            tracker.track_rating(
                user_id=interaction.user_id,
                domain=interaction.domain,
                item_id=interaction.item_id,
                rating=interaction.rating,
                item_metadata=interaction.item_metadata,
                context=interaction.context
            )
        
        return {"status": "success", "message": "Interaction logged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interactions/bulk", response_model=Dict[str, str])
def log_bulk_interactions(request: BulkInteractionRequest):
    """Log multiple interactions at once."""
    try:
        interactions = []
        for interaction in request.interactions:
            interactions.append({
                'user_id': interaction.user_id,
                'domain': interaction.domain,
                'item_id': interaction.item_id,
                'interaction_type': interaction.interaction_type,
                'rating': interaction.rating,
                'item_metadata': interaction.item_metadata,
                'context': interaction.context
            })
        
        tracker.bulk_track(interactions)
        return {"status": "success", "message": f"{len(interactions)} interactions logged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/items", response_model=Dict[str, str])
def upsert_item(item: ItemRequest):
    """Add or update item in catalog."""
    try:
        engine.warehouse.upsert_item(
            item_id=item.item_id,
            domain=item.domain,
            title=item.title,
            description=item.description,
            metadata=item.metadata
        )
        return {"status": "success", "message": "Item upserted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations."""
    try:
        recommendations = engine.recommend(
            user_id=request.user_id,
            domain=request.domain,
            model_types=request.model_types,
            top_k=request.top_k,
            exclude_interacted=request.exclude_interacted,
            query_text=request.query_text
        )
        
        return RecommendationResponse(
            recommendations=recommendations,
            user_id=request.user_id,
            domain=request.domain,
            generated_at=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/{user_id}/{domain}")
def get_recent_recommendations(user_id: str, domain: str, limit: int = 10):
    """Get recent recommendations for user."""
    try:
        recs = engine.warehouse.get_recent_recommendations(
            user_id=user_id,
            domain=domain,
            limit=limit
        )
        return {"recommendations": recs.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/history")
def get_user_history(user_id: str, domain: Optional[str] = None, limit: Optional[int] = None):
    """Get user interaction history."""
    try:
        history = tracker.get_user_history(user_id, domain, limit)
        return {"history": history.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/stats")
def get_user_stats(user_id: str, domain: Optional[str] = None):
    """Get user statistics."""
    try:
        stats = tracker.get_user_stats(user_id, domain)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/train")
def trigger_training(domain: Optional[str] = None):
    """Manually trigger model training."""
    try:
        pipeline.trigger_retrain(domain)
        return {"status": "success", "message": f"Training triggered for {domain or 'all domains'}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/stats")
def get_training_stats(domain: Optional[str] = None):
    """Get training statistics."""
    try:
        stats = engine.warehouse.get_training_stats(domain)
        return {"stats": stats.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Specialized endpoints for automation
@app.get("/automation/daily-playlist/{user_id}")
def get_daily_playlist(user_id: str):
    """Get daily playlist recommendations."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    top_k = config['automation']['daily_playlist'].get('top_k', 20)
    
    try:
        recommendations = engine.recommend(
            user_id=user_id,
            domain='music',
            top_k=top_k
        )
        return {
            "user_id": user_id,
            "domain": "music",
            "playlist": [rec['item_id'] for rec in recommendations],
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/automation/weekly-movies/{user_id}")
def get_weekly_movies(user_id: str):
    """Get weekly movie recommendations."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    top_k = config['automation']['weekly_movies'].get('top_k', 5)
    
    try:
        recommendations = engine.recommend(
            user_id=user_id,
            domain='movies',
            top_k=top_k
        )
        return {
            "user_id": user_id,
            "domain": "movies",
            "movies": [rec['item_id'] for rec in recommendations],
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/automation/task-prioritization/{user_id}")
def get_task_prioritization(user_id: str):
    """Get task prioritization recommendations."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    top_k = config['automation']['task_prioritization'].get('top_k', 10)
    
    try:
        recommendations = engine.recommend(
            user_id=user_id,
            domain='tasks',
            top_k=top_k
        )
        return {
            "user_id": user_id,
            "domain": "tasks",
            "prioritized_tasks": [rec['item_id'] for rec in recommendations],
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port']
    )

