# Personalized Recommendation Engine + Automation Layer

A production-ready ML engineering system for personalized recommendations with real-time automation through n8n integration.

## Overview

This system combines multiple recommendation algorithms (KNN, Collaborative Filtering, Embeddings) with automated workflows to deliver personalized content recommendations across multiple domains (movies, music, tasks, notes).

### Key Features

- **Multiple ML Models**: KNN-based collaborative filtering, matrix factorization (ALS/BPR), and semantic embeddings
- **Data Warehouse**: DuckDB-based warehouse for scalable data storage and retrieval
- **Behavior Tracking**: Comprehensive user interaction logging and analysis
- **Automation Layer**: n8n workflows for automated recommendation delivery
- **Auto-Retraining**: Scheduled model retraining based on new user interactions
- **REST API**: FastAPI-based API for easy integration with n8n and other services

## System Architecture

```
┌─────────────────┐
│   n8n Workflows │
│  (Automation)   │
└────────┬────────┘
         │
         │ HTTP API
         ▼
┌─────────────────┐
│   FastAPI Server│
│  (REST API)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────────┐
│Behavior │ │Recommendation    │
│Tracker  │ │Engine            │
└────┬────┘ └────────┬─────────┘
     │               │
     │               │
     ▼               ▼
┌──────────────────────────┐
│    Data Warehouse        │
│    (DuckDB)              │
└──────────────────────────┘
```

## 🚀 Quick Start (See It In Action!)

The fastest way to see the system in action:

```bash
# Make quickstart script executable
chmod +x quickstart.sh

# Run the quickstart (installs deps, seeds data, trains models, starts server)
./quickstart.sh
```

This will:
1. ✅ Set up virtual environment and install dependencies
2. ✅ Seed sample data (users, movies, music, tasks)
3. ✅ Train all ML models
4. ✅ Start the API server at http://localhost:8000

Then in **another terminal**:

```bash
# Run the demo to see recommendations in action
python demo.py

# Or test the API endpoints
chmod +x test_api.sh
./test_api.sh

# Or visit the interactive API docs
open http://localhost:8000/docs
```

### Quick Demo Script

```bash
# Run the interactive demo
python demo.py
```

This shows:
- User statistics
- Movie recommendations
- Music recommendations  
- Task prioritization
- Semantic search example

## Installation

### Prerequisites

- Python 3.8+
- n8n (for automation workflows - optional)

### Manual Setup

1. **Clone the repository**:
```bash
git clone <repo-url>
cd mxn_learning
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Seed sample data**:
```bash
python -m src.seed_data
```

4. **Train initial models**:
```bash
python -m src.main --mode training
```

Or use the API:
```bash
curl -X POST http://localhost:8000/training/train
```

5. **Start the API server**:
```bash
python -m src.main --mode api
```

Or both API and training pipeline:
```bash
python -m src.main --mode both
```

## Usage

### API Endpoints

#### User Interactions

**Log interaction**:
```bash
POST /interactions
{
  "user_id": "user1",
  "domain": "movies",
  "item_id": "movie_1",
  "interaction_type": "like",
  "rating": 5.0
}
```

**Bulk log interactions**:
```bash
POST /interactions/bulk
{
  "interactions": [...]
}
```

#### Recommendations

**Get personalized recommendations**:
```bash
POST /recommendations
{
  "user_id": "user1",
  "domain": "movies",
  "top_k": 10,
  "model_types": ["knn", "collab", "embedding"],
  "query_text": "sci-fi action movies"
}
```

**Get daily playlist**:
```bash
GET /automation/daily-playlist/{user_id}
```

**Get weekly movies**:
```bash
GET /automation/weekly-movies/{user_id}
```

**Get task prioritization**:
```bash
GET /automation/task-prioritization/{user_id}
```

#### Training

**Trigger model training**:
```bash
POST /training/train?domain=movies
```

**Get training statistics**:
```bash
GET /training/stats?domain=movies
```

### n8n Integration

1. **Import workflows**:
   - Import JSON files from `n8n/` directory into your n8n instance
   - Configure environment variables:
     - `USER_ID`: Default user ID for recommendations
     - `MUSIC_SERVICE_WEBHOOK`: Webhook URL for music service
     - `MOVIE_SERVICE_WEBHOOK`: Webhook URL for movie service
     - `TASK_SERVICE_WEBHOOK`: Webhook URL for task service
     - `NOTIFICATION_SERVICE_WEBHOOK`: Webhook URL for notifications

2. **Available workflows**:
   - `daily-playlist-workflow.json`: Daily music playlist generation
   - `weekly-movies-workflow.json`: Weekly movie recommendations
   - `task-prioritization-workflow.json`: Daily task prioritization
   - `auto-retraining-workflow.json`: Automated model retraining

### Python API

```python
from src.recommendation_engine import RecommendationEngine
from src.behavior_tracker import BehaviorTracker

# Initialize
engine = RecommendationEngine()
tracker = BehaviorTracker(engine.warehouse)

# Track behavior
tracker.track_like(
    user_id="user1",
    domain="movies",
    item_id="movie_1",
    rating=5.0
)

# Get recommendations
recommendations = engine.recommend(
    user_id="user1",
    domain="movies",
    top_k=10
)

# Train models
engine.train_all_models("movies")
```

## Model Details

### 1. KNN Recommender
- **Type**: Item-based collaborative filtering
- **Algorithm**: k-Nearest Neighbors with cosine/euclidean distance
- **Use Case**: Similar item recommendations based on user-item matrix

### 2. Collaborative Filtering
- **Type**: Matrix factorization
- **Algorithm**: Alternating Least Squares (ALS) or Bayesian Personalized Ranking (BPR)
- **Use Case**: Personalized recommendations based on user-item interactions

### 3. Embedding Recommender
- **Type**: Semantic embeddings
- **Algorithm**: Sentence Transformers (all-MiniLM-L6-v2 by default)
- **Use Case**: Content-based recommendations using text descriptions

### Ensemble Strategy
The system combines recommendations from all available models using weighted averaging to provide robust, diverse recommendations.

## Configuration

Edit `config/config.yaml` to customize:

- **Domains**: Enable/disable domains and configure model parameters
- **API**: Server host and port
- **Training**: Retraining intervals and minimum samples
- **Automation**: Schedule and top-k settings for automated workflows

## Data Warehouse Schema

The system uses DuckDB with the following tables:

- `user_interactions`: All user interactions with items
- `users`: User metadata
- `items_catalog`: Item catalog with metadata and embeddings
- `recommendations`: Cached recommendations
- `training_history`: Model training logs and metrics

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

```
mxn_learning/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Main entry point
│   ├── api_server.py              # FastAPI server
│   ├── recommendation_engine.py   # Unified recommendation engine
│   ├── behavior_tracker.py        # User behavior tracking
│   ├── data_warehouse.py          # DuckDB data warehouse
│   ├── training_pipeline.py       # Auto-retraining pipeline
│   ├── seed_data.py               # Sample data generator
│   └── models/
│       ├── __init__.py
│       ├── knn_recommender.py
│       ├── collaborative_filtering.py
│       └── embedding_recommender.py
├── config/
│   └── config.yaml                # Configuration file
├── n8n/
│   ├── daily-playlist-workflow.json
│   ├── weekly-movies-workflow.json
│   ├── task-prioritization-workflow.json
│   └── auto-retraining-workflow.json
├── models/                        # Trained models (gitignored)
├── data/                          # Data warehouse (gitignored)
├── requirements.txt
└── README.md
```

## Production Considerations

1. **Model Storage**: Trained models are saved to `models/` directory. Consider using cloud storage for production.

2. **Data Warehouse**: DuckDB is file-based. For production, consider PostgreSQL or cloud data warehouses.

3. **API Security**: Add authentication/authorization for production deployment.

4. **Monitoring**: Add logging and metrics collection for production monitoring.

5. **Scaling**: Consider horizontal scaling with multiple API instances behind a load balancer.

6. **Model Versioning**: Implement model versioning and A/B testing for production.

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
