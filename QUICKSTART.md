# 🚀 Quick Start Guide - See It In Action!

This guide will help you see the recommendation engine in action in just a few minutes.

## Option 1: Automated Quickstart (Easiest)

```bash
# Make script executable
chmod +x quickstart.sh

# Run everything automatically
./quickstart.sh
```

This script will:
1. Create a virtual environment
2. Install all dependencies
3. Seed sample data (5 users, 8 movies, 8 songs, 8 tasks)
4. Train all ML models (KNN, Collaborative Filtering, Embeddings)
5. Start the API server at http://localhost:8000

**Wait for the script to finish**, then open a **new terminal** and run:

```bash
# Option A: Run the interactive demo
python demo.py

# Option B: Test the API with curl
./test_api.sh

# Option C: Open interactive API docs in your browser
# Visit: http://localhost:8000/docs
```

## Option 2: Step-by-Step Manual Setup

### Step 1: Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Seed Sample Data

```bash
python -m src.seed_data
```

This creates:
- 5 sample users
- 8 movies (The Matrix, Inception, Interstellar, etc.)
- 8 songs (Bohemian Rhapsody, Stairway to Heaven, etc.)
- 8 tasks (Review report, Prepare presentation, etc.)
- 20+ interactions per user per domain

### Step 3: Train Models

```bash
# Train all models (this downloads the embedding model first time)
python -c "
from src.recommendation_engine import RecommendationEngine
engine = RecommendationEngine()
for domain in ['movies', 'music', 'tasks']:
    print(f'Training {domain}...')
    engine.train_all_models(domain)
    print(f'✓ {domain} done')
"
```

This trains:
- KNN models for item similarity
- Collaborative filtering models (ALS)
- Embedding models using sentence transformers

### Step 4: Run the Demo

```bash
python demo.py
```

You'll see:
- User statistics
- Personalized movie recommendations
- Music playlist suggestions
- Task prioritization
- Semantic search examples

### Step 5: Start the API Server

```bash
python -m src.main --mode api
```

The server will start at http://localhost:8000

## Testing the API

### Interactive API Docs

Visit http://localhost:8000/docs in your browser for interactive API documentation.

### Test Endpoints with curl

```bash
# Health check
curl http://localhost:8000/health

# Get daily playlist for user1
curl http://localhost:8000/automation/daily-playlist/user1

# Get weekly movies
curl http://localhost:8000/automation/weekly-movies/user1

# Get task prioritization
curl http://localhost:8000/automation/task-prioritization/user1

# Get personalized recommendations
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user1",
    "domain": "movies",
    "top_k": 5
  }'

# Log an interaction
curl -X POST http://localhost:8000/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user1",
    "domain": "movies",
    "item_id": "movie_1",
    "interaction_type": "like",
    "rating": 5.0
  }'
```

### Run Test Script

```bash
chmod +x test_api.sh
./test_api.sh
```

## Example Output

### Demo Script Output

```
🎬 Recommendation Engine Demo
============================================================

📊 User Statistics
============================================================
Total interactions: 60
Average rating: 4.2

Per-domain stats:
  movies: 20 interactions, 5 likes, 0 completes
  music: 20 interactions, 8 likes, 3 completes
  tasks: 20 interactions, 0 likes, 12 completes

🎯 Getting Movie Recommendations
============================================================
User: user1
Domain: movies

Recommendations:
------------------------------------------------------------
  1. Item: movie_3
     Score: 0.8542
     Models: knn, collab, embedding

  2. Item: movie_2
     Score: 0.8123
     Models: knn, collab

  ...
```

## Next Steps

1. **Customize for your data**: Replace seed data with your own items and interactions
2. **Integrate with n8n**: Import the workflows from `n8n/` directory
3. **Add more domains**: Extend to other content types (books, articles, products, etc.)
4. **Tune models**: Adjust parameters in `config/config.yaml`

## Troubleshooting

### Models not found
If you see "Model must be fitted" errors, run the training step:
```bash
curl -X POST http://localhost:8000/training/train
```

### Port already in use
Change the port in `config/config.yaml`:
```yaml
api:
  port: 8001  # Change from 8000
```

### Out of memory
If you run out of memory during training:
- Reduce `knn_neighbors` in `config/config.yaml`
- Use smaller embedding model
- Train domains one at a time

## Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review API docs at http://localhost:8000/docs
- Check logs in the terminal output

