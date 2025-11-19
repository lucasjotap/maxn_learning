#!/bin/bash
# Quick Start Script for Recommendation Engine

set -e

echo "🚀 Starting Recommendation Engine Quick Start..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Seed sample data
echo "🌱 Seeding sample data..."
python -m src.seed_data

# Train initial models
echo "🧠 Training initial models (this may take a few minutes)..."
python << 'PYTHON_SCRIPT'
from src.recommendation_engine import RecommendationEngine
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

engine = RecommendationEngine()
domains = ['movies', 'music', 'tasks']
for domain in domains:
    print(f'  Training {domain}...')
    try:
        engine.train_all_models(domain)
        print(f'  ✓ {domain} trained successfully')
    except Exception as e:
        print(f'  ⚠ {domain} training failed: {e}')
print('✅ Model training complete!')
PYTHON_SCRIPT

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Starting API server..."
echo "   API will be available at: http://localhost:8000"
echo "   API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the API server
python -m src.main --mode api

