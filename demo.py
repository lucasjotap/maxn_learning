#!/usr/bin/env python3
"""Demo script to showcase the recommendation engine in action."""

import json
from src.recommendation_engine import RecommendationEngine
from src.behavior_tracker import BehaviorTracker

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_recommendations(recs, title="Recommendations"):
    """Print formatted recommendations."""
    print(f"\n{title}:")
    print("-" * 60)
    if not recs:
        print("  No recommendations available")
        return
    
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. Item: {rec['item_id']}")
        print(f"     Score: {rec['score']:.4f}")
        if 'model_types' in rec:
            print(f"     Models: {', '.join(rec['model_types'])}")
        print()

def main():
    """Run the demo."""
    print_section("🎬 Recommendation Engine Demo")
    
    # Initialize components
    print("Initializing recommendation engine...")
    engine = RecommendationEngine()
    tracker = BehaviorTracker(engine.warehouse)
    
    user_id = "user1"
    domain = "movies"
    
    # Check if we have data
    interactions = tracker.get_user_history(user_id, domain, limit=1)
    if interactions.empty:
        print("\n⚠️  No data found. Run 'python -m src.seed_data' first to generate sample data.")
        print("   Then train models with: POST /training/train")
        return
    
    print_section("📊 User Statistics")
    stats = tracker.get_user_stats(user_id)
    print(f"Total interactions: {stats['total_interactions']}")
    print(f"Average rating: {stats.get('avg_rating', 'N/A'):.2f}" if stats.get('avg_rating') else "Average rating: N/A")
    print("\nPer-domain stats:")
    for dom, dom_stats in stats.get('domains', {}).items():
        print(f"  {dom}: {dom_stats['interactions']} interactions, "
              f"{dom_stats.get('likes', 0)} likes, "
              f"{dom_stats.get('completes', 0)} completes")
    
    print_section("🎯 Getting Movie Recommendations")
    print(f"User: {user_id}")
    print(f"Domain: {domain}")
    
    recommendations = engine.recommend(
        user_id=user_id,
        domain=domain,
        top_k=5
    )
    
    print_recommendations(recommendations, f"Top 5 Movie Recommendations for {user_id}")
    
    print_section("🎵 Getting Music Recommendations")
    music_recs = engine.recommend(
        user_id=user_id,
        domain="music",
        top_k=5
    )
    print_recommendations(music_recs, f"Top 5 Music Recommendations for {user_id}")
    
    print_section("📝 Getting Task Prioritization")
    task_recs = engine.recommend(
        user_id=user_id,
        domain="tasks",
        top_k=5
    )
    print_recommendations(task_recs, f"Top 5 Task Recommendations for {user_id}")
    
    print_section("🔍 Semantic Search Example")
    # Try semantic search with text query
    query_recs = engine.recommend(
        user_id=user_id,
        domain="movies",
        top_k=3,
        query_text="sci-fi action movie with virtual reality"
    )
    print(f"Query: 'sci-fi action movie with virtual reality'")
    print_recommendations(query_recs, "Semantic Search Results")
    
    print_section("📈 Training History")
    training_stats = engine.warehouse.get_training_stats()
    if not training_stats.empty:
        print("\nRecent training runs:")
        for _, row in training_stats.head(5).iterrows():
            print(f"  {row['domain']} - {row['model_type']} - {row['training_date']}")
            print(f"    Samples: {row['training_samples']}")
    else:
        print("  No training history yet")
    
    print_section("✨ Demo Complete!")
    print("\n💡 Next steps:")
    print("   1. Start the API server: python -m src.main --mode api")
    print("   2. Visit http://localhost:8000/docs for interactive API docs")
    print("   3. Test with curl or import n8n workflows")
    print("\n")

if __name__ == "__main__":
    main()

