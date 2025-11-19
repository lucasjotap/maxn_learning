"""Script to seed initial data for testing and development."""

import random
from datetime import datetime, timedelta
from recommendation_engine import RecommendationEngine
from behavior_tracker import BehaviorTracker


def seed_sample_data():
    """Seed sample data for all domains."""
    engine = RecommendationEngine()
    tracker = BehaviorTracker(engine.warehouse)
    
    # Sample users
    users = ["user1", "user2", "user3", "user4", "user5"]
    
    # Movies data
    movies = [
        {"item_id": "movie_1", "title": "The Matrix", "description": "Sci-fi action movie about virtual reality"},
        {"item_id": "movie_2", "title": "Inception", "description": "Mind-bending thriller about dreams"},
        {"item_id": "movie_3", "title": "Interstellar", "description": "Space epic about saving humanity"},
        {"item_id": "movie_4", "title": "The Dark Knight", "description": "Superhero movie with Batman"},
        {"item_id": "movie_5", "title": "Pulp Fiction", "description": "Crime drama with non-linear narrative"},
        {"item_id": "movie_6", "title": "Fight Club", "description": "Psychological thriller about consumerism"},
        {"item_id": "movie_7", "title": "The Shawshank Redemption", "description": "Prison drama about hope"},
        {"item_id": "movie_8", "title": "Forrest Gump", "description": "Drama about an extraordinary life"},
    ]
    
    # Music data
    songs = [
        {"item_id": "song_1", "title": "Bohemian Rhapsody", "description": "Rock opera by Queen"},
        {"item_id": "song_2", "title": "Stairway to Heaven", "description": "Classic rock by Led Zeppelin"},
        {"item_id": "song_3", "title": "Hotel California", "description": "Rock song by Eagles"},
        {"item_id": "song_4", "title": "Sweet Child O' Mine", "description": "Rock ballad by Guns N' Roses"},
        {"item_id": "song_5", "title": "Comfortably Numb", "description": "Progressive rock by Pink Floyd"},
        {"item_id": "song_6", "title": "Billie Jean", "description": "Pop by Michael Jackson"},
        {"item_id": "song_7", "title": "Like a Rolling Stone", "description": "Folk rock by Bob Dylan"},
        {"item_id": "song_8", "title": "Smells Like Teen Spirit", "description": "Grunge by Nirvana"},
    ]
    
    # Tasks data
    tasks = [
        {"item_id": "task_1", "title": "Review quarterly report", "description": "Financial analysis for Q4"},
        {"item_id": "task_2", "title": "Prepare presentation", "description": "Client meeting slides"},
        {"item_id": "task_3", "title": "Code review", "description": "Review PR #123"},
        {"item_id": "task_4", "title": "Team meeting", "description": "Weekly standup"},
        {"item_id": "task_5", "title": "Update documentation", "description": "API documentation"},
        {"item_id": "task_6", "title": "Fix bug #456", "description": "Critical issue"},
        {"item_id": "task_7", "title": "Write tests", "description": "Unit tests for new feature"},
        {"item_id": "task_8", "title": "Deploy to staging", "description": "Release v2.1.0"},
    ]
    
    # Notes data
    notes = [
        {"item_id": "note_1", "title": "Project ideas", "description": "Brainstorming session notes"},
        {"item_id": "note_2", "title": "Meeting notes", "description": "Product planning discussion"},
        {"item_id": "note_3", "title": "Research findings", "description": "Market analysis results"},
        {"item_id": "note_4", "title": "Learning resources", "description": "ML engineering books"},
        {"item_id": "note_5", "title": "Conference takeaways", "description": "Key insights from event"},
        {"item_id": "note_6", "title": "Architecture notes", "description": "System design decisions"},
    ]
    
    # Add items to catalog
    print("Adding items to catalog...")
    for item in movies:
        engine.warehouse.upsert_item(
            item_id=item["item_id"],
            domain="movies",
            title=item["title"],
            description=item["description"]
        )
    
    for item in songs:
        engine.warehouse.upsert_item(
            item_id=item["item_id"],
            domain="music",
            title=item["title"],
            description=item["description"]
        )
    
    for item in tasks:
        engine.warehouse.upsert_item(
            item_id=item["item_id"],
            domain="tasks",
            title=item["title"],
            description=item["description"]
        )
    
    for item in notes:
        engine.warehouse.upsert_item(
            item_id=item["item_id"],
            domain="notes",
            title=item["title"],
            description=item["description"]
        )
    
    # Generate interactions
    print("Generating user interactions...")
    interactions_per_user = 20
    
    for user in users:
        # Movie interactions
        for _ in range(interactions_per_user):
            movie = random.choice(movies)
            interaction_type = random.choice(["view", "like", "rating"])
            
            if interaction_type == "rating":
                tracker.track_rating(
                    user_id=user,
                    domain="movies",
                    item_id=movie["item_id"],
                    rating=random.uniform(3.0, 5.0)
                )
            elif interaction_type == "like":
                tracker.track_like(
                    user_id=user,
                    domain="movies",
                    item_id=movie["item_id"],
                    rating=5.0
                )
            else:
                tracker.track_view(
                    user_id=user,
                    domain="movies",
                    item_id=movie["item_id"]
                )
        
        # Music interactions
        for _ in range(interactions_per_user):
            song = random.choice(songs)
            interaction_type = random.choice(["view", "like", "complete"])
            
            if interaction_type == "like":
                tracker.track_like(
                    user_id=user,
                    domain="music",
                    item_id=song["item_id"]
                )
            elif interaction_type == "complete":
                tracker.track_complete(
                    user_id=user,
                    domain="music",
                    item_id=song["item_id"]
                )
            else:
                tracker.track_view(
                    user_id=user,
                    domain="music",
                    item_id=song["item_id"]
                )
        
        # Task interactions
        for _ in range(interactions_per_user):
            task = random.choice(tasks)
            interaction_type = random.choice(["view", "complete"])
            
            if interaction_type == "complete":
                tracker.track_complete(
                    user_id=user,
                    domain="tasks",
                    item_id=task["item_id"]
                )
            else:
                tracker.track_view(
                    user_id=user,
                    domain="tasks",
                    item_id=task["item_id"]
                )
    
    print("Sample data seeded successfully!")
    print(f"Users: {len(users)}")
    print(f"Movies: {len(movies)}")
    print(f"Songs: {len(songs)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Notes: {len(notes)}")
    print("\nTo train models, run:")
    print("  python -m src.recommendation_engine")
    print("\nOr use the API:")
    print("  POST /training/train")


if __name__ == "__main__":
    seed_sample_data()

