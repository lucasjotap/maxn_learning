"""Main entry point for Recommendation Engine."""

import argparse
import signal
import sys
from pathlib import Path

from training_pipeline import TrainingPipeline
from api_server import app
import uvicorn
import yaml


def main():
    """Main function to run the recommendation engine."""
    parser = argparse.ArgumentParser(description="Personalized Recommendation Engine")
    parser.add_argument(
        "--mode",
        choices=["api", "training", "both"],
        default="both",
        help="Mode to run: api server, training pipeline, or both"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize training pipeline
    pipeline = None
    if args.mode in ["training", "both"]:
        pipeline = TrainingPipeline(str(config_path))
        pipeline.start()
        print("Training pipeline started")
    
    # Run API server
    if args.mode in ["api", "both"]:
        api_config = config.get('api', {})
        host = api_config.get('host', '0.0.0.0')
        port = api_config.get('port', 8000)
        
        print(f"Starting API server on {host}:{port}")
        
        # Handle shutdown gracefully
        def shutdown_handler(sig, frame):
            print("\nShutting down...")
            if pipeline:
                pipeline.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=api_config.get('reload', False)
        )
    else:
        # If only training mode, keep running
        try:
            while True:
                import time
                time.sleep(60)
        except KeyboardInterrupt:
            if pipeline:
                pipeline.stop()


if __name__ == "__main__":
    main()

