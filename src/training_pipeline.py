"""Training Pipeline with Auto-Retraining Capability."""

import schedule
import time
from threading import Thread
from typing import Optional
import yaml
from datetime import datetime

from recommendation_engine import RecommendationEngine


class TrainingPipeline:
    """Automated training pipeline with scheduling."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize training pipeline."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.engine = RecommendationEngine(config_path)
        self.is_running = False
        self.thread: Optional[Thread] = None
    
    def train_domain(self, domain: str):
        """Train all models for a specific domain."""
        print(f"[{datetime.now()}] Starting training for {domain}...")
        try:
            self.engine.train_all_models(domain)
            print(f"[{datetime.now()}] Training completed for {domain}")
        except Exception as e:
            print(f"[{datetime.now()}] Error training {domain}: {e}")
    
    def train_all_domains(self):
        """Train all enabled domains."""
        domains = [
            domain for domain, config in self.engine.domains.items()
            if config.get('enabled', True)
        ]
        
        for domain in domains:
            self.train_domain(domain)
    
    def schedule_retraining(self):
        """Schedule automatic retraining."""
        retrain_interval = self.config['models'].get('retrain_interval_hours', 24)
        
        # Schedule retraining every N hours
        schedule.every(retrain_interval).hours.do(self.train_all_domains)
        
        print(f"Scheduled auto-retraining every {retrain_interval} hours")
    
    def _run_scheduler(self):
        """Run scheduler in background thread."""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start(self):
        """Start the training pipeline."""
        if self.is_running:
            print("Pipeline already running")
            return
        
        self.is_running = True
        self.schedule_retraining()
        
        # Initial training
        print("Running initial training...")
        self.train_all_domains()
        
        # Start scheduler thread
        self.thread = Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        
        print("Training pipeline started")
    
    def stop(self):
        """Stop the training pipeline."""
        self.is_running = False
        schedule.clear()
        print("Training pipeline stopped")
    
    def trigger_retrain(self, domain: Optional[str] = None):
        """Manually trigger retraining."""
        if domain:
            self.train_domain(domain)
        else:
            self.train_all_domains()

