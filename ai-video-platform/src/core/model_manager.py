import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import json
import os
from pathlib import Path
import logging
from datetime import datetime

class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_model(
        self,
        model_id: str,
        model: nn.Module,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new model with its configuration and metadata."""
        self.models[model_id] = {
            "model": model,
            "config": config,
            "metadata": metadata or {},
            "last_updated": datetime.now().isoformat()
        }
        self.logger.info(f"Registered model: {model_id}")
    
    def load_model(
        self,
        model_id: str,
        checkpoint_path: Optional[str] = None
    ) -> nn.Module:
        """Load a model from disk or return cached version."""
        if model_id in self.models:
            return self.models[model_id]["model"]
        
        if checkpoint_path is None:
            checkpoint_path = self.model_dir / f"{model_id}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        # Load model configuration
        config_path = checkpoint_path.parent / f"{model_id}_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}
        
        # Load model state
        model = self._create_model_from_config(config)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        # Register loaded model
        self.register_model(model_id, model, config)
        
        return model
    
    def save_model(
        self,
        model_id: str,
        checkpoint_path: Optional[str] = None
    ) -> None:
        """Save model state and configuration to disk."""
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")
        
        model_data = self.models[model_id]
        
        if checkpoint_path is None:
            checkpoint_path = self.model_dir / f"{model_id}.pt"
        
        # Save model state
        torch.save(model_data["model"].state_dict(), checkpoint_path)
        
        # Save configuration
        config_path = checkpoint_path.parent / f"{model_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_data["config"], f, indent=2)
        
        self.logger.info(f"Saved model: {model_id}")
    
    def update_model(
        self,
        model_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update model configuration or metadata."""
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")
        
        model_data = self.models[model_id]
        
        if "config" in updates:
            model_data["config"].update(updates["config"])
        
        if "metadata" in updates:
            model_data["metadata"].update(updates["metadata"])
        
        model_data["last_updated"] = datetime.now().isoformat()
        self.logger.info(f"Updated model: {model_id}")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information including configuration and metadata."""
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")
        
        return {
            "model_id": model_id,
            **self.models[model_id]
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models with their information."""
        return [
            {
                "model_id": model_id,
                **model_data
            }
            for model_id, model_data in self.models.items()
        ]
    
    def delete_model(self, model_id: str) -> None:
        """Delete a model and its associated files."""
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")
        
        # Delete model files
        checkpoint_path = self.model_dir / f"{model_id}.pt"
        config_path = self.model_dir / f"{model_id}_config.json"
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if config_path.exists():
            config_path.unlink()
        
        # Remove from memory
        del self.models[model_id]
        self.logger.info(f"Deleted model: {model_id}")
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> nn.Module:
        """Create a model instance from configuration."""
        model_type = config.get("type")
        
        if model_type == "resnet50":
            return models.resnet50(pretrained=False)
        elif model_type == "fasterrcnn":
            return models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        elif model_type == "whisper":
            return WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3"
            )
        elif model_type == "mbart":
            return AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt"
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def optimize_model(
        self,
        model_id: str,
        optimization_config: Dict[str, Any]
    ) -> None:
        """Optimize model for inference."""
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")
        
        model = self.models[model_id]["model"]
        
        # Apply optimizations
        if optimization_config.get("quantization"):
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        
        if optimization_config.get("fuse"):
            model = torch.quantization.fuse_modules(
                model,
                [['conv', 'bn', 'relu']]
            )
        
        # Update model in registry
        self.models[model_id]["model"] = model
        self.models[model_id]["config"]["optimization"] = optimization_config
        self.models[model_id]["last_updated"] = datetime.now().isoformat()
        
        self.logger.info(f"Optimized model: {model_id}")
    
    def validate_model(
        self,
        model_id: str,
        validation_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Validate model performance on test data."""
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")
        
        model = self.models[model_id]["model"]
        model.eval()
        
        metrics = {}
        
        with torch.no_grad():
            # Implement validation logic based on model type
            if isinstance(model, models.ResNet):
                metrics = self._validate_classification_model(
                    model,
                    validation_data
                )
            elif isinstance(model, models.detection.FasterRCNN):
                metrics = self._validate_detection_model(
                    model,
                    validation_data
                )
            elif isinstance(model, WhisperForConditionalGeneration):
                metrics = self._validate_transcription_model(
                    model,
                    validation_data
                )
        
        # Update model metadata with validation results
        self.models[model_id]["metadata"]["validation"] = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        return metrics
    
    def _validate_classification_model(
        self,
        model: nn.Module,
        validation_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Validate classification model."""
        # Implement classification validation
        return {}
    
    def _validate_detection_model(
        self,
        model: nn.Module,
        validation_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Validate object detection model."""
        # Implement detection validation
        return {}
    
    def _validate_transcription_model(
        self,
        model: nn.Module,
        validation_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Validate speech transcription model."""
        # Implement transcription validation
        return {} 