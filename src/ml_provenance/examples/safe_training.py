import torch
import torch.nn as nn
from ..frameworks.pytorch.wrapper import PyTorchSafetyWrapper
from ..frameworks.pytorch.trainer import PyTorchSafetyTrainer
from ..safety_checks import SafetyConfig

# Example model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.layers(x)

def train_with_safety():
    # Create model
    model = SimpleModel()
    
    # Configure safety
    safety_config = SafetyConfig(
        min_age_rating="ALL_AGES",
        content_filters=["violence", "explicit"],
        max_input_length=1000,
        max_output_length=2000,
        block_sensitive_topics=True,
        require_content_warning=False
    )
    
    # Training configuration
    training_config = {
        "safety_config": safety_config,
        "safety_weight": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    }
    
    # Create safety trainer
    trainer = PyTorchSafetyTrainer(model, training_config)
    
    # Example training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    
    for epoch in range(training_config["epochs"]):
        for batch_idx, batch in enumerate(train_loader):  # You need to define train_loader
            # Training step with safety checks
            loss = trainer.training_step(batch, batch_idx)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log safety metrics
            if batch_idx % 100 == 0:
                safety_metrics = trainer.get_safety_metrics()
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"Safety metrics: {safety_metrics}")
                
    # Save safety provenance
    trainer.safety_wrapper.safety_checker.save_safety_provenance(
        "safety_provenance.json"
    )

if __name__ == "__main__":
    train_with_safety() 