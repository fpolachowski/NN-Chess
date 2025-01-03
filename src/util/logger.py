import wandb
from dataclasses import dataclass

@dataclass
class ModelConfig():
    dict_size: int
    transformer_width: int
    transformer_layers: int
    transformer_nheads: int
    dropout: float

@dataclass
class TrainingConfig():
    learning_rate: float
    epochs: int
    batch_size: int
    modelconfig: ModelConfig
    early_stoping_step: int
    loss_accumulation_step: int
    scheduler_steps: list[int]
    scheduler_decay: float
    model_save_interval: int
    experiment_folder: str
    debug_flag: bool
    files: list[str]

class WandbLogger():
    def __init__(self, name, options: TrainingConfig) -> None:
        self.options = options
        wandb.login()
        self.run = wandb.init(
            # Set the project where this run will be logged
            project=name,
            # Track hyperparameters and run metadata
            config={
                "epochs": options.epochs,
                "learning_rate": options.learning_rate,
                "scheduler_steps": options.scheduler_steps,
                "scheduler_decay": options.scheduler_decay,
                "batch_size": options.batch_size,
                "early_stoping_step": options.early_stoping_step
            },
            settings=wandb.Settings(disable_git=True)
        )
        
        for f in options.files:
            wandb.save(f)
        
        
            
    def log(self, value_dict: dict[str, float]):
        for name in value_dict:
            self.run.log({name: value_dict[name]})