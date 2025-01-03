from train_model import train
from util.logger import ModelConfig, TrainingConfig

mcfg = ModelConfig(
    dict_size=1900,
    transformer_width=512,
    transformer_layers=12,
    transformer_nheads=8,
    dropout=0.1
)

options = TrainingConfig(
    learning_rate=1e-4,
    epochs=20,
    batch_size=64,
    modelconfig=mcfg,
    scheduler_steps=[12, 18],
    loss_accumulation_step=2,
    scheduler_decay=0.1,
    model_save_interval=5,
    early_stoping_step=3,
    experiment_folder="experiments",
    debug_flag = False,
    files=["src/model/model.py", "src/dataset/dataset.py", "src/train_model.py"],
)

if __name__ == "__main__":
    train(options)