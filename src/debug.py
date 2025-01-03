import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from model.model import ChessModel
from dataset.dataset import prepare_data

from util.utils import n_params, safe_create_folder
from util.evaluator import Evaluator
from util.logger import ModelConfig, TrainingConfig
from util.accuracy import calculate_accuracy, calculate_top_N_accuracy

TEST_TRAIN = True
TEST_EVAL = True

def debug(options: TrainingConfig):
    
    # update experiment folder name
    options.experiment_folder = os.path.join(options.experiment_folder, "debug")
    
    mcfg = options.modelconfig
    
    model = ChessModel(
        num_embeddings=mcfg.dict_size, 
        transformer_width=mcfg.transformer_width, 
        transformer_layers=mcfg.transformer_layers, 
        transfromer_nheads=mcfg.transformer_nheads,
        dropout=mcfg.dropout
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=options.scheduler_steps, gamma=options.scheduler_decay)
    
    
    print(f"Model has {n_params(model)} parameters.")
    use_gpu = torch.cuda.is_available()
    print(f"Using GPU: {use_gpu}.\n")
    if use_gpu:
        model.gpu_mode()
    else:
        model.cpu_mode()
        
    dataset = prepare_data(batch_size=64)
    train_dl = dataset["train_dl"]
    test_dl = dataset["test_dl"]
    eval_dl = dataset["eval_dl"]
    token_dict = dataset["token_dict"]
    
    evaluator = Evaluator(options.early_stoping_step)
    
    ### training loop test
    if TEST_TRAIN:
        model.train_mode()
        print("Testing training")
        batch = next(iter(train_dl))
        src_input = batch["src_input"].to(model.device)
        tgt_input = batch["tgt_input"].to(model.device)
        tgt_output = batch["tgt_output"].to(model.device)
        
        output = model(src_input, tgt_input)
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)
        tgt_encoding = model.encode_moves(tgt_output)
        
        # normalized features
        output = output / output.norm(dim=1, keepdim=True)
        tgt_encoding = tgt_encoding / tgt_encoding.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logits_per_move = output @ tgt_encoding.t()
        logits_per_move_t = logits_per_move.t()
        labels = torch.arange(0, logits_per_move.shape[0]).to(model.device)

        ce_loss = (criterion(logits_per_move, labels) + criterion(logits_per_move_t, labels)) / 2
        sim_loss = 1 - F.cosine_similarity(output, tgt_encoding, dim=-1).mean()

        loss = ce_loss + sim_loss
        
        print("Loss:", loss)
            
        loss /= options.loss_accumulation_step # for loss accumulation
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
                
            
        with torch.no_grad():
            print("Testing test")
            
            model.eval_mode()
            encoded_move_set = model.encode_moves(torch.arange(0, len(token_dict)).to(model.device))
            batch = next(iter(test_dl))
            src_input = batch["src_input"].to(model.device)
            tgt_input = batch["tgt_input"].to(model.device)
            tgt_output = batch["tgt_output"].to(model.device)
            
            output = model(src_input, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            labels = tgt_output.reshape(-1)
        
            logits = output @ encoded_move_set.t()
            
            evaluator.run_evaluation("train accuracy", calculate_accuracy, (logits, labels))
            evaluator.run_evaluation("train top_5_acc", calculate_top_N_accuracy, (logits, labels, 5))
                
            _, accuracy = evaluator.step("train accuracy", len(test_dl), False)
            _, top_5_acc = evaluator.step("train top_5_acc", len(test_dl), False)
            
            print("Accuracy:", accuracy)
            print("Top 5 Accuracy:", top_5_acc)
            
            model.train_mode()
            evaluator.reset()
                    
        # step learning rate scheduler
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        # run eval dataset
        with torch.no_grad():
            print("Testing evaluation")
            
            model.eval_mode()
            encoded_move_set = model.encode_moves(torch.arange(0, len(token_dict)).to(model.device))
            batch = next(iter(eval_dl))
            src_input = batch["src_input"].to(model.device)
            tgt_input = batch["tgt_input"].to(model.device)
            tgt_output = batch["tgt_output"].to(model.device)
            
            output = model(src_input, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            labels = tgt_output.reshape(-1)
        
            logits = output @ encoded_move_set.t()
            
            evaluator.run_evaluation("accuracy", calculate_accuracy, (logits, labels))
            evaluator.run_evaluation("top_5_acc", calculate_top_N_accuracy, (logits, labels, 5))

            best_epoch, accuracy = evaluator.step("accuracy", len(test_dl))
            _, top_5_acc = evaluator.step("top_5_acc", len(test_dl), False)
            
            print("Accuracy:", accuracy)
            print("Top 5 Accuracy:", top_5_acc)
            
            if best_epoch:
                safe_create_folder(options.experiment_folder)
                model.save_checkpoint(options.experiment_folder, "best")
                
if __name__ == "__main__":
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
        batch_size=32,
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
    
    debug(options)