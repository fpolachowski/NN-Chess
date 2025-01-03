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
from util.logger import TrainingConfig
from util.accuracy import calculate_accuracy, calculate_top_N_accuracy


def train(options: TrainingConfig):
    from util.logger import WandbLogger
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    logger = WandbLogger(name="NN-Chess", options=options)
    
    # update experiment folder name
    options.experiment_folder = os.path.join(options.experiment_folder, f"{logger.run.name}")
    
    # disable torch features for speedup
    torch.autograd.set_detect_anomaly(mode=False)
    torch.autograd.profiler.profile(enabled=False)
    
    # enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    
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
        
    dataset = prepare_data(batch_size=options.batch_size)
    train_dl = dataset["train_dl"]
    test_dl = dataset["test_dl"]
    eval_dl = dataset["eval_dl"]
    token_dict = dataset["token_dict"]
    
    evaluator = Evaluator(options.early_stoping_step)
    
    for epoch in range(1, options.epochs + 1):
        model.train_mode()
        print("Training epoch {} with lr {} (0/{})".format(epoch, optimizer.param_groups[0]["lr"], len(train_dl)))
        for i, batch in enumerate(train_dl):
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
            
            logger.log({
                "Loss": loss.item(), 
                "Similarity Loss":sim_loss.item(), 
                "Cross Entropy Loss": ce_loss.item(),
                })
            
            loss /= options.loss_accumulation_step # for loss accumulation
            loss.backward()
            
             # accumulate loss over options.loss_accumulation_step steps
            if i % options.loss_accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            # run test on training data; no best epoch logging
            if i % (len(train_dl) // 3) == len(train_dl) // 3 - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    print("Running test in epoch {} with lr {} (0/{})".format(epoch, optimizer.param_groups[0]["lr"], len(test_dl)))
                    
                    model.eval_mode()
                    encoded_move_set = model.encode_moves(torch.arange(0, len(token_dict)).to(model.device))
                    for batch in test_dl:
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
                    logger.log({"train accuracy": accuracy, "train top_5_acc": top_5_acc})
                    
                    print("Finished test in epoch {} with accuracy {}".format(epoch, accuracy))
                    
                    model.train_mode()
                    evaluator.reset()
                    
        # step learning rate scheduler
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        # run eval dataset
        with torch.no_grad():
            print("Running evaluation in epoch {} with lr {} (0/{})".format(epoch, optimizer.param_groups[0]["lr"], len(eval_dl)))
            
            model.eval_mode()
            encoded_move_set = model.encode_moves(torch.arange(0, len(token_dict)).to(model.device))
            for batch in eval_dl:
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
            logger.log({"accuracy": accuracy, "top_5_acc": top_5_acc})
            
            if best_epoch:
                safe_create_folder(options.experiment_folder)
                model.save_checkpoint(options.experiment_folder, "best")
                
            # epoch cleanup section
            stop_early = evaluator.check_early_stoping(epoch)
            if stop_early and epoch > options.epochs*(1/2):
                print("Stoping experiment early due to stop critereon reached!")
                return