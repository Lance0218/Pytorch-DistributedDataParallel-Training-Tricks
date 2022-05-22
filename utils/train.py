# System / Python
import os
import time
import math
# Custom
from utils.customized import same_seeds, EarlyStopping
from utils.model import Toy_Net
from utils.process import iterate_loader
from utils.lookahead import Lookahead
# ML
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from torch.nn.parallel import DistributedDataParallel as DDP


def train(args):
    # Load MNIST
    train_set = MNIST(root=args.data_path, download=True, train=True, transform=ToTensor())
    train_sampler = DistributedSampler(train_set)
    same_seeds(args.seed_num)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler)
    valid_set = MNIST(root=args.data_path, download=True, train=False, transform=ToTensor())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print(f"Now Training: {args.exp_name}")
    # Load model
    same_seeds(args.seed_num)
    model = Toy_Net()
    model = model.to(args.local_rank)

    # Model parameters
    os.makedirs(os.path.join(args.model_path, "logs"), exist_ok=True)
    latest_model_path = os.path.join(args.model_path, args.exp_name)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    lookahead = Lookahead(optimizer=optimizer, k=10, alpha=0.5)
    loss_function = nn.CrossEntropyLoss()
    if args.local_rank == 0:
        best_valid_acc = 0

    # Callbacks
    if args.warmup_type == "linear":
        warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1
    elif args.warmup_type == "cosine":
        warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 0.5 * (math.cos((epoch - args.warmup_epochs) /(args.epochs - args.warmup_epochs) * math.pi) + 1)
    scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=6, verbose=True)
    early_stopping = EarlyStopping(patience=30, verbose=True)
            
    # Apex
    #amp.register_float_function(torch, "sigmoid")   # register for uncommonly function
    model, apex_optimizer = amp.initialize(model, optimizers=lookahead, opt_level="O1")

    # Build training model
    parallel_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Train model
    if args.local_rank == 0:
        tb = SummaryWriter(os.path.join(args.model_path, "logs", args.exp_name))
    #apex_optimizer.zero_grad()
    #apex_optimizer.step()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)

        # Train
        parallel_model.train()
        # Warm up
        #if epoch < args.warmup_epochs:
        #    scheduler_wu.step()
        train_loss, train_acc, curr_lr = iterate_loader(
            loader=train_loader,
            model=parallel_model,
            loss_function=loss_function,
            local_rank=args.local_rank,
            apex_optimizer=apex_optimizer,
            training=True)

        # Valid
        parallel_model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = iterate_loader(
                loader=valid_loader,
                model=parallel_model,
                loss_function=loss_function,
                local_rank=args.local_rank,
                apex_optimizer=None,
                training=False)
        if args.local_rank == 0:
            tb.add_scalar("LR", curr_lr, epoch)
            tb.add_scalars("Loss", {"train": train_loss, "valid": valid_loss}, epoch)
            tb.add_scalars("Acc", {"train": train_acc, "valid": valid_acc}, epoch)
            
        # Print result
        print(f"epoch: {epoch:03d}/{args.epochs}, time: {time.time()-epoch_start_time:.2f}s, learning_rate: {curr_lr}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}")

        # Learning_rate callbacks
        if epoch <= args.warmup_epochs:
            scheduler_wu.step()
        scheduler_re.step(valid_loss)
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            break

        # Save_checkpoint
        if args.local_rank == 0:
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(parallel_model.module.state_dict(), f"{latest_model_path}.pt")

    if args.local_rank == 0:
        tb.close()