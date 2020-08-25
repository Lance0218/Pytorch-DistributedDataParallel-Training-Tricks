# System / Python
import os
import argparse
import time
from tqdm import tqdm
import math
# Custom
from customized_function import same_seeds, EarlyStopping
from lookahead import Lookahead
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Toy_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512,10)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.dense(out)
        return out
        

def model_trainer(args):
    # Load MNIST
    data_root = './'
    train_set = MNIST(root=data_root, download=True, train=True, transform=ToTensor())
    train_sampler = DistributedSampler(train_set)
    same_seeds(args.seed_num)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler)
    valid_set = MNIST(root=data_root, download=True, train=False, transform=ToTensor())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print(f'Now Training: {args.exp_name}')
    # Load model
    same_seeds(args.seed_num)
    model = Toy_Net()
    model = model.to(args.local_rank)

    # Model parameters
    os.makedirs(f'./experiment_model/', exist_ok=True)
    latest_model_path = f'./experiment_model/{args.exp_name}'
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    lookahead = Lookahead(optimizer=optimizer, k=10, alpha=0.5)
    loss_function = nn.CrossEntropyLoss()
    if args.local_rank == 0:
        best_valid_acc = 0
                    
    # Callbacks
    warm_up_cos = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 0.5 * (math.cos((epoch - args.warmup_epochs) /(args.epochs - args.warmup_epochs) * math.pi) + 1)
    scheduler_wucos = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up_cos)
    early_stopping = EarlyStopping(patience=50, verbose=True) 

    # Apex
    #amp.register_float_function(torch, 'sigmoid')   # register for uncommonly function
    model, apex_optimizer = amp.initialize(model, optimizers=lookahead, opt_level="O1")

    # Build training model
    parallel_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Train model
    if args.local_rank == 0:
        tb = SummaryWriter(f'./tensorboard_runs/{args.exp_name}')
    #apex_optimizer.zero_grad()
    #apex_optimizer.step()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = 0., 0.
        valid_loss, valid_acc = 0., 0.
        train_num, valid_num = 0, 0
        train_sampler.set_epoch(epoch)

        # Train
        parallel_model.train()
        # Warm up
        #if epoch < args.warmup_epochs:
        #    scheduler_wu.step()
        for image, target in tqdm(train_loader, total=len(train_loader)):
            apex_optimizer.zero_grad()
            image = image.to(args.local_rank)
            target = target.to(args.local_rank, dtype=torch.long)
            outputs = parallel_model(image)
            predict = torch.argmax(outputs, dim=1)
            batch_loss = loss_function(outputs, target)
            batch_loss /= len(outputs)
            # Apex
            with amp.scale_loss(batch_loss, apex_optimizer) as scaled_loss:
                scaled_loss.backward()
            apex_optimizer.step()

            # Calculate loss & acc
            train_loss += batch_loss.item() * len(image)
            train_acc += (predict == target).sum().item()
            train_num += len(image)

        train_loss = train_loss / train_num
        train_acc = train_acc / train_num
        curr_lr = apex_optimizer.param_groups[0]['lr']
        if args.local_rank == 0:
            tb.add_scalar('LR', curr_lr, epoch)
            tb.add_scalar('Loss/train', train_loss, epoch)
            tb.add_scalar('Acc/train', train_acc, epoch)

        # Valid
        parallel_model.eval()
        with torch.no_grad():
            for image, target in tqdm(valid_loader, total=len(valid_loader)):
                image = image.to(args.local_rank)
                target = target.to(args.local_rank, dtype=torch.long)
                outputs = parallel_model(image)
                predict = torch.argmax(outputs, dim=1)
                batch_loss = loss_function(outputs, target)
                batch_loss /= len(outputs)
                    
                # Calculate loss & acc
                valid_loss += batch_loss.item() * len(image)
                valid_acc += (predict == target).sum().item()
                valid_num += len(image)

        valid_loss = valid_loss / valid_num
        valid_acc = valid_acc / valid_num
        if args.local_rank == 0:
            tb.add_scalar('Loss/valid', valid_loss, epoch)
            tb.add_scalar('Acc/valid', valid_acc, epoch)
            
        # Print result
        print(f'epoch: {epoch:03d}/{args.epochs}, time: {time.time()-epoch_start_time:.2f}s, learning_rate: {curr_lr}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}')

        # Learning_rate callbacks
        scheduler_wucos.step()
        early_stopping(valid_acc)
        if early_stopping.early_stop:
            break

        # Save_checkpoint
        if args.local_rank == 0:
            if valid_acc > best_valid_acc:
                best_valid_loss = valid_acc
                torch.save(parallel_model.module.state_dict(), f'{latest_model_path}.pt')

    if args.local_rank == 0:
        tb.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--exp_name', default='try', type=str, help='name of experiment')
    parser.add_argument('-lr', '--learning_rate', default=1e-1, type=float, help='learning rate')
    parser.add_argument('-bs', '--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('-es', '--epochs', default=500, type=int, help='epochs')
    parser.add_argument('-we', '--warmup_epochs', default=10, type=int, help='epochs for warmup')
    parser.add_argument('-sn', '--seed_num', default=42, type=int, help='number of random seed')
    parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    
    # Multi GPU
    print(f'Running DDP on rank: {args.local_rank}')
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    model_trainer(args)


if __name__=="__main__":
    main()