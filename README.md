# Pytorch-DistributedDataParallel-Training-Tricks

This is an example that integrates **Pytorch DistributedDataParallel, Apex, warmup, learning rate scheduler**, if you need to read this article in Chinese, please check my [Medium](https://medium.com/@Lance0218/training-tricks-for-pytorch-distributed-data-parallel-1cd48cc7d97a).
In the following chapters, I will introduce how to use DistributedDataParallel (DDP). And how to assimilate three training techniques of Apex, warmup, and learning rate scheduler into DDP training in order. I will also mention the set-up of early-stopping and Random seed.

## DistributedDataParallel (DDP)

1. Pytorch official website also recommends using DistributedDataParallel (multi-process control multi-GPU) instead of DataParallel (single-process control multi-GPU) when using    multi-GPU training, which improves the speed and solves the problem of uneven GPU loading.
2. The basic usage is to load the model to be used and wrap it with DDP (L16), local_rank is the rank of the current GPU generated by calling `torch.distributed.launch`.
3. Data transmission under multiple nodes will seriously affect efficiency, so DistributedSampler is used to ensure that DataLoader will only load a specific subset of the data    set (L4). The batch_size under DistributedSampler is the actual batch size used by a single GPU.
4. Call set_epoch(epoch) at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs (L21).

```
# Load MNIST
data_root = './'
train_set = MNIST(root=data_root, download=True, train=True, transform=ToTensor())
train_sampler = DistributedSampler(train_set)   # (L4)
same_seeds(args.seed_num)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler)
valid_set = MNIST(root=data_root, download=True, train=False, transform=ToTensor())
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

# Load model
same_seeds(args.seed_num)
model = Toy_Net()
model = model.to(args.local_rank)

# Build training model
parallel_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)   # (L16)

# Train model
for epoch in range(args.epochs):
    #...
    train_sampler.set_epoch(epoch)   # (L21)
    for image, target in tqdm(train_loader, total=len(train_loader)):
    #...
```

## Apex

1. Apex reduces memory usage by sacrificing some precision, allowing us to use larger network architecture and larger batch size.
2. Initialize your constructed model and optimizer through `amp.initialize` (L12). If you use non-Pytorch's built-in optimizer (such as lookahead) like me, you should pay          attention to the initialized object.
3. And it is used to clear the past gradients (L19), backpropagate the current gradients (L23), and update the network parameters according to the current gradients (L24).
4. To use functions such as sigmoid and softmax (for details, please refer to the [GitHub](https://github.com/NVIDIA/apex/tree/master/apex/amp) of `apex.amp`), you need to          register before initialization (L11).

```
# Load model
same_seeds(args.seed_num)
model = Toy_Net()
model = model.to(args.local_rank)

# Model parameters
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
lookahead = Lookahead(optimizer=optimizer, k=10, alpha=0.5)

# Apex
#amp.register_float_function(torch, 'sigmoid')   # register for uncommonly function   # (L11)
model, apex_optimizer = amp.initialize(model, optimizers=lookahead, opt_level="O1")   # (L12)

# Train model
for epoch in range(args.epochs):
    #...
    # Train
    for image, target in tqdm(train_loader, total=len(train_loader)):
        apex_optimizer.zero_grad()   # (L19)
        #...
        # Apex
        with amp.scale_loss(batch_loss, apex_optimizer) as scaled_loss:
            scaled_loss.backward()   # (L23)
        apex_optimizer.step()   # (L24)
```

## Warmup & learning rate scheduler & early-stopping

1. Warmup is a training trick. By warming up the learning rate from small to large, the instability caused by the initial learning rate can be avoided, and the direction of        stable optimization can be found faster. It has been proven effective in some tasks. The article is realized through lr_scheduler (L6, L7, L28, L29).
2. Learning rate scheduler is also a technique for training models. This article uses `lr_scheduler.ReduceLROnPlateau`, which I prefer to use, as an example (L8, L30). Note that    the optimizer in `lr_scheduler` should point to the built-in Pytorch instead of the additional one (L7, L8).
3. This article also includes early-stopping to reduce overfitting (L9, L31~L33), and there are also examples of using consine learning rate as attenuation. If you are              interested, please check [here](https://medium.com/r?url=https%3A%2F%2Fgithub.com%2FLance0218%2FPytorch-DistributedDataParallel-Training-Tricks%2Fblob%2Fmaster%2FDDP_warmupcos.py).
4. The warmup learning rate in this article starts from 0. If you want to have a value from the beginning, you can add (L12, L13) and replace (L28, L29) with (L17, L18).

```
import torch
from customized_function import same_seeds, EarlyStopping

# Model parameters
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
lookahead = Lookahead(optimizer=optimizer, k=10, alpha=0.5)

# Callbacks
warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1   # (L6)
scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)   # (L7)
scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True)   # (L8)
early_stopping = EarlyStopping(patience=50, verbose=True)   # (L9)

# Train model
#apex_optimizer.zero_grad()   # (L12)
#apex_optimizer.step()   # (L13)
for epoch in range(args.epochs):
    #...
    # Train
    #if epoch < args.warmup_epochs:   # (L17)
    #    scheduler_wu.step()   # (L18)
    for image, target in tqdm(train_loader, total=len(train_loader)):
        #...
    #...
    # Valid
    with torch.no_grad():
        for image, target in tqdm(valid_loader, total=len(valid_loader)):
            #...
    #...
    # Learning_rate callbacks
    if epoch <= args.warmup_epochs:   # (L28)
        scheduler_wu.step()   # (L29)
    scheduler_re.step(valid_acc)   # (L30)
    early_stopping(valid_acc)   # (L31)
    if early_stopping.early_stop:   # (L32)
        break   # (L33)
```

## Random seed

1. If you want to train a model that can be implemented repeatedly, you must set a random seed.
2. There are two places that need to be set: before Training DataLoader (L5) to fix the shuffle result, and before model (L11) to fix the initial weight.
3. In this article, I directly use the function same_seeds to adjust all random seeds at once. You can also try to adjust only specific random seeds.

```
# Load MNIST
data_root = './'
train_set = MNIST(root=data_root, download=True, train=True, transform=ToTensor())
train_sampler = DistributedSampler(train_set)
same_seeds(args.seed_num)   # (L5)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler)
valid_set = MNIST(root=data_root, download=True, train=False, transform=ToTensor())
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

# Load model
same_seeds(args.seed_num)   # (L11)
model = Toy_Net()
model = model.to(args.local_rank)
```

## Run

1. Pytorch officially provides two running methods: `torch.distributed.launch` and `torch.multiprocessing.spawn`, but when I use the latter, sometimes the GPU will not be          automatically released after training, so this article uses `torch.distributed.launch` to do it Demo.
2. This article mainly demonstrates the single-node multi-GPU operation mode: CUDA_VISIBLE_DEVICES specifies the GPU used, nproc_per_node is the number of GPUs used by the node,    if you need to perform multiple trainings, stagger the master_port, [here](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py) is a more detailed        description.

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=6666 DDP_warmup.py -en=DDP_warmup
```

The operating environment of this article is CUDA10.2, cuDNN7.6.5, Python3.7, Pytorch1.3.1

## Function implementation

1. Lookahead implementation: [lookahead.pytorch](https://github.com/alphadl/lookahead.pytorch)
2. Early-stopping, random seed implementation: [customized_function.py](https://github.com/Lance0218/Pytorch-DistributedDataParallel-Training-Tricks/blob/master/customized_function.py)
