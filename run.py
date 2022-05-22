import argparse
from utils.train import train
import torch
import torch.distributed as dist
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--exp_name", default="DDP_warmup", type=str, help="name of experiment")
    parser.add_argument("-l", "--learning_rate", default=1e-1, type=float, help="learning rate")
    parser.add_argument("-b", "--batch_size", default=1024, type=int, help="batch size")
    parser.add_argument("-e", "--epochs", default=500, type=int, help="epochs")
    parser.add_argument("-w", "--warmup_epochs", default=10, type=int, help="epochs for warmup")
    parser.add_argument("-t", "--warmup_type", default="linear", type=str, help="warmup type")
    parser.add_argument("-s", "--seed_num", default=42, type=int, help="number of random seed")
    parser.add_argument("-d", "--data_path", default="./datasets/", type=str, help="path of dataset")
    parser.add_argument("-p", "--model_path", default="./experiment_model/", type=str, help="path of model")
    parser.add_argument("--local_rank", type=int, help="local rank for DistributedDataParallel")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Multi GPU
    print(f"Running DDP on rank: {args.local_rank}")
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    train(args)

if __name__=="__main__":
    main()