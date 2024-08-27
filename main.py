import argparse
import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model

DEFAULT_TIMEOUT = timedelta(seconds=10)

# Function to train for one epoch
def train(net, data_loader, train_optimizer, temperature):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        out = torch.cat([out_1.clone(), out_2.clone()], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)

        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.size(0), device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.size(0), -1)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos_sim = torch.cat([pos_sim.clone(), pos_sim.clone()], dim=0)

        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += pos_1.size(0)
        total_loss += loss.item() * pos_1.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

# Function to test for one epoch
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            sim_matrix = torch.mm(feature, feature_bank)
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

# Distributed training function
def distributed_train(local_rank, args):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend, init_method=args.dist_url, world_size=args.world_size, rank=local_rank)

    train_data = utils.CustomDatasetPair(data=utils.train_images, targets=utils.train_labels, transform=utils.train_transform)
    memory_data = utils.CustomDatasetPair(data=utils.mem_images, targets=utils.mem_labels, transform=utils.test_transform)
    test_data = utils.CustomDatasetPair(data=utils.test_images, targets=utils.test_labels, transform=utils.test_transform)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = Model(args.feature_dim).cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer,args.temperature)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            if local_rank == 0:  # Save only on rank 0
                torch.save(model.state_dict(), 'results/model.pth')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", help="'gloo' or 'nccl'.")
    parser.add_argument("--num-machines", type=int, default=1, help="# of machines.")
    parser.add_argument("--num-gpus", type=int, default=1, help="# GPUs per machine.")
    parser.add_argument("--machine-rank", type=int, default=0, help="Rank of this machine.")
    parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:23456", help="Init URL for distributed training.")
    parser.add_argument("--feature_dim", default=128, type=int, help="Feature dim for latent vector")
    parser.add_argument("--temperature", default=0.5, type=float, help="Temperature for softmax")
    parser.add_argument("--k", default=200, type=int, help="Top k similar images to predict the label")
    parser.add_argument("--batch_size", default=512, type=int, help="Mini-batch size")
    parser.add_argument("--epochs", default=500, type=int, help="Number of training epochs")
    
    args = parser.parse_args()
    args.world_size = args.num_gpus * args.num_machines
    
    mp.spawn(distributed_train, nprocs=args.num_gpus, args=(args,))

if __name__ == '__main__':
    main()
