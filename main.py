import multiprocessing

# Ensure the 'spawn' start method is set before anything else
multiprocessing.set_start_method('spawn', force=True)
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile, clever_format
import utils
from model import Model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '172.19.2.2'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['NCCL_IB_DISABLE'] = '1'  # Disable Infiniband (IB), which can sometimes cause issues
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P communication if it causes issues
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Specify the network interface

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()
def train(rank, world_size, net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.size(0), device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.size(0), -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += pos_1.size(0)
        total_loss += loss.item() * pos_1.size(0)
        train_bar.set_description(f'Train Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f}')

    return total_loss / total_num

def test(rank, world_size, net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description(f'Test Epoch: [{epoch}/{epochs}] Acc@1:{total_top1 / total_num * 100:.2f}% Acc@5:{total_top5 / total_num * 100:.2f}%')

    return total_top1 / total_num * 100, total_top5 / total_num * 100

def main(rank, world_size):
    setup(rank, world_size)
    # Define device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        parser = argparse.ArgumentParser(description='Train SimCLR')
        parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
        parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
        parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
        parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
        parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')

        args = parser.parse_args()
        feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
        batch_size, epochs = args.batch_size, args.epochs

        # data prepare
        train_data = utils.CustomDatasetPair(data=utils.train_images, targets=utils.train_labels, transform=utils.train_transform)
        memory_data = utils.CustomDatasetPair(data=utils.mem_images, targets=utils.mem_labels, transform=utils.test_transform)
        test_data = utils.CustomDatasetPair(data=utils.test_images, targets=utils.test_labels, transform=utils.test_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,sampler=DistributedSampler(train_data))
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,sampler=DistributedSampler(train_data))
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False,sampler=DistributedSampler(train_data))

        # model setup and optimizer config
        model = Model(feature_dim).to(rank)
       
        
        print(f"Rank {rank}: before DDP model creation")

        # Use DDP only if using GPU
        if torch.cuda.is_available():
            print(f"Rank {rank}: setting up DDP model on GPU")
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        else:
            print(f"Rank {rank}: setting up DDP model on CPU")
            model = torch.nn.parallel.DistributedDataParallel(model)
        
        print(f"Rank {rank}: DDP model created successfully")
        flops, params = profile(model.module, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
        flops, params = clever_format([flops, params])
        if rank == 0:
            print('# Model Params: {} FLOPs: {}'.format(params, flops))
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        c = len(memory_data.classes)

        # training loop
        results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
        save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
        if rank == 0 and not os.path.exists('results'):
            os.mkdir('results')
        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            train_loss = train(rank, world_size, model, train_loader, optimizer)
            results['train_loss'].append(train_loss)
            test_acc_1, test_acc_5 = test(rank, world_size, model, memory_loader, test_loader)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)
            if rank == 0:
                # save statistics
                data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
                data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
                if test_acc_1 > best_acc:
                    best_acc = test_acc_1
                    torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        
        cleanup()
if __name__ == '__main__':
    world_size = 2  # ou le nombre de GPUs disponibles
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
