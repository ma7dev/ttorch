import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import transforms, datasets
import torch.nn as nn
import time
import torch
import torch.distributed as dist
from torchvision.models import resnet18

import warnings
warnings.filterwarnings("ignore")
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if local_rank == 0:
                progress.display(i)
        if local_rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg

def get_train_dataset():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    trainset = datasets.CIFAR100(root='./data',
                                train=True,
                                download=True,
                                transform=transform_train)
    return trainset


def get_test_dataset():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    testset = datasets.CIFAR100(root='./data',
                                train=False,
                                download=True,
                                transform=transform_test)
    return testset

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
    parser.add_argument('--batch_size','--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--ip', default='localhost', type=str)
    parser.add_argument('--port', default='23456', type=str)

    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    worker(args.local_rank, args.nprocs, args)

def worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    init_method = 'tcp://' + args.ip + ':' + args.port
    cudnn.benchmark = True
    dist.init_process_group(
        backend='gloo', 
        init_method=init_method, 
        world_size=args.nprocs,
        rank=local_rank
    )
    model = resnet18()
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    ##
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    ##

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    ##
    batch_size = int(args.batch_size / nprocs)
    train_dataset = get_train_dataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=train_sampler)
    test_dataset = get_test_dataset()
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=test_sampler)
    ##

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        
        ##
        train_sampler.set_epoch(epoch)
        ##

        train_scheduler.step(epoch)

        for step, (images, labels) in enumerate(train_loader):
            ##
            images = images.cuda(local_rank, non_blocking=True)
            labels = labels.cuda(local_rank, non_blocking=True)
            ##

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            ##
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)
            ##
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ##
            if args.local_rank == 0 and (step * args.batch_size + len(images)) % 100 == 0:
                print(
                    'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                        reduced_loss,
                        optimizer.param_groups[0]['lr'],
                        epoch=epoch+1,
                        trained_samples=step * args.batch_size + len(images),
                        total_samples=len(train_loader.dataset)
                    ))
            ##
        
        finish = time.time()

        ##
        if args.local_rank == 0:
            print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
        ##

        validate(test_loader, model, criterion, local_rank, args)


if __name__ == '__main__':
    main()