import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torchvision.models import resnet18

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    cudnn.benchmark = True
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
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

def validate(val_loader, model, criterion, rank, world_size):
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
            images = images.to(rank, non_blocking=True)
            target = target.to(rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, world_size)
            reduced_acc1 = reduce_mean(acc1, world_size)
            reduced_acc5 = reduce_mean(acc5, world_size)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if rank == 0:
                progress.display(i)
        if rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg
def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    seed = 1; batch_size = 100; epochs = 200; lr = 0.1
    # create model and move it to GPU with id rank
    model = resnet18()
    torch.cuda.set_device(rank)
    model.to(rank)

    ##
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
    model = DDP(model, device_ids=[rank])
    ##

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    ##
    batch_size = int(batch_size / world_size)
    train_dataset = get_train_dataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=train_sampler)
    test_dataset = get_test_dataset()
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=test_sampler)
    ##
    for epoch in range(epochs):
        start = time.time()
        model.train()
        
        ##
        train_sampler.set_epoch(epoch)
        ##

        for step, (images, labels) in enumerate(train_loader):
            ##
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)
            ##
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            ##
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, world_size)
            ##
            
            loss.backward()
            optimizer.step()
            
            ##
            if rank == 0 and (step * batch_size + len(images)) % 100 == 0:
                print(
                    'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                        reduced_loss,
                        optimizer.param_groups[0]['lr'],
                        epoch=epoch+1,
                        trained_samples=step * batch_size + len(images),
                        total_samples=len(train_loader.dataset)
                    ))
            ##
        
        finish = time.time()

        ##
        if rank == 0:
            print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
        ##

        validate(test_loader, model, criterion, rank, world_size)

        train_scheduler.step(epoch)

    cleanup()


def run(fn, world_size):
    mp.spawn(
        fn,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    run(demo_basic, torch.cuda.device_count())
    # run_demo(demo_checkpoint, 2)
    # run_demo(demo_model_parallel, 2)