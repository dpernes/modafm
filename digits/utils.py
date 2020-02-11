import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

class CombineLoaders:
    def __init__(self, loaders):
        self.loaders = list(loaders)
        self.iters = self.reset_iters()

    def reset_iters(self):
        return [iter(ldr) for ldr in self.loaders]

    def __iter__(self):
        return self

    def __len__(self):
        return max([len(ldr) for ldr in self.loaders])

    def __next__(self):
        batches = []
        for ldr, itr in zip(self.loaders, self.iters):
            try:
                batch = next(itr)
                batches.append(batch)
            except StopIteration:
                if len(ldr) == len(self):
                    self.iters = self.reset_iters()
                    raise StopIteration
                itr = iter(ldr)
                batch = next(itr)
                batches.append(batch)
        return tuple(batches)

class MSDA_Loader:
    def __init__(self, datasets, target, batch_size=8, shuffle=True, device='cpu'):
        # the first dataloader is for the target domain, the remaining are for the source domains
        dataloaders = ([DataLoader(datasets[target], batch_size=batch_size, shuffle=shuffle)]
                       +[DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
                         for (ds_name, ds) in datasets.items() if ds_name != target])

        self.sources = [ds_name for ds_name in datasets if ds_name != target]
        self.dataloader = CombineLoaders(dataloaders)
        self.device = device

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        batch = next(self.dataloader)
        Xt = batch[0][0].float().to(self.device)
        Xs = [batch[i][0].float().to(self.device) for i in range(1, len(self.sources)+1)]
        ys = [batch[i][1].long().to(self.device) for i in range(1, len(self.sources)+1)]
        return Xt, Xs, ys

def fixmatch_loss(logits, aug_logits, min_conf=0.9):
    with torch.no_grad():
        pseudo_labels = torch.argmax(logits, dim=1)
        mask = (torch.max(F.softmax(logits, dim=1), dim=1).values > min_conf)
    loss_per_example = F.cross_entropy(aug_logits, pseudo_labels, reduction='none')
    loss = torch.mean(mask*loss_per_example, dim=0)
    return loss

def save_checkpoint(epoch, model, optimizer, lr_sched, name='checkpoint.pth'):
    if isinstance(model, list):  # in case we are saving multiple models
        model_sd = [mdl.state_dict() for mdl in model]
    else:
        model_sd = model.state_dict()
    if isinstance(optimizer, list):  # in case we are saving multiple optimizers
        optimizer_sd = [opt.state_dict() for opt in optimizer]
    else:
        optimizer_sd = optimizer.state_dict()
    checkpoint = {
        'epoch': epoch,
        'model': model_sd,
        'optimizer': optimizer_sd,
        'lr_sched': lr_sched}
    torch.save(checkpoint, name)
