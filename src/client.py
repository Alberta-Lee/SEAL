import torch
from minimizer import ASAM
from loss import FGDLoss
import copy


class Client_GC():
    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size    # number of training samples
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def download_from_server(self, args, server):
        self.gconvNames = server.W.keys()
        if args.alg == 'fedstar':
            for k in server.W:
                if '_s' in k:
                    self.W[k].data = server.W[k].data.clone()
        else:
            for k in server.W:
                self.W[k].data = server.W[k].data.clone()

    def cache_weights(self):
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def reset(self):
        copy(target=self.W, source=self.W_old, keys=self.gconvNames)

    def evaluate(self):
        return eval_gc(self.model, self.dataLoader['test'], self.args.device)
    
    def local_train_seal(self, local_epoch, rho):
        train_stats = train_gc_seal(self.model, self.dataLoader, self.optimizer, local_epoch, self.args, rho)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()


def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def flatten(w):
    vals = [v.flatten() for v in w.values() if v is not None]
    if vals:
        return torch.cat(vals)
    else:
        return torch.tensor([0.], requires_grad=True)
    # return torch.cat([v.flatten() for v in w.values() if v is not None])


def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm


def eval_gc(model, test_loader, device):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs


def train_gc_seal(model, dataloaders, optimizer, local_epoch, args, rho):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    fedloss = FGDLoss()
    for epoch in range(local_epoch):
        minimizer = ASAM(model=model, optimizer=optimizer, rho=rho)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, local_epoch)
        model.train()
        total_loss = 0.
        ngraphs = 0
        acc_sum = 0

        for _, batch in enumerate(train_loader):
            batch.to(args.device)
            optimizer.zero_grad()

            if args.fgd:
                pred, features = model(batch, return_features=True)
            else:
                pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
            if args.fgd:
                loss_fgd = fedloss(features)
                loss = loss + args.fgd_coef * loss_fgd
            loss.backward()
            minimizer.ascent_step()

            if args.fgd:
                pred, features = model(batch, return_features=True)
                loss_fgd = fedloss(features)
            else:
                pred = model(batch)
            loss = model.loss(pred, label)
            if args.fgd:
                loss = loss + args.fgd_coef * loss_fgd
            loss.backward()
            minimizer.descent_step()
            scheduler.step()

            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        loss_v, acc_v = eval_gc(model, val_loader, args.device)
        loss_tt, acc_tt = eval_gc(model, test_loader, args.device)
        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)
    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}