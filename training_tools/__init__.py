from sklearn.metrics import confusion_matrix
import numpy as np
from .embedding_tools import load_or_compute_projections


class AverageMeter(object):
    def __init__(self):
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


class MetricComputer(object):
    def __init__(self, metric_names=None, n_classes=5):
        __all_metrics__ = {"accuracy": self._accuracy, 
                            "class-level-accuracy": self._class_level_accuracy,
                            "confusion_matrix": self._confusion_matrix}
        all_names = list(__all_metrics__.keys())
        if metric_names is None:
            metric_names = all_names
        for n in metric_names: assert n in all_names
        self.metrics = {m: __all_metrics__[m] for m in metric_names}
        self.n_classes = n_classes
    
    def __call__(self, out, target):
        """
        Args:
            out (torch.Tensor): Model output
            target (torch.Tensor): Target labels
        """
        pred = out.argmax(dim=1)
        result = {m: self.metrics[m](out, pred, target) for m in self.metrics.keys()}
        return result
    
    def _accuracy(self, out, pred, target):
        acc = (pred == target).float().detach().mean()
        return acc.item()

    def _class_level_accuracy(self, out, pred, target):
        per_class_acc = {}
        for c in range(self.n_classes):
            count = (target == c).sum().detach().item()
            if count == 0:
                continue
            class_true = ((pred == target) * (target == c)).float().sum().item()
            per_class_acc[c] = (class_true, count)
        return per_class_acc
    
    def _confusion_matrix(self, out, pred, target):
        y_true = target.detach().cpu()
        y_pred = pred.detach().cpu()
        return confusion_matrix(y_true, y_pred, normalize=None, labels=np.arange(self.n_classes))
