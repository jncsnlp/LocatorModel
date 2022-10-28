import torch
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

'''
Loss with CrossEntropy or Distributional
Parameter:
    outputs: logits output from model
    targets: simple LongInteger indicating correct class, or logits output
    is_adv: if adversarial training then maximize the loss
    is_dist: if distributional loss then set to True
'''


class AdvLoss:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        assert self.num_classes > 1

    # targets could be distribution
    def __call__(self, outputs, targets, is_adv=False, is_dist=False):
        # Transform label to one-hot
        # bsize = outputs.size(0)
        # print(bsize)
        # if is_dist:
        # else:
        #     tmp = torch.zeros(bsize, self.num_classes).scatter_(1, targets.view(-1, 1).long(), 1)
        #     targets = tmp.to(outputs.device)

        if is_dist:
            assert targets.size(1) == self.num_classes
            probs = torch.softmax(outputs, dim=1)  # turn model prediction to probs
            # print(probs)
            targets = torch.softmax(targets, dim=1)  # turn targets logits to probs
            # print(targets)
            # print(((probs - targets) ** 2)[:,1])
            loss = torch.sum(((probs - targets) ** 2), 1)
        else:

            weight = np.array([1, 3])
            weight = torch.from_numpy(weight).float().to(device)
            loss = F.cross_entropy(outputs, targets, weight=weight)


        # adversarial attack
        if is_adv:
            loss = -loss

        return loss


class AdvTargetLoss:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        assert self.num_classes > 1

    # targets could be distribution
    def __call__(self, outputs, targets, target_class, is_adv=False, is_dist=False):
        # Transform label to one-hot
        bsize = outputs.size(0)
        if is_dist:
            assert targets.size(1) == self.num_classes
        else:
            targets = torch.zeros(bsize, self.num_classes).to(outputs.device).scatter_(1, targets.view(-1, 1).long(), 1)
            targets = targets.to(outputs.device)

        assert 0 <= target_class < self.num_classes

        if is_dist:
            probs = torch.softmax(outputs, dim=1)  # turn model prediction to probs
            targets = torch.softmax(targets, dim=1)  # turn targets logits to probs
            diff = (probs - targets) ** 2
            loss = torch.mean(diff[:, target_class])
        else:
            diff = F.log_softmax(outputs, dim=1) * targets
            loss = -torch.mean(diff[:, target_class])

        # adversarial attack
        if is_adv:
            loss = -loss

        return loss