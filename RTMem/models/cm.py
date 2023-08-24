import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs2, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            # ctx.features[y] = 0.1 * ctx.features[y] + (1. - 0.1) * x 
            # ctx.features[y] = x                       
            ctx.features[y] /= ctx.features[y].norm()
        return grad_inputs, None, None, None, None


def cm(inputs, inputs2, indexes, features, momentum=0.5):
    return CM.apply(inputs, inputs2, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)


        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))  # Hard
            # median = np.argmax(np.array(distances))  # Easy
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            # ctx.features[y] = x                       
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = 0.0
        # self.momentum_hard = 0.5
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, momentum):
        inputs = F.normalize(inputs, dim=1).cuda()


        outputs2 = cm(inputs, inputs, targets, self.features, self.momentum)


        outputs2 /= self.temp
        loss = F.cross_entropy(outputs2, targets)
        
        # outputs_hard /= self.temp
        # loss_hard = F.cross_entropy(outputs_hard, targets)

        # loss_triplet = TripletLoss(margin=0.2)(inputs, targets)

        loss1 = loss #+ loss_triplet

        
        return loss1

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1-cosine

def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p = sorted_mat_distance[:, :4]#.mean(1)
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, :4]#.mean(1)
    hard_n_indice = negative_indices[:, 0]
    if(indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n

class TripletLoss(nn.Module):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''

    def __init__(self, margin, normalize_feature=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 1
        self.scale_neg = 1 # 40

    def forward(self, emb, label):
        if self.normalize_feature:
            # equal to cosine similarity
            emb = F.normalize(emb)
        mat_dist = euclidean_dist(emb, emb)

        # mat_dist = cosine_dist(emb, emb)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

        dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
        assert dist_an.size(0)==dist_ap.size(0)
        y = torch.ones_like(dist_ap[:,0])
        # loss = self.margin_loss(dist_an[:,0], dist_ap[:,0], y) # + self.margin_loss(dist_an[:,1], dist_ap[:,1], y) + self.margin_loss(dist_an[:,2], dist_ap[:,2], y) +\
                # self.margin_loss(dist_an[:,3], dist_ap[:,3], y)
        # prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)


        pos_loss = 1.0 / self.scale_pos * torch.log(
            1 + torch.sum(torch.exp(-self.scale_pos * (dist_an[:,0] - self.thresh))))
        neg_loss = 1.0 / self.scale_neg * torch.log(
            1 + torch.sum(torch.exp(self.scale_neg * (dist_ap[:,0] - self.thresh))))

        # loss = -torch.sum(torch.log(torch.exp(self.scale_pos * (dist_an[:,0] - self.thresh))/(
        #     torch.exp(self.scale_pos * (dist_an[:,0] - self.thresh))
        #     +torch.exp(self.scale_neg * (dist_ap[:,0])))))


        loss = pos_loss + neg_loss

        return loss

class MultiSimilarityLoss(nn.Module):
    def __init__(self, thresh=0.5, margin=0.1, scale_pos=2, scale_neg=40):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2
        self.scale_neg = 40

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)

        # sim_mat = torch.matmul(feats, torch.t(feats))

        sim_mat = cosine_dist(feats, feats)

        epsilon = 1e-5
        loss = list()


        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            if len(pos_pair_) == 0:
                continue

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], device='cuda:0', requires_grad=True)


        loss = sum(loss) / batch_size
        return loss



class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x  
            # ctx.features[y] = 0.1 * ctx.features[y] + (1. - 0.1) * x                       
                     
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = 0.0
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes, momentum):
        # inputs: B*2048, features: L*2048

        inputs = hm(inputs, indexes, self.features, self.momentum)
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            # exps = torch.exp(vec)
            # masked_exps = exps * mask.float().clone()
            masked_exps = vec * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        ####################
        # loss_triplet = TripletLoss(margin=0.2)(inputs, targets)
        #####################

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        ################
        inputs = torch.exp(inputs)
        ################

        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        #
        # sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-6), targets) #+ loss_triplet





