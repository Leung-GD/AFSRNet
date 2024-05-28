'''this Hard_loss is intercepted from HardNet
"Working hard to know your neighbor's margins: Local descriptor learning loss" '''

import torch
import torch.nn as nn
import sys
import numpy as np

def distance_matrix_vector(f1, f2):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    MM = torch.mm(f1, torch.t(f2))
    MM = (1 - MM) * 2


    eps = 1e-6
    return torch.sqrt(MM+eps)

def hynet_dist_SL(f1, f2):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    MM = torch.mm(f1, torch.t(f2))
    MM = (1 - MM) * 2


    eps = 1e-6
    return torch.sqrt(MM+eps)

def hynet_dist_SI(f1, f2):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    MM = torch.mm(f1, torch.t(f2))
    eps = 1e-6
    return MM+eps



#def distance_matrix_vector(anchor, positive):
#    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

#    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
#    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

#    eps = 1e-6
#    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
#                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

def distance_vectors_pairwise(anchor, positive, negative = None):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)

    eps = 1e-8
    d_a_p = torch.sqrt(a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1) + eps)
    if negative is not None:
        n_sq = torch.sum(negative * negative, dim=1)
        d_a_n = torch.sqrt(a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1) + eps)
        d_p_n = torch.sqrt(p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1) + eps)
        return d_a_p, d_a_n, d_p_n
    return d_a_p
def loss_random_sampling(anchor, positive, negative, anchor_swap = False, margin = 1.0, loss_type = "triplet_margin"):
    """Loss with random sampling (no hard in batch).
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    (pos, d_a_n, d_p_n) = distance_vectors_pairwise(anchor, positive, negative)
    if anchor_swap:
       min_neg = torch.min(d_a_n, d_p_n)
    else:
       min_neg = d_a_n

    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_L2Net(eye,anchor, positive):
    """L2Net losses: using whole batch as negatives, not only hardest.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive)
    #eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()
    l = int(dist_matrix.size(0))
    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye[0:l,0:l]*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    

    exp_pos = torch.exp(2.0 - pos1);
    exp_den = torch.sum(torch.exp(2.0 - dist_matrix),1) + eps;
    loss = -torch.log( exp_pos / exp_den )
    loss = torch.mean(loss)
    return loss

def loss_HardNet(eye,anchor, positive):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps
    l = int(dist_matrix.size(0))
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye[0:l,0:l]*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    min_neg = torch.min(dist_without_min_on_diag,1)[0]
    min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
    min_neg = torch.min(min_neg,min_neg2)
    min_neg = min_neg
    pos = pos1
    loss = torch.clamp(1 + pos - min_neg, min=0.0)

    loss = torch.mean(loss)
    return loss

def loss_HardNet_facescape(eye,anchor, positive):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps
    l = int(dist_matrix.size(0))
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye[0:l,0:l]*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    max_neg = torch.max(dist_without_min_on_diag,1)[0]
    max_neg2 = torch.max(dist_without_min_on_diag,0)[0]
    max_neg = torch.max(max_neg,max_neg2)
    max_neg = max_neg
    pos = pos1
    loss = torch.clamp(1 + pos - max_neg, min=0.0)

    loss = torch.mean(loss)
    return loss

def compute_RSOS(distance_matrix,K):
    RSOS=0
    for i in range(0,K):
        for j in range(0,K):
            if i==j:
                RSOStemp=0
            else:
                RSOStemp=torch.square(distance_matrix[i][j]-distance_matrix[j][i])
        RSOS=RSOS+RSOStemp
    RSOS=torch.sqrt(RSOS)
    RSOS=(RSOS/K)
    return RSOS 

def compute_ASOSR(distance_matrix):
    ASOSRtemp=torch.abs(distance_matrix-distance_matrix.T)
    ASOSR=torch.mean(ASOSRtemp)
    #print(ASOSR.shape)
    return ASOSR

def loss_SOSNet(eye,anchor, positive):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = hynet_dist_SL(anchor, positive) +eps
    l = int(dist_matrix.size(0))
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye[0:l,0:l]*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    
    min_neg = torch.min(dist_without_min_on_diag,1)[0]
    min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
    min_neg = torch.min(min_neg,min_neg2)
    min_neg = min_neg
    pos = pos1
    lFOS = torch.clamp(1 + pos - min_neg, min=0.0)
    lFOS = torch.square(lFOS)
    RSOS = compute_RSOS(dist_without_min_on_diag,50)
    loss = lFOS+RSOS
    loss = torch.mean(loss)
    return loss
def compute_ASOSR_matr(distance_matrix):
    ASOSRtemp=distance_matrix-distance_matrix.T
    fenzi=torch.norm(ASOSRtemp)
    fenmu=torch.norm(distance_matrix)+1e-5
    ASOSR=fenzi/fenmu
    #print(ASOSR.shape)
    return ASOSR

def loss_hynet(eye,anchor, positive):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    SI=hynet_dist_SI(anchor, positive)+eps
    #SL=hynet_dist_SL(anchor, positive)+eps
    #SH=(2*(1-SI)+SL)
    
    #dist_matrix = SH
    dist_matrix = 2*(1.0-SI)
    l = int(dist_matrix.size(0))
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye[0:l,0:l]*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    min_neg = torch.min(dist_without_min_on_diag,1)[0]
    min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
    min_neg = torch.min(min_neg,min_neg2)
    min_neg = min_neg
    pos = pos1
    #L2_Regularisation = torch.mean(torch.diag(SL))
    #RAL_Regularisation=torch.mean(torch.diag(dist_matrix))
    ASOSR=compute_ASOSR_matr(dist_matrix)
    #loss = torch.clamp(1 + pos - min_neg, min=0.0)+0.1*RAL_Regularisation
    loss = 1+torch.tanh(pos-min_neg)
    #loss=torch.log(1+torch.exp((min_neg-0)*min_neg)*torch.exp((pos-1)*pos))
    loss = torch.mean(loss)+0.1*ASOSR
    return loss


