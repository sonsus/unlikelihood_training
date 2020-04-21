#reference: https://github.com/sonsus/NLUguidedNLG    losses.py
import torch
from torch.nn import CrossEntropyLoss, TripletMarginLoss
import torch.nn.functional as F


#from config import * #SPECIAL TOKENS


'''
class CELoss(CrossEntropyLoss):
    def __init__(self, ignore_index=-100, smooth=0):
        super().__init__(ignore_index=ignore_index)
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, trg):
        if pred.nelement() == 0 or trg.nelement()==0:
            return None
        pred = pred.contiguous().view(-1, pred.shape[-1] )
        trg = trg.contiguous().view(-1)

        if self.smooth>0:
            n_class = pred.shape[1]
            onehot = torch.zeros_like(pred).scatter(1,trg.view(-1,1), 1)
            onehot = onehot *( 1- self.smooth) + (1-onehot) *self.smooth / (n_class-1)  # -1 for original target
            logprob = F.log_softmax(pred, dim=1)

            donotpad = (trg!=self.ignore_index)
            loss = -(onehot * logprob).sum(dim=1)
            loss = loss.masked_select(donotpad).mean()
        else:
            loss = super().forward(pred, trg)

        return loss
'''

class TPLoss_w_Cosdist(TripletMarginLoss):
    def __init__(self, neg='randsample', margin=1.0, p=2.0, eps=1e-6, reduction='mean'):
        super().__init__(margin=margin, p=p, eps=eps, reduction=reduction)
        self.neg = neg # use all or sample
        #self.dist = dist
        self.eps = eps
        self.reduction= reduction
        #becomes different meaning when dist == 'cos'


    def forward(self, pred_cls_feat, tru_cls_feat, fals_cls_feat):
        #sent_feature: bsz, hsz
        bsz, n_choices, hsz = list(fals_cls_feat.shape)

        if n_choices == 1:
            fals_cls_feat = fals_cls_feat.squeeze(1)

        #pos_f = pn_features[:, 0] #bsz, hsz
        #neg_f = pn_features[:, 1:] #bsz, num_choice-1, hsz (if num_choice=2: bsz, hsz)
        assert fals_cls_feat.dim()==3, f"fals_cls_feat.dim() == {fals_cls_feat.dim()}: check run_gpt2.py: ~201": #multiple negs
            if self.neg == 'meanpool':
                pooled_neg = fals_cls_feat.mean(dim=1)
            elif self.neg == 'randsample':
                if n_choices == 1:
                    break
                else:
                    b_idxs = torch.randint(n_choices-1, (bsz,)).to(torch.device('cuda'))
                    pooled_neg = fals_cls_feat.index_select(dim=1, index=b_idxs)
            else:
                assert False, f"check args.negex which is currently {args.negex} choices: ['randsample', 'meanpool'] "
            assert pooled_neg.shape == tru_cls_feat.shape


        '''
        cosine distance of a, b is defined as
        cossim = dot(a,b) / norm(a)norm(b) # [-1, 1]
        cosdist = 1- (cossim+1)/2 = (1 - cossim) / 2
        tploss = margin + positivedist - negativedist

            1. for cosdist margin == maxdist == 1
            2. tploss is 0(min) when positivedist == 0, and negativedist == 1
            3. tploss is 2(max) when positivedist == 1, and negativedist == 0
        '''
        p_cosdist= (1. - F.cosine_similarity(tru_cls_feat, pred_cls_feat, dim=1, eps=self.eps) )/2. # bsz[*numchoices-1]  // [-1, 1] -> [0,1]
        n_cosdist= (1. - F.cosine_similarity(fals_cls_feat, pred_cls_feat, dim=1, eps=self.eps) )/2. # bsz[*numchoices-1]  // [-1, 1] -> [0,1]


        loss_tensor = 1. + p_cosdist - n_cosdist  # max:2, min:0

        #reduce

        if self.reduction == 'mean':
            loss = loss_tensor.mean()
        elif self.reduction == 'sum':
            loss = loss_tensor.sum()
        elif self.reduction == 'none':
            loss = loss_tensor

        return loss
