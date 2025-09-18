import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import diags
from globals import device
from scipy.sparse import diags

class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        return self.mse_loss(outputs['predicts'], targets)


class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.xent_loss(outputs['predicts'], targets)



class SupConLoss(nn.Module):

    def __init__(self, alpha, temp, losspull=0):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp
        self.losspull = losspull

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            # delete diagonal elements
            mask = mask ^ torch.diag_embed(torch.diag(mask))

            # 以下是IP
            n = mask.shape[0]
            if self.losspull == 0:
                seq_mask = torch.eye(n).bool().to(device) # no additional positive samples
            else:
                if type(self.losspull) is not str: 
                    if self.losspull > 0:
                        offsets = list(range(1, self.losspull + 1))
                    elif self.losspull < 0:
                        offsets = list(range(self.losspull, 0))
                else:  # self.losspull is '±n'
                    offsets = list(range(-abs(int(self.losspull[1])), abs(int(self.losspull[1])) + 1))
                    offsets.remove(0)

                diagonals = [1] * len(offsets)
                seq_mask = diags(diagonals, offsets, shape=(n, n)).toarray()
                seq_mask = torch.tensor(seq_mask).bool().to(device)

            # Combine sequence mask with class-based mask
            mask = mask * seq_mask

        # compute logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        # delete diagonal elements
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        # compute log prob
        exp_logits = torch.exp(logits)
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # handle cases where mask.sum(1) is zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss


class DualLoss(SupConLoss):

    def __init__(self, alpha, temp, losspull):
        super().__init__(alpha, temp, losspull)

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)
        # 从 normed_label_feats 中提取与 targets 对应的特征，得到正样本特征
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1, normed_label_feats.size(-1))).squeeze(1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss_1 = 0.5 * self.alpha * self.nt_xent_loss(normed_pos_label_feats, normed_cls_feats, targets)
        cl_loss_2 = 0.5 * self.alpha * self.nt_xent_loss(normed_cls_feats, normed_pos_label_feats, targets)
        return ce_loss + cl_loss_1 + cl_loss_2



class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        class_preds = outputs['predicts']
        features = outputs['cls_feats']
        
        normalized_features = F.normalize(features, dim=-1)
        anchor_dot_contrast = torch.matmul(normalized_features, normalized_features.T)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits / self.temperature)
        cluster_sum = exp_logits.sum(dim=1, keepdim=True)
        log_prob = logits - torch.log(cluster_sum + 1e-12)
        
        labels = targets.unsqueeze(-1)
        mask = torch.eq(labels, labels.T).float()
        
        pos_log_prob = (log_prob * mask).sum(dim=1) / mask.sum(dim=1).detach()
        nce_loss = -1 * pos_log_prob.mean()
 
        ce_loss = self.xent_loss(class_preds, targets)
        
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * nce_loss
        return total_loss
    

class StocknetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xent_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        out, mu, log_var = outputs
        targets = F.one_hot(targets, 2).to(dtype=torch.float)
        recon_loss = self.xent_loss(out, targets)

        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        total_loss = recon_loss + kl_loss
        return total_loss


class AlstmLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xent_loss = nn.MSELoss()
        
    def forward(self, outputs, targets):
        targets = F.one_hot(targets, 2).to(dtype=torch.float)
        recon_loss = self.xent_loss(outputs, targets)
        return recon_loss
    
    
class TransamLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        
    def forward(self, outputs, targets):
        # outputs = outputs[:,-1,:]
        outputs = torch.argmax(outputs,dim=1)
        return self.loss(outputs, targets.to(dtype=torch.float)).requires_grad_(True)
    

def IndexganLoss(fake_output, fake_data, real_data,args):
    entropy = torch.mean(fake_output)
    mae = torch.mean(abs(fake_data-real_data))
    weight_accuracy = (0.8 * torch.sum(torch.sign(fake_data[real_data<0])==torch.sign(real_data[real_data<0]))+
                    0.2 * torch.sum(torch.sign(fake_data[real_data>=0])==torch.sign(real_data[real_data>=0]))) / fake_data.nelement()
    loss = -args.entropy_penalty * entropy + args.mae_penalty * mae - args.weight_acc_penalty * weight_accuracy
    return loss


class DTMLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        
    def forward(self, outputs, targets):
        # outputs = outputs[:,-1,:]
        outputs = torch.argmax(outputs,dim=1)
        return self.loss(outputs, targets.to(dtype=torch.float)).requires_grad_(True)
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)    

class ESPMPLoss(nn.Module):
    def __init__(self, alpha=0.5, temp=0.1):
        super().__init__()
        self.loss_fn = DualLoss(alpha=alpha, temp=temp)
        self.triplet_loss = TripletLoss()

    def forward(self, text, price, targets, anchor, positive, negative):
        outputs = self.model(text, price)
        combined_loss = self.loss_fn(outputs, targets)
        triplet_loss_value = self.triplet_loss(anchor, positive, negative)
        return combined_loss + triplet_loss_value