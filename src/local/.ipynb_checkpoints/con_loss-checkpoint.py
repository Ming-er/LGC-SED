import pandas as pd
import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.10, pos_thresh=0.90):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.pos_thresh = pos_thresh

    def forward(self, features_stu,  pseudo_lb_stu, features_proto, lb_proto):
        """
        Args:
            features_stu: features from student model.
            features_proto: features from prototypes.
            pseudo_lb_stu: frame-level prob vector for student preds.
            lb_proto: labels for prototypes.
        """        
        bg_prob_stu = ((1.0 - pseudo_lb_stu.max(1)[0])).float().unsqueeze(1) # background class
        pseudo_lb_stu = torch.cat([pseudo_lb_stu, bg_prob_stu], dim=1)  
        
        # get (hard) mask
        pseudo_lb_stu[pseudo_lb_stu > self.pos_thresh] = 1.0
        pseudo_lb_stu[pseudo_lb_stu < self.pos_thresh] = 0.0   
        
        mask = pseudo_lb_stu
        
        # compute feature similarity logits
        # features_stu (n, d), features_proto (c, m, d) -> (c, m, n) -> (n, c, m) -> (n, c)
        feat_sim_mat = torch.max(torch.matmul(features_proto, features_stu.T).permute(2, 0, 1), dim=2)[0]
        feat_sim_mat = torch.div(feat_sim_mat, self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(feat_sim_mat, dim=1, keepdim=True)
        feat_sim_mat = feat_sim_mat - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(feat_sim_mat)
        log_prob = feat_sim_mat - torch.log(1e-7 + exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)

        # loss
        loss = - mean_log_prob_pos
        return loss.mean()
