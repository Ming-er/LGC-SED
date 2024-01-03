import numpy as np
import torch
import random

def time_mask(features, labels=None, net_pooling=4, mask_ratios=(5, 20)):
    _, _, n_frame = labels.shape
    t_width = torch.randint(low=int(n_frame/mask_ratios[1]), high=int(n_frame/mask_ratios[0]), size=(1,))   # [low, high)
    t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
    features[:, :, t_low * net_pooling:(t_low+t_width)*net_pooling] = 0
    labels[:, :, t_low:t_low+t_width] = 0
    return features, labels

def filt_aug(features, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
    if not isinstance(filter_type, str):
        if torch.rand(1).item() < filter_type:
            filter_type = "step"
            n_band = [2, 5]
            min_bw = 4
        else:
            filter_type = "linear"
            n_band = [3, 6]
            min_bw = 6

    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()   # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        if filter_type == "step":
            band_factors = torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

        elif filter_type == "linear":
            band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                for j in range(batch_size):
                    freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                        torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                       band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
            freq_filt = 10 ** (freq_filt / 20)
        return features * freq_filt

    else:
        return features
    
def cut_mix(data, target, indx_synth=12, low_r=0.20, high_r=0.50):
    """
    Args:
        features_stu: features from student model.
        features_proto: features from prototypes.
        pseudo_lb_stu: frame-level prob vector for student preds.
        lb_proto: labels for prototypes.
    """       
    with torch.no_grad():
        batch_size, feat_dims, feat_len  = data.size()
        _, _, n_frame = target.size()
        net_pooling = feat_len // feat_dims

        mask_ratio = np.random.uniform(low_r, high_r, batch_size)

        # mix length
        lb_mask_len = (mask_ratio * n_frame).astype(int) # for target
        feat_mask_len = lb_mask_len * net_pooling # for data

        # start point
        lb_st = np.random.randint(0, n_frame - lb_mask_len, batch_size)
        feat_st = lb_st * net_pooling

        lb_mask = torch.ones((batch_size, n_frame))
        feat_mask = torch.ones((batch_size, feat_len))
        ulb_frame = torch.ones((batch_size, n_frame))
        # trace the unlabeled sample
        ulb_frame[:indx_synth] = 0

        for i in range(batch_size):
            lb_mask[i, lb_st[i] : lb_st[i] + lb_mask_len[i]] = 0           
            feat_mask[i, feat_st[i] : feat_st[i] + feat_mask_len[i]] = 0
        
        lb_mask = lb_mask.unsqueeze(-1).permute(0, 2, 1).cuda()
        feat_mask = feat_mask.unsqueeze(-1).permute(0, 2, 1).cuda()
        ulb_frame = ulb_frame.unsqueeze(-1).permute(0, 2, 1).cuda()
        
        # cutmix
        perm = torch.randperm(batch_size)
        mixed_data = feat_mask * data + (1 - feat_mask) * data[perm, :]
        mixed_target = lb_mask * target + (1 - lb_mask) * target[perm, :]
        mixed_ulb_frame = lb_mask * ulb_frame + (1 - lb_mask) * ulb_frame[perm, :]

        return mixed_data, mixed_target, mixed_ulb_frame

    
def frame_shift(mels, labels, net_pooling=4):
    bsz, n_bands, frames = mels.shape
    shifted = []
    new_labels = []
    for bindx in range(bsz):
        shift = int(random.gauss(0, 90))
        shifted.append(torch.roll(mels[bindx], shift, dims=-1))
        shift = -abs(shift) // net_pooling if shift < 0 else shift // net_pooling
        new_labels.append(torch.roll(labels[bindx], shift, dims=-1))
    return torch.stack(shifted), torch.stack(new_labels)


def mixup(data, target=None, alpha=0.2, beta=0.2, mixup_label_type="soft"):
    """Mixup data augmentation by permuting the data

    Args:
        data: input tensor, must be a batch so data can be permuted and mixed.
        target: tensor of the target to be mixed, if None, do not return targets.
        alpha: float, the parameter to the np.random.beta distribution
        beta: float, the parameter to the np.random.beta distribution
        mixup_label_type: str, the type of mixup to be used choice between {'soft', 'hard'}.
    Returns:
        torch.Tensor of mixed data and labels if given
    """
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, beta)

        perm = torch.randperm(batch_size)

        mixed_data = c * data + (1 - c) * data[perm, :]
        if target is not None:
            if mixup_label_type == "soft":
                mixed_target = torch.clamp(
                    c * target + (1 - c) * target[perm, :], min=0, max=1
                )
            elif mixup_label_type == "hard":
                mixed_target = torch.clamp(target + target[perm, :], min=0, max=1)
            else:
                raise NotImplementedError(
                    f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                    f"{'soft', 'hard'}"
                )

            return mixed_data, mixed_target
        else:
            return mixed_data


def add_noise(mels, snrs=(6, 30), dims=(1, 2)):
    """ Add white noise to mels spectrograms
    Args:
        mels: torch.tensor, mels spectrograms to apply the white noise to.
        snrs: int or tuple, the range of snrs to choose from if tuple (uniform)
        dims: tuple, the dimensions for which to compute the standard deviation (default to (1,2) because assume
            an input of a batch of mel spectrograms.
    Returns:
        torch.Tensor of mels with noise applied
    """
    if isinstance(snrs, (list, tuple)):
        snr = (snrs[0] - snrs[1]) * torch.rand(
            (mels.shape[0],), device=mels.device
        ).reshape(-1, 1, 1) + snrs[1]
    else:
        snr = snrs

    snr = 10 ** (snr / 20)  # linear domain
    sigma = torch.std(mels, dim=dims, keepdim=True) / snr
    mels = mels + torch.randn(mels.shape, device=mels.device) * sigma

    return mels
