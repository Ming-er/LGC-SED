import os
import random
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from .con_loss import SupConLoss
from desed_task.data_augm import mixup, filt_aug, frame_shift, time_mask, cut_mix
from desed_task.utils.scaler import TorchScaler
from sklearn.cluster import KMeans
import numpy as np
import torchmetrics
import torch.nn.functional as F
from .utils import (
    batched_decode_preds,
    log_sedeval_metrics,
)
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
)

from codecarbon import EmissionsTracker


class SEDTask4(pl.LightningModule):
    """ Pytorch lightning module for the SED 2021 baseline
    Args:
        hparams: dict, the dictionary to be used for the current experiment/
        encoder: ManyHotEncoder object, object to encode and decode labels.
        sed_student: torch.Module, the student model to be trained. The teacher model will be
        opt: torch.optimizer.Optimizer object, the optimizer to be used
        train_data: torch.utils.data.Dataset subclass object, the training data to be used.
        valid_data: torch.utils.data.Dataset subclass object, the validation data to be used.
        test_data: torch.utils.data.Dataset subclass object, the test data to be used.
        train_sampler: torch.utils.data.Sampler subclass object, the sampler to be used in the training dataloader.
        scheduler: asteroid.engine.schedulers.BaseScheduler subclass object, the scheduler to be used. This is
            used to apply ramp-up during training for example.
        fast_dev_run: bool, whether to launch a run with only one batch for each set, this is for development purpose,
            to test the code runs.
    """

    def __init__(
        self,
        hparams,
        encoder,
        sed_student,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
        fast_dev_run=False,
        evaluation=False,
        synth_data=None,
        sed_teacher=None
    ):
        super(SEDTask4, self).__init__()
        self.hparams.update(hparams)

        try:
            log_dir = self.logger.log_dir
        except Exception as e:
            log_dir = self.hparams["log_dir"]
        self.exp_dir = log_dir

        self.encoder = encoder
        self.sed_student = sed_student
        if sed_teacher is None:
            self.sed_teacher = deepcopy(sed_student)
        else:
            self.sed_teacher = sed_teacher
        self.opt = opt
        self.synth_data = synth_data
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        self.fast_dev_run = fast_dev_run
        self.evaluation = evaluation

        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]

        feat_params = self.hparams["feats"]
        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )
        for param in self.sed_teacher.parameters():
            param.detach_()

        # instantiating losses
        self.supervised_loss = torch.nn.BCELoss()
        if hparams["training"]["self_sup_loss"] == "mse":
            self.selfsup_loss = torch.nn.MSELoss()
        elif hparams["training"]["self_sup_loss"] == "bce":
            self.selfsup_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError

        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro = torchmetrics.classification.f_beta.F1(
            len(self.encoder.labels),
            average="macro",
            compute_on_step=False,
        )

        self.get_weak_teacher_f1_seg_macro = torchmetrics.classification.f_beta.F1(
            len(self.encoder.labels),
            average="macro",
            compute_on_step=False,
        )

        self.scaler = self._init_scaler()
        # buffer for event based scores which we compute using sed-eval

        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.val_buffer_student_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer_student = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_teacher = {k: pd.DataFrame() for k in test_thresholds}
        self.decoded_student_05_buffer = pd.DataFrame()
        self.decoded_teacher_05_buffer = pd.DataFrame()

        self.num_centers = self.hparams["LGC"]["proto_nums"]
        self.start_contrast_epochs = self.hparams["LGC"]["start_contrast_epochs"]
        self.num_class = self.hparams["LGC"]["num_class"]
        self.prototype_ema = self.hparams["LGC"]["prototype_ema"]
        self.feat_dim = self.hparams["LGC"]["feat_dim"]
        self.pos_thresh = self.hparams["LGC"]["pos_thresh"]
        self.neg_thresh = self.hparams["LGC"]["neg_thresh"]

        self.prototype_vec = []

    def init_prototypes(self):
        self.sed_teacher.eval()
        indx_synth, indx_weak, indx_unlabelled = self.hparams["training"]["batch_size"]
        feats_lst = [[] for _ in range(self.num_class + 1)]
        
        for _, (audio, labels, _) in enumerate(self.train_loader):
            audio = audio.to('cuda')
            labels = labels.to('cuda')
            features = self.mel_spec(audio)
            
            batch_num = features.shape[0]
            
            strong_mask = torch.zeros(batch_num).to(features).bool()
            ulb_mask = torch.zeros(batch_num).to(features).bool()
        
            strong_mask[:indx_synth] = 1
            ulb_mask[indx_synth:] = 1
            
            # for labeled data, we use ground truth labels
            st_labels = labels[strong_mask].permute(0, 2, 1).reshape(-1, self.num_class)
            # for unlabeled data, we use pseudo labels
            strong_preds_teacher, _, cnn_frame_features_t = self.detect(
                features, self.sed_teacher, proj=True
            )    
            p_labels = strong_preds_teacher[ulb_mask].permute(0, 2, 1).reshape(-1, self.num_class)
            # cat together
            labels = torch.cat([st_labels, p_labels], dim=0)
            
            for cls in range(self.num_class + 1):
                # high-quality features
                if cls < self.num_class:
                    cls_idx = (labels[:, cls].squeeze(-1) >= self.pos_thresh).nonzero().squeeze(-1)
                else:
                    cls_idx = ((labels.max(1)[0] <= 1 - self.pos_thresh).nonzero()).squeeze(-1) # background class

                if cls_idx.size(0) == 0:
                    continue    
                    
                lb_features_t = cnn_frame_features_t[cls_idx].clone()
                feats_lst[cls].append(lb_features_t)
        
        for cls in range(self.num_class + 1):
            feats_lst[cls] = torch.cat(feats_lst[cls], dim=0)
            
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        for cls in range(self.num_class + 1):
            cls_feat_samples = feats_lst[cls].cpu().numpy().astype('float32')
            # initialze prototypes by kmeans
            kmeans.fit(cls_feat_samples)
            cls_centers = torch.tensor(kmeans.cluster_centers_)
            cls_centers = F.normalize(cls_centers)
            self.prototype_vec.append(cls_centers)
            
        self.sed_teacher.train()
    
    def update_prototypes(self, features_t, pred_t, st_labels=None):
        frame_pred_t = pred_t.detach().clone()
        if st_labels == None:
            frame_pred_t = frame_pred_t.permute(0, 2, 1).view(-1, self.num_class)
        else:
            st_len = st_labels.size(0)
            # for labeled data, we use ground truth labels
            frame_pred_t[: st_len] = st_labels
            frame_pred_t = frame_pred_t.permute(0, 2, 1).reshape(-1, self.num_class)
        
        for lb in range(self.num_class + 1):
            # high-quality features
            if lb < self.num_class:
                lb_idx = (frame_pred_t[:, lb].squeeze(-1) >= self.pos_thresh).nonzero().squeeze(-1)
            else:
                lb_idx = ((frame_pred_t.max(1)[0] < 1 - self.pos_thresh).nonzero()).squeeze(-1) # background class
            if lb_idx.size(0) == 0:
                continue    
            cls_prototypes = self.prototype_vec[lb].to('cuda') # (m, d)
            lb_features_t = features_t[lb_idx].clone() # (b, d)
            prototypes_idx = torch.max(torch.matmul(lb_features_t, cls_prototypes.T), dim=1)[1] # equation 5 in our paper, (b, m) -> (b)
            for m in range(self.num_centers):
                i = (prototypes_idx == m).nonzero()
                if i.size(0) == 0:
                    continue
                cur_mean_vector = torch.mean(lb_features_t[i], dim = 0) # equation 5 in our paper
                # equation 4 in our paper, update with momentum
                self.prototype_vec[lb][m] = (self.prototype_ema * self.prototype_vec[lb][m]) + (1 - self.prototype_ema) * cur_mean_vector.cpu()   
                self.prototype_vec[lb][m] = F.normalize(self.prototype_vec[lb][m], dim=0)
        
    def gather_protos(self, to_device):
        proto_labels = torch.zeros((self.num_class + 1, self.num_class + 1)).fill_diagonal_(1)
        proto_feats = torch.stack(self.prototype_vec, dim=0)
        return proto_labels.to(to_device), proto_feats.to(to_device)
    
    def select_contrastive_samples(self, pred_s, pred_t):
        # (selective anchor sampling (2))
        background_prob_s = ((1.0 - pred_s.max(1)[0])).float().unsqueeze(1)
        background_prob_t = ((1.0 - pred_t.max(1)[0])).float().unsqueeze(1)
        
        pred_s = torch.cat([pred_s, background_prob_s], dim=1)  
        pred_t = torch.cat([pred_t, background_prob_t], dim=1)  
        # > tau + and < tau -
        sel_idx = ((pred_t > self.pos_thresh).logical_and(pred_s < self.neg_thresh)).sum(-1).nonzero().squeeze(-1)
        return sel_idx
    
    def on_train_start(self) -> None:

        os.makedirs(os.path.join(self.exp_dir, "training_codecarbon"), exist_ok=True)
        self.tracker_train = EmissionsTracker("DCASE Task 4 SED TRAINING",
                                        output_dir=os.path.join(self.exp_dir,
                                                                "training_codecarbon"))
        self.tracker_train.start()

    def update_ema(self, alpha, global_step, model, ema_model):
        """ Update teacher model parameters

        Args:
            alpha: float, the factor to be used between each updated step.
            global_step: int, the current global step to be used.
            model: torch.Module, student model to use
            ema_model: torch.Module, teacher model to use
        """
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(ema_model.parameters(), model.parameters()):
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

    def _init_scaler(self):
        """ Scaler inizialization

        Raises:
            NotImplementedError: in case of not Implemented scaler

        Returns:
            TorchScaler: returns the scaler
        """

        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(
                "instance",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )

            return scaler
        elif self.hparams["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
        )

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler

    def take_log(self, mels):
        """ Apply the log transformation to mel spectrograms.
        Args:
            mels: torch.Tensor, mel spectrograms for which to apply log.

        Returns:
            Tensor: logarithmic mel spectrogram of the mel spectrogram given as input
        """

        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def detect(self, mel_feats, model, proj=False):
        return model(self.scaler(self.take_log(mel_feats)), proj=proj)

    def training_step(self, batch, batch_indx):
        """ Apply the training for one batch (a step). Used during trainer.fit

        Args:
            batch: torch.Tensor, batch input tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.

        Returns:
           torch.Tensor, the loss to take into account.
        """
        audio, labels, padded_indxs = batch
        indx_synth, indx_weak, indx_unlabelled = self.hparams["training"]["batch_size"]
        features = self.mel_spec(audio)
        feat_len = features.size(-1) // 4

        batch_num = features.shape[0]
        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num).to(features).bool()
        weak_mask = torch.zeros(batch_num).to(features).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth : indx_weak + indx_synth] = 1

        # deriving weak labels
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()
        
        mixup_type = self.hparams["training"].get("mixup")
        if mixup_type is not None and 0.5 > random.random():
            features[weak_mask], labels_weak = mixup(
                features[weak_mask], labels_weak, mixup_label_type=mixup_type
            )
            features[strong_mask], labels[strong_mask] = mixup(
                features[strong_mask], labels[strong_mask], mixup_label_type=mixup_type
            ) 
        
        # sed student forward
        strong_preds_student, weak_preds_student, cnn_frame_features_s = self.detect(
            features, self.sed_student, proj=True
        )
        
        # supervised loss on strong labels
        loss_strong = self.supervised_loss(
            strong_preds_student[strong_mask], labels[strong_mask]
        )
        # supervised loss on weakly labelled
        loss_weak = self.supervised_loss(weak_preds_student[weak_mask], labels_weak)
        
        # total supervised loss
        tot_loss_supervised = loss_strong + loss_weak
        
        with torch.no_grad():
            # after several epoch, we involve the global consistency regularization
            if self.current_epoch > self.start_contrast_epochs:
                strong_preds_teacher, weak_preds_teacher, cnn_frame_features_t = self.detect(
                    features, self.sed_teacher, proj=True
                )
                self.update_prototypes(cnn_frame_features_t, strong_preds_teacher, labels[strong_mask])    
            else:
                strong_preds_teacher, weak_preds_teacher = self.detect(
                    features, self.sed_teacher, proj=False
                )

            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[strong_mask], labels[strong_mask]
            )
            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[weak_mask], labels_weak
            )
            
        # we apply consistency between the predictions, use the scheduler for learning rate (to be changed ?)
        weight = (
            self.hparams["training"]["const_max"]
            * self.scheduler["scheduler"]._get_scaling_factor()
        )

        strong_self_sup_loss = self.selfsup_loss(
            strong_preds_student, strong_preds_teacher.detach()
        )
        weak_self_sup_loss = self.selfsup_loss(
            weak_preds_student, weak_preds_teacher.detach()
        )
        
        pseudo_labels = strong_preds_teacher.detach().clone()
        # cutmix
        mixed_features, mixed_pseudo_labels, mixed_ulb_frame = cut_mix(features, pseudo_labels)
        # ignore labeled samples (selective anchor sampling (1))
        mixed_ulb_idx = torch.nonzero(mixed_ulb_frame.reshape(-1)).reshape(-1)
        
        strong_preds_student_cutmix, _, cnn_frame_features_cutmix = self.detect(
            mixed_features, self.sed_student, proj=True
        )
        # Loss_{CLC}
        strong_cutmix_loss = self.selfsup_loss(
            strong_preds_student_cutmix, mixed_pseudo_labels.detach()
        )
        
        tot_self_loss = (strong_self_sup_loss + weak_self_sup_loss + strong_cutmix_loss) * weight
        
        frame_pred_stu = strong_preds_student_cutmix.detach().clone().permute(0, 2, 1).view(-1, self.num_class)[mixed_ulb_idx]
        frame_pred_tea = mixed_pseudo_labels.detach().clone().permute(0, 2, 1).view(-1, self.num_class)[mixed_ulb_idx]
        
        anchor_feats = cnn_frame_features_cutmix[mixed_ulb_idx]
        contrastive_criterion = SupConLoss()
        if self.current_epoch > self.start_contrast_epochs:
            proto_labels, proto_feats = self.gather_protos(anchor_feats.device)
            # select frames where the student is likely to make wrong preds due to the inferiority of frame features
            sel_idx = self.select_contrastive_samples(frame_pred_stu, frame_pred_tea)
            anchor_feats, anchor_labels = anchor_feats[sel_idx], frame_pred_tea[sel_idx]
            # Loss_{PGC}
            contrastive_loss = contrastive_criterion(anchor_feats, anchor_labels, proto_feats, proto_labels) 
            contrastive_loss = contrastive_loss * 0.10
            tot_loss = tot_loss_supervised + tot_self_loss + contrastive_loss
        else:
            contrastive_loss = 0
            tot_loss = tot_loss_supervised + tot_self_loss
            
        self.log("train/student/loss_strong", loss_strong)
        self.log("train/student/loss_weak", loss_weak)
        self.log("train/teacher/loss_strong", loss_strong_teacher)
        self.log("train/teacher/loss_weak", loss_weak_teacher)
        self.log("train/step", self.scheduler["scheduler"].step_num, prog_bar=True)
        self.log("train/student/tot_self_loss", tot_self_loss, prog_bar=True)
        self.log("train/weight", weight)
        self.log("train/student/tot_supervised", tot_loss_supervised, prog_bar=True)
        self.log("train/student/weak_self_sup_loss", weak_self_sup_loss)
        self.log("train/student/strong_self_sup_loss", strong_self_sup_loss)
        self.log("train/student/contrastive_loss", contrastive_loss)
        self.log("train/lr", self.opt.param_groups[-1]["lr"], prog_bar=True)

        return tot_loss

    def on_before_zero_grad(self, *args, **kwargs):
        # update EMA teacher
        self.update_ema(
            self.hparams["training"]["ema_factor"],
            self.scheduler["scheduler"].step_num,
            self.sed_student,
            self.sed_teacher,
        )
        
    def training_epoch_end(self, outputs):
        if self.current_epoch == self.start_contrast_epochs:
            self.init_prototypes()

    def validation_step(self, batch, batch_indx):
        """ Apply validation to a batch (step). Used during trainer.fit

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, padded_indxs, filenames = batch


        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(mels, self.sed_student)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher, cnn_frame_features_t = self.detect(mels, self.sed_teacher, proj=True)
                
        # we derive masks for each dataset based on folders of filenames
        mask_weak = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["weak_folder"]))
                    for x in filenames
                ]
            )
            .to(audio)
            .bool()
        )
        mask_synth = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["val_folder"]))
                    for x in filenames
                ]
            )
            .to(audio)
            .bool()
        )

        if torch.any(mask_weak):
            labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()

            loss_weak_student = self.supervised_loss(
                weak_preds_student[mask_weak], labels_weak
            )
            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[mask_weak], labels_weak
            )
            self.log("val/weak/student/loss_weak", loss_weak_student)
            self.log("val/weak/teacher/loss_weak", loss_weak_teacher)

            # accumulate f1 score for weak labels
            self.get_weak_student_f1_seg_macro(
                weak_preds_student[mask_weak], labels_weak.long()
            )
            self.get_weak_teacher_f1_seg_macro(
                weak_preds_teacher[mask_weak], labels_weak.long()
            )

        if torch.any(mask_synth):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_synth], labels[mask_synth]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_synth], labels[mask_synth]
            )

            self.log("val/strong/student/loss_strong", loss_strong_student)
            self.log("val/strong/teacher/loss_strong", loss_strong_teacher)

            filenames_synth = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["val_folder"])
            ]

            decoded_student_strong = batched_decode_preds(
                strong_preds_student[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.val_buffer_student_synth.keys()),
            )

            for th in self.val_buffer_student_synth.keys():
                self.val_buffer_student_synth[th] = self.val_buffer_student_synth[
                    th
                ].append(decoded_student_strong[th], ignore_index=True)

            decoded_teacher_strong = batched_decode_preds(
                strong_preds_teacher[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.val_buffer_teacher_synth.keys()),
            )
            for th in self.val_buffer_teacher_synth.keys():
                self.val_buffer_teacher_synth[th] = self.val_buffer_teacher_synth[
                    th
                ].append(decoded_teacher_strong[th], ignore_index=True)

        return

    def validation_epoch_end(self, outputs):
        """ Fonction applied at the end of all the validation steps of the epoch.

        Args:
            outputs: torch.Tensor, the concatenation of everything returned by validation_step.

        Returns:
            torch.Tensor, the objective metric to be used to choose the best model from for example.
        """

        weak_student_f1_macro = self.get_weak_student_f1_seg_macro.compute()
        weak_teacher_f1_macro = self.get_weak_teacher_f1_seg_macro.compute()

        # synth dataset
        intersection_f1_macro_student = compute_per_intersection_macro_f1(
            self.val_buffer_student_synth,
            self.hparams["data"]["val_tsv"],
            self.hparams["data"]["val_dur"],
        )

        synth_student_event_macro = log_sedeval_metrics(
            self.val_buffer_student_synth[0.5], self.hparams["data"]["val_tsv"],
        )[0]

        intersection_f1_macro_teacher = compute_per_intersection_macro_f1(
            self.val_buffer_teacher_synth,
            self.hparams["data"]["val_tsv"],
            self.hparams["data"]["val_dur"],
        )

        synth_teacher_event_macro = log_sedeval_metrics(
            self.val_buffer_teacher_synth[0.5], self.hparams["data"]["val_tsv"],
        )[0]

        obj_metric_synth_type = self.hparams["training"].get("obj_metric_synth_type")
        if obj_metric_synth_type is None:
            synth_metric = intersection_f1_macro_student
        elif obj_metric_synth_type == "event":
            synth_metric = synth_student_event_macro
        elif obj_metric_synth_type == "intersection":
            synth_metric = intersection_f1_macro_student
        elif obj_metric_synth_type == "teacher_event":
            synth_metric = synth_teacher_event_macro
        elif obj_metric_synth_type == "teacher_intersection":
            synth_metric = intersection_f1_macro_teacher
        elif obj_metric_synth_type == "teacher_intersection_weak_f1":
            synth_metric = intersection_f1_macro_teacher + weak_teacher_f1_macro
        else:
            raise NotImplementedError(
                f"obj_metric_synth_type: {obj_metric_synth_type} not implemented."
            )

        # obj_metric = torch.tensor(weak_student_f1_macro.item() + synth_metric)
        obj_metric = torch.tensor(synth_metric)

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/weak/student/macro_F1", weak_student_f1_macro)
        self.log("val/weak/teacher/macro_F1", weak_teacher_f1_macro)
        self.log(
            "val/strong/student/intersection_f1_macro", intersection_f1_macro_student
        )
        self.log(
            "val/strong/teacher/intersection_f1_macro", intersection_f1_macro_teacher
        )
        self.log("val/strong/student/event_f1_macro", synth_student_event_macro)
        self.log("val/strong/teacher/event_f1_macro", synth_teacher_event_macro)

        # free the buffers
        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.get_weak_student_f1_seg_macro.reset()
        self.get_weak_teacher_f1_seg_macro.reset()

        return obj_metric
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint["sed_student"] = self.sed_student.state_dict()
        checkpoint["sed_teacher"] = self.sed_teacher.state_dict()
        return checkpoint

    def test_step(self, batch, batch_indx):
        """ Apply Test to a batch (step), used only when (trainer.test is called)

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """
        
        audio, labels, padded_indxs, filenames = batch        
        
        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(mels, self.sed_student)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher, cnn_frame_features_t = self.detect(mels, self.sed_teacher, proj=True)
        
        if not self.evaluation:
            loss_strong_student = self.supervised_loss(strong_preds_student, labels)
            loss_strong_teacher = self.supervised_loss(strong_preds_teacher, labels)

            self.log("test/student/loss_strong", loss_strong_student)
            self.log("test/teacher/loss_strong", loss_strong_teacher)

        # compute psds
        decoded_student_strong = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_student.keys()),
        )

        for th in self.test_psds_buffer_student.keys():
            self.test_psds_buffer_student[th] = self.test_psds_buffer_student[
                th
            ].append(decoded_student_strong[th], ignore_index=True)

        decoded_teacher_strong = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_teacher.keys()),
        )

        for th in self.test_psds_buffer_teacher.keys():
            self.test_psds_buffer_teacher[th] = self.test_psds_buffer_teacher[
                th
            ].append(decoded_teacher_strong[th], ignore_index=True)

        
        # compute f1 score
        decoded_student_strong = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )

        self.decoded_student_05_buffer = self.decoded_student_05_buffer.append(
            decoded_student_strong[0.5]
        )

        decoded_teacher_strong = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )

        self.decoded_teacher_05_buffer = self.decoded_teacher_05_buffer.append(
            decoded_teacher_strong[0.5]
        )

    def on_test_epoch_end(self):
        # pub eval dataset
        save_dir = os.path.join(self.exp_dir, "metrics_test")
        
        if self.evaluation:
            # only save the predictions
            save_dir_student = os.path.join(save_dir, "student")
            os.makedirs(save_dir_student, exist_ok=True)
            self.decoded_student_05_buffer.to_csv(
                os.path.join(save_dir_student, f"predictions_05_student.tsv"),
                sep="\t",
                index=False
            )

            for k in self.test_psds_buffer_student.keys():
                self.test_psds_buffer_student[k].to_csv(
                    os.path.join(save_dir_student, f"predictions_th_{k:.2f}.tsv"),
                    sep="\t",
                    index=False,
                )
            print(f"\nPredictions for student saved in: {save_dir_student}")
            
            save_dir_teacher = os.path.join(save_dir, "teacher")
            os.makedirs(save_dir_teacher, exist_ok=True)
           
            self.decoded_teacher_05_buffer.to_csv(
                os.path.join(save_dir_teacher, f"predictions_05_teacher.tsv"),
                sep="\t",
                index=False
            )

            for k in self.test_psds_buffer_student.keys():
                self.test_psds_buffer_student[k].to_csv(
                    os.path.join(save_dir_teacher, f"predictions_th_{k:.2f}.tsv"),
                    sep="\t",
                    index=False,
                )
            print(f"\nPredictions for teacher saved in: {save_dir_teacher}")

        else:
            # calculate the metrics
            psds_score_scenario1 = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )

            psds_score_scenario2 = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )

            psds_score_teacher_scenario1 = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )

            psds_score_teacher_scenario2 = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )

            event_macro_student = log_sedeval_metrics(
                self.decoded_student_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "student"),
            )[0]

            event_macro_teacher = log_sedeval_metrics(
                self.decoded_teacher_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "teacher"),
            )[0]

            # synth dataset
            intersection_f1_macro_student = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_student_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            # synth dataset
            intersection_f1_macro_teacher = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_teacher_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            best_test_result = torch.tensor(max(psds_score_scenario1, psds_score_scenario2))

            results = {
                "hp_metric": best_test_result,
                "test/student/psds_score_scenario1": psds_score_scenario1,
                "test/student/psds_score_scenario2": psds_score_scenario2,
                "test/teacher/psds_score_scenario1": psds_score_teacher_scenario1,
                "test/teacher/psds_score_scenario2": psds_score_teacher_scenario2,
                "test/student/event_f1_macro": event_macro_student,
                "test/student/intersection_f1_macro": intersection_f1_macro_student,
                "test/teacher/event_f1_macro": event_macro_teacher,
                "test/teacher/intersection_f1_macro": intersection_f1_macro_teacher,
            }

            if self.evaluation:
                self.tracker_eval.stop()
                eval_kwh = self.tracker_eval._total_energy.kwh
                results.update({"/eval/tot_energy_kWh": torch.tensor(float(eval_kwh))})
                with open(os.path.join(self.exp_dir, "evaluation_codecarbon", "eval_tot_kwh.txt"), "w") as f:
                    f.write(str(eval_kwh))
            else:
                self.tracker_devtest.stop()
                eval_kwh = self.tracker_devtest._total_energy.kwh
                results.update({"/test/tot_energy_kWh": torch.tensor(float(eval_kwh))})
                with open(os.path.join(self.exp_dir, "devtest_codecarbon", "devtest_tot_kwh.txt"), "w") as f:
                    f.write(str(eval_kwh))

            if self.logger is not None:
                self.logger.log_metrics(results)
                self.logger.log_hyperparams(self.hparams, results)

            for key in results.keys():
                self.log(key, results[key], prog_bar=True, logger=False)

    def configure_optimizers(self):
        return [self.opt], [self.scheduler]

    def train_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

        return self.train_loader
    
    def synth_dataloader(self):
        return torch.utils.data.DataLoader(
            self.synth_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.test_loader

    def on_train_end(self) -> None:
        # dump consumption
        self.tracker_train.stop()
        training_kwh = self.tracker_train._total_energy.kwh
        self.logger.log_metrics({"/train/tot_energy_kWh": torch.tensor(float(training_kwh))})
        with open(os.path.join(self.exp_dir, "training_codecarbon", "training_tot_kwh.txt"), "w") as f:
            f.write(str(training_kwh))

    def on_test_start(self) -> None:

        if self.evaluation:
            os.makedirs(os.path.join(self.exp_dir, "evaluation_codecarbon"), exist_ok=True)
            self.tracker_eval = EmissionsTracker("DCASE Task 4 SED EVALUATION",
                                                 output_dir=os.path.join(self.exp_dir,
                                                                         "evaluation_codecarbon"))
            self.tracker_eval.start()
        else:
            os.makedirs(os.path.join(self.exp_dir, "devtest_codecarbon"), exist_ok=True)
            self.tracker_devtest = EmissionsTracker("DCASE Task 4 SED DEVTEST",
                                                 output_dir=os.path.join(self.exp_dir,
                                                                         "devtest_codecarbon"))
            self.tracker_devtest.start()


