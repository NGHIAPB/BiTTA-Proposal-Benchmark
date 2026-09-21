from abc import abstractmethod
from collections import deque

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision

import conf
from data_loader.data_loader import load_cache, save_cache
from models.ResNet import ResNetDropout18
from utils.active_memory import ActivePriorityFIFO
from utils import memory, active_memory
from utils.calibration import expected_calibration_error
from utils.logging import *
from utils.loss_functions import *
from utils.memory import FIFO
from utils.normalize_layer import *
import utils.reset_utils as reset_utils
import random
import time
import os
import data_loader.data_loader as data_loader_module

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(conf.args.gpu_idx)

TRAINED = 0
SKIPPED = 1
FINISHED = 2


class DNN():
    """
    Base Deep Neural Network class for test-time adaptation methods.
    
    This class provides the foundation for implementing various test-time adaptation 
    algorithms with features like memory management, model checkpointing, online 
    evaluation, and active learning capabilities.
    """
    
    def __init__(self, model_, corruption_list_):
        """
        Initialize the DNN with model architecture and corruption settings.
        
        Args:
            model_: Model class or factory function
            corruption_list_: List of corruption types to handle
        """
        self.temp_value = 0
        self.count_bs1 = 63
        self.random_indexes_bs1 = np.arange(64)
        np.random.shuffle(self.random_indexes_bs1)
        self.random_indexes_bs1 = self.random_indexes_bs1[:3]
        
        self.count_skip = conf.args.enable_skip
        
        self.corruption_list = corruption_list_
        self.tgt_train_dist = conf.args.tgt_train_dist

        if conf.args.dataset == "imagenetR":
            filter_=conf.args.opt['indices_in_1k']
        else:
            filter_ = None

        # Init & prepare model. Kich ban chi dung ResNet-18: "resnet18" (kieu CIFAR, huan luyen tu dau) hoac
        # "resnet18_pretrained" (torchvision, khoi tao ImageNet).
        if conf.args.model == "resnet18_pretrained":
            pretrained = model_(pretrained=True)
            model = ResNetDropout18(filter=filter_)
            model.load_state_dict(pretrained.state_dict())
            del pretrained
        elif conf.args.model == "resnet18":
            model = model_(filter=filter_)
        else:
            raise NotImplementedError(conf.args.model)

        if conf.args.dataset == "imagenetR":
            self.net = model  # giu nguyen fc 1000 lop; ResNetDropout(filter=...) loc ra 200 lop cua ImageNet-R
        else:
            num_feats = model.fc.in_features
            num_class = conf.args.opt['num_class']
            model.fc = nn.Linear(num_feats, num_class)
            self.net = model

        if conf.args.load_checkpoint_path:
            self.load_checkpoint(conf.args.load_checkpoint_path)

        norm_layer = get_normalize_layer(conf.args.dataset)

        if norm_layer:
            self.net = torch.nn.Sequential(norm_layer, self.net)

        if conf.args.parallel and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)

        self.net.to(device)

        # Initialize optimizer (some TTA methods may overwrite this)
        self.optimizer = self.init_learner()
        self.class_criterion = nn.CrossEntropyLoss()

        # Initialize memory for online learning
        if conf.args.memory_type == 'FIFO':
            self.mem = memory.FIFO(capacity=conf.args.memory_size)
        elif conf.args.memory_type == "ActivePriorityFIFO":
            self.mem = active_memory.ActivePriorityFIFO(conf.args.update_every_x, pop="", delay=conf.args.feedback_delay)
        else:
            raise NotImplementedError(conf.args.memory_type)

        if conf.args.enable_bitta:
            self.active_mem = active_memory.ActivePriorityFIFO(conf.args.n_active_sample, pop="")
        else:
            self.active_mem = None

        self.fifo = memory.FIFO(conf.args.update_every_x)
        self.mem_state = self.mem.save_state_dict()
        self.net_state, self.optimizer_state = reset_utils.copy_model_and_optimizer(self.net, self.optimizer)

        self.num_batch_adapt = 0
        self.budget = 0

        # For BATTA tracking
        self.rank = []
        self.rank_wrong = []
        self.temperature = 1.0
        self.num_correct = 0
        self.num_wrong = 0
        self.conf_sum = 0.0
        self.conf_correct_sum = 0.0

        # C1: online Platt calibration of P(correct|x) from observed binary feedback.
        self._init_platt()

    # Smallest slope accepted by _fit_platt. Below this the calibration map is effectively flat,
    # which would make every candidate score identically and turn selection into a no-op.
    PLATT_A_MIN = 1e-3

    def _init_platt(self):
        """Reset the C1 calibration state. Split out so reset() can reuse it."""
        buf_size = getattr(conf.args, 'platt_buffer_size', 300)
        self.platt_conf_buf = deque(maxlen=buf_size)
        self.platt_bin_buf = deque(maxlen=buf_size)
        self.platt_a = 1.0
        self.platt_b = 0.0
        self.platt_fitted = False
        self.platt_n_calls = 0
        self.platt_n_rejected = 0

    @abstractmethod
    def init_learner(self):
        """
        Initialize the optimizer for the learning algorithm.
        This method should be overridden by subclasses to provide 
        algorithm-specific optimization configuration.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            conf.args.opt['learning_rate'],
            momentum=conf.args.opt['momentum'],
            weight_decay=conf.args.opt['weight_decay'],
            nesterov=True)
        return optimizer

    @abstractmethod
    def test_time_adaptation(self):
        """
        Perform test-time adaptation using samples from memory.
        This method should be implemented by subclasses to define
        the specific adaptation strategy.
        """
        assert isinstance(self.mem, FIFO)
        feats, labels, _ = self.mem.get_memory()
        feats = torch.stack(feats).to(device)
        labels = torch.Tensor(labels).type(torch.long).to(device)

        dataset = torch.utils.data.TensorDataset(feats, labels)
        data_loader = DataLoader(dataset, batch_size=conf.args.update_every_x,
                                 shuffle=True, drop_last=False, pin_memory=False)

        for e in range(conf.args.epoch):
            for batch_idx, (feats, _) in enumerate(data_loader):
                if len(feats) == 1:
                    self.net.eval()
                else:
                    self.net.train()

                if conf.args.method in ['Src']:
                    pass
                else:
                    raise NotImplementedError

    @abstractmethod
    def run_before_training(self):
        """
        Execute any necessary setup before training begins.
        This method should be implemented by subclasses.
        """
        pass

    def reset(self):
        """
        Reset the model and optimizer to their initial state.
        
        Raises:
            Exception: If no saved model/optimizer state is available
        """
        if self.net_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer")
        reset_utils.load_model_and_optimizer(self.net, self.optimizer, self.net_state, self.optimizer_state)
        self.mem.reset()
        # The Platt map calibrates the CURRENT net. Restoring the source net makes it stale.
        self._init_platt()

    def init_json(self, log_path):
        """
        Initialize JSON logging structures for evaluation and active learning metrics.
        
        Args:
            log_path (str): Path where log files will be written
        """
        self.write_path = log_path
        self.json_eval = {
            k: [] for k in ['gt', 'accuracy', 'current_accuracy',
                            'pred', 'confidence', 'entropy','energy','grad',
                            'dropout_confidence', 'dropout_01_confidence',
                            'original_ebce', 'dropout_ebce', 'cumul_original_ebce', 'cumul_dropout_ebce', 'cumul_dropout_01_ebce']
        }

        self.json_active = {
            k: [] for k in ["budgets", "correct_loss", "wrong_loss", "unlabel_loss",
                            "num_correct_mem", "num_wrong_mem",
                            # --- C1 selection diagnostics (see _log_selection_diagnostics) ---
                            "platt_a", "platt_b", "platt_fitted", "platt_n_pairs",
                            "platt_n_rejected", "overlap_mc_conf", "sel_mean_conf"]
        }
    def set_target_data(self, source_data_loader, source_val_data_loader, target_data_loader, corruption):
        """
        Set up target domain data for test-time adaptation.
        
        Args:
            source_data_loader: DataLoader for source domain training data
            source_val_data_loader: DataLoader for source domain validation data
            target_data_loader: DataLoader for target domain data
            corruption (str): Type of corruption applied to target data
        """
        self.source_dataloader = source_data_loader
        self.source_val_dataloader = source_val_data_loader
        self.target_dataloader = target_data_loader

        dataset = conf.args.dataset
        cond = corruption

        filename = f"{dataset}_{conf.args.seed}_dist{conf.args.tgt_train_dist}"

        if conf.args.tgt_train_dist == 4:
            filename += f"_gamma{conf.args.dirichlet_beta}"

        file_path = conf.args.opt['file_path'] + "_target_train_set"

        self.target_train_set = load_cache(filename, cond, file_path, transform=None)

        if not self.target_train_set:
            self.target_data_processing()
            save_cache(self.target_train_set, filename, cond, file_path, transform=None)
        

    def target_data_processing(self):
        """
        Process target domain data according to the specified distribution.
        
        Supports different target training distributions:
        - 0: Real distribution (sequential)
        - 1: Random distribution 
        - 4: Dirichlet distribution for class imbalance
        """
        features = []
        cl_labels = []
        do_labels = []

        for b_i, (feat, cl, dl) in enumerate(self.target_dataloader['train']):
            features.append(feat.squeeze(0))
            cl_labels.append(cl.squeeze())
            do_labels.append(dl.squeeze())

        tmp = list(zip(features, cl_labels, do_labels))
        features, cl_labels, do_labels = zip(*tmp)
        features, cl_labels, do_labels = list(features), list(cl_labels), list(do_labels)

        result_feats = []
        result_cl_labels = []
        result_do_labels = []

        tgt_train_dist_ = self.tgt_train_dist
        
        if tgt_train_dist_ == 0:  # real distribution
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = 0
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        elif tgt_train_dist_ == 1:  # random distribution
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = np.random.randint(len(features))
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        elif self.tgt_train_dist == 4:  # dirichlet distribution
            dirichlet_numchunks = conf.args.opt['num_class']
            num_class = conf.args.opt['num_class']

            min_size = -1
            N = len(features)
            min_size_thresh = 10
            while min_size < min_size_thresh:
                idx_batch = [[] for _ in range(dirichlet_numchunks)]
                idx_batch_cls = [[] for _ in range(dirichlet_numchunks)]
                for k in range(num_class):
                    cl_labels_np = torch.Tensor(cl_labels).numpy()
                    idx_k = np.where(cl_labels_np == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(
                        np.repeat(conf.args.dirichlet_beta, dirichlet_numchunks))

                    proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in
                                            zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

                    for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                        idx_j.append(idx)

            sequence_stats = []

            # Create temporally correlated dataset by shuffling classes
            for chunk in idx_batch_cls:
                cls_seq = list(range(num_class))
                np.random.shuffle(cls_seq)
                for cls in cls_seq:
                    idx = chunk[cls]
                    result_feats.extend([features[i] for i in idx])
                    result_cl_labels.extend([cl_labels[i] for i in idx])
                    result_do_labels.extend([do_labels[i] for i in idx])
                    sequence_stats.extend(list(np.repeat(cls, len(idx))))

            num_samples = conf.args.nsample if conf.args.nsample < len(result_feats) else len(result_feats)
            result_feats = result_feats[:num_samples]
            result_cl_labels = result_cl_labels[:num_samples]
            result_do_labels = result_do_labels[:num_samples]

        else:
            raise NotImplementedError

        remainder = len(result_feats) % conf.args.update_every_x
        if remainder != 0:
            result_feats = result_feats[:-remainder]
            result_cl_labels = result_cl_labels[:-remainder]
            result_do_labels = result_do_labels[:-remainder]

        try:
            self.target_train_set = (torch.stack(result_feats),
                                     torch.stack(result_cl_labels),
                                     torch.stack(result_do_labels))
        except:
            try:
                self.target_train_set = (torch.stack(result_feats),
                                         result_cl_labels,
                                         torch.stack(result_do_labels))
            except:
                self.target_train_set = (result_feats,
                                         result_cl_labels,
                                         torch.stack(result_do_labels))

    def save_checkpoint(self, epoch, epoch_acc, best_acc, checkpoint_path):
        """
        Save model checkpoint to specified path.
        
        Args:
            epoch: Current training epoch
            epoch_acc: Current epoch accuracy
            best_acc: Best accuracy achieved so far
            checkpoint_path (str): Path to save the checkpoint
        """
        if isinstance(self.net, nn.Sequential):
            if isinstance(self.net[0], NormalizeLayer) or isinstance(self.net[0], IdentityLayer):
                cp = self.net[1]
            else:
                raise NotImplementedError
        else:
            cp = self.net

        if isinstance(self.net, nn.DataParallel):
            cp = self.net.module

        torch.save(cp.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        if checkpoint_path.split(".")[-1] == ("pickle"):
            import pickle
            with open(os.path.join(checkpoint_path), 'rb') as f:
                loaded_net = pickle.load(f)

            if 'resnet18' in conf.args.model:
                if conf.args.dataset == "colored-mnist":
                    self.net = ResNetDropout18()
                    num_feats = self.net.fc.in_features
                    num_class = conf.args.opt['num_class']
                    self.net.fc = nn.Linear(num_feats, num_class)
                    self.net = self.net
                    self.net.load_state_dict(loaded_net.state_dict())
                    self.net.to(device)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self.checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{conf.args.gpu_idx}')
            self.net.load_state_dict(self.checkpoint, strict=True)
            self.net.to(device)

    def log_loss_results(self, condition, epoch, loss_avg, train_acc_avg=None, val_loss_avg=None, val_acc_avg=None):
        """
        Log training/validation loss and accuracy results.
        
        Args:
            condition (str): Training condition ('train', 'train_online', etc.)
            epoch (int): Current epoch or sample number
            loss_avg (float): Average loss value
            train_acc_avg (float, optional): Average training accuracy
            val_loss_avg (float, optional): Average validation loss
            val_acc_avg (float, optional): Average validation accuracy
            
        Returns:
            float: The loss average that was logged
        """
        if condition == 'train_online':
            print('{:s}: [current_sample: {:d}]'.format(condition, epoch))
        else:
            text = '{:s}: [epoch: {:d}]\tLoss: {:.6f} \t'.format(condition, epoch, loss_avg)
            if train_acc_avg != None:
                text += 'TrainAcc: {:.6f} \t'.format(train_acc_avg)
            if val_loss_avg != None:
                text += 'val_loss_avg: {:.6f} \t'.format(val_loss_avg)
            if val_acc_avg != None:
                text += 'val_acc_avg: {:.6f} \t'.format(val_acc_avg)
            print(text)

        return loss_avg

    def log_accuracy_results(self, condition, suffix, epoch, cm_class):
        """
        Log accuracy results from confusion matrix.
        
        Args:
            condition (str): Evaluation condition ('valid', 'test')
            suffix (str): Data subset identifier
            epoch (int): Current epoch
            cm_class: Confusion matrix for class predictions
            
        Returns:
            float: Class accuracy percentage
        """
        assert (condition in ['valid', 'test'])

        class_accuracy = 100.0 * np.sum(np.diagonal(cm_class)) / np.sum(cm_class)
        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, suffix, class_accuracy))

        return class_accuracy

    def train(self, epoch):
        """
        Train the model for one epoch on source domain data.
        
        Args:
            epoch (int): Current training epoch
            
        Returns:
            float: Average training loss for the epoch
        """
        self.net.train()

        class_loss_sum = 0.0
        class_loss_sum_val = 0.0
        total_iter = 0
        total_iter_val = 0
        total_num_correct = 0
        total_num_correct_val = 0

        total_num_samples = len(self.source_dataloader['train']) * conf.args.opt['batch_size']
        total_num_samples_val= 0

        if conf.args.method in ['Src', 'Src_Tgt']:
            num_iter = len(self.source_dataloader['train'])
            total_iter += num_iter

            for batch_idx, labeled_data in tqdm(enumerate(self.source_dataloader['train']), total=num_iter):
                feats, cls, _ = labeled_data
                if len(cls.shape) > 1:
                    cls = cls.flatten()
                feats, cls = feats.to(device), cls.to(device)

                preds = self.net(feats)
                with torch.no_grad():
                    pred_cls = preds.max(axis = 1).indices
                    num_correct_preds = (cls == pred_cls).sum()
                    total_num_correct += num_correct_preds
                class_loss = self.class_criterion(preds, cls)
                class_loss_sum += float(class_loss * feats.size(0))

                self.optimizer.zero_grad()
                class_loss.backward()
                self.optimizer.step()

            if self.source_dataloader['valid'] != None:
                total_num_samples_val = len(self.source_dataloader['valid']) * conf.args.opt['batch_size']
                with torch.no_grad():
                    num_iter_val = len(self.source_dataloader['valid'])
                    total_iter_val += num_iter_val
                    for batch_idx, labeled_data in tqdm(enumerate(self.source_dataloader['valid']), total=num_iter_val):
                        feats, cls, _ = labeled_data
                        if len(cls.shape) > 1:
                            cls = cls.flatten()
                        feats, cls = feats.to(device), cls.to(device)

                        preds = self.net(feats)
                        pred_cls = preds.max(axis = 1).indices
                        num_correct_preds = (cls == pred_cls).sum()
                        total_num_correct_val += num_correct_preds
                        class_loss_val = self.class_criterion(preds, cls)
                        class_loss_sum_val += float(class_loss_val * feats.size(0))

        dict_wandb_log = {
                        'epoch' : epoch,
                        'train_loss': class_loss_sum/total_iter,
                        'val_loss': class_loss_sum_val/total_iter_val if total_iter_val != 0 else 0,
                        'train_acc' : total_num_correct/total_num_samples,
                        'val_acc' : total_num_correct_val/total_num_samples_val if total_num_samples_val != 0 else 0
                    }

        if conf.args.wandb:
            import wandb
            wandb.log(dict_wandb_log)

        self.log_loss_results('train', epoch=epoch, 
                              loss_avg=class_loss_sum / total_iter,
                              train_acc_avg=total_num_correct/total_num_samples,
                              val_loss_avg=class_loss_sum_val/total_iter_val if total_iter_val != 0 else 0,
                              val_acc_avg=total_num_correct_val/total_num_samples_val if total_num_samples_val != 0 else 0,
        )
        avg_loss = class_loss_sum / total_iter

        return avg_loss

    def logger(self, name, value, epoch, condition):
        """
        Log a named metric value to file.
        
        Args:
            name (str): Name of the metric to log
            value: Value to log (can be tensor or scalar)
            epoch (int): Current epoch/iteration
            condition (str): Logging condition identifier
        """
        if not hasattr(self, name + '_log'):
            exec(f'self.{name}_log = []')
            exec(f'self.{name}_file = open(self.write_path + name + ".txt", "w")')

        exec(f'self.{name}_log.append(value)')

        if isinstance(value, torch.Tensor):
            value = value.item()
        write_string = f'{epoch}\t{value}\n'
        exec(f'self.{name}_file.write(write_string)')

    def evaluation(self, epoch, condition):
        """
        Evaluate model performance on target domain data in batch mode.
        
        Args:
            epoch (int): Current epoch
            condition (str): Evaluation condition identifier
            
        Returns:
            tuple: (class_accuracy, loss, confusion_matrix)
        """
        self.net.eval()

        with torch.no_grad():
            inputs, cls, dls = self.target_train_set
            tgt_inputs = inputs.to(device)
            tgt_cls = cls.to(device)

            preds = self.net(tgt_inputs)

            labels = [i for i in range(len(conf.args.opt['classes']))]

            class_loss_of_test_data = self.class_criterion(preds, tgt_cls)
            y_pred = preds.max(1, keepdim=False)[1]
            class_cm_test_data = confusion_matrix(tgt_cls.cpu(), y_pred.cpu(), labels=labels)

        print('{:s}: [epoch : {:d}]\tLoss: {:.6f} \t'.format(
            condition, epoch, class_loss_of_test_data
        ))
        class_accuracy = 100.0 * np.sum(np.diagonal(class_cm_test_data)) / np.sum(class_cm_test_data)
        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, 'test', class_accuracy))

        self.logger('accuracy', class_accuracy, epoch, condition)
        self.logger('loss', class_loss_of_test_data, epoch, condition)

        return class_accuracy, class_loss_of_test_data, class_cm_test_data

    def evaluation_online(self, epoch, current_samples):
        """
        Evaluate model performance on online samples that arrive sequentially.
        
        Args:
            epoch (int): Current sample number/epoch
            current_samples: List containing [features, cl_labels, do_labels]
        """
        self.net.eval()

        with torch.no_grad():
            features, cl_labels, do_labels = current_samples
            feats, cls, dls = (torch.stack(features), torch.stack(cl_labels), torch.stack(do_labels))
            self.evaluation_online_body(epoch, current_samples, feats, cls, dls)

    def model_inference(self, feats, net=None, temp=1.0, get_embedding=False, eval_mode=True):
        """
        Perform model inference on input features.
        
        Args:
            feats: Input features tensor
            net: Neural network model (uses self.net if None)
            temp (float): Temperature for softmax scaling
            get_embedding (bool): Whether to return embeddings
            eval_mode (bool): Whether to set model in eval mode
            
        Returns:
            tuple: (predictions, confidence, entropy, energy, embeddings, softmax_probs, logits)
        """
        if net is None:
            net = self.net

        if eval_mode:
            net.eval()
        else:
            net.train()

        len_feats = len(feats)

        if len(feats) == 1:
            if not conf.args.use_learned_stats:
                feats = torch.concat([feats, feats])
                len_feats = 1
            else:
                net.eval()

        if get_embedding:
            y_logit, y_embedding = net[1](net[0](feats), get_embedding=True)
            y_embedding = y_embedding[:len_feats]
        else:
            y_logit = net(feats)
            y_embedding = None

        y_logit = y_logit[:len_feats]

        y_entropy: torch.Tensor = softmax_entropy(y_logit)
        y_pred_softmax: torch.Tensor = F.softmax(y_logit / temp, dim=1)
        y_conf: torch.Tensor = y_pred_softmax.max(1, keepdim=False)[0]
        y_energy: torch.Tensor = calc_energy(y_logit).cpu()
        y_pred: torch.Tensor = y_logit.max(1, keepdim=False)[1]

        return y_pred, y_conf, y_entropy, y_energy, y_embedding, y_pred_softmax, y_logit

    def evaluation_online_body(self, epoch, current_samples, feats, cls, dls):
        """
        Core evaluation logic for processing online samples and updating metrics.
        
        Args:
            epoch (int): Current sample number/epoch
            current_samples: Current batch of samples being processed
            feats: Feature tensors
            cls: Class labels
            dls: Domain labels
        """
        true_cls_list = self.json_eval['gt']
        pred_cls_list = self.json_eval['pred']
        accuracy_list = self.json_eval['accuracy']
        current_accuracy_list = self.json_eval['current_accuracy']

        cls = cls.to(torch.int32)
        feats, cls, dls = feats.to(device), cls.to(device), dls.to(device)

        # Inference
        y_pred, y_conf, y_entropy, y_energy, y_embeddings, y_logit, y_output = self.model_inference(feats)

        # Append values to lists
        current_true_cls_list = [int(c) for c in cls.tolist()]
        true_cls_list += current_true_cls_list
        current_pred_cls_list = [int(c) for c in y_pred.tolist()]
        pred_cls_list += current_pred_cls_list

        if len(true_cls_list) > 0:
            current_accuracy = sum(1 for gt, pred in zip(current_true_cls_list, current_pred_cls_list) if gt == pred) \
                               / float(len(current_true_cls_list)) * 100
            current_accuracy_list.append(current_accuracy)
            cumul_accuracy = sum(1 for gt, pred in zip(true_cls_list, pred_cls_list) if gt == pred) \
                             / float(len(true_cls_list)) * 100
            accuracy_list.append(cumul_accuracy)

            dict_wandb_log = {
                        'num_batch_adapt': self.num_batch_adapt,
                        'accuracy': cumul_accuracy,
                        'current_accuracy': current_accuracy,
                    }

            if conf.args.wandb:
                import wandb
                wandb.log(dict_wandb_log)

            self.occurred_class = [0 for i in range(conf.args.opt['num_class'])]

            progress_checkpoint = [int(i * (len(self.target_train_set[0]) / 100.0)) for i in range(1, 101)]
            for i in range(epoch + 1 - len(current_samples[0]), epoch + 1):
                if i in progress_checkpoint:
                    print(f'[Online Eval][NumSample:{i}][Epoch:{progress_checkpoint.index(i) + 1}][Accuracy:{cumul_accuracy}]')

        # Update JSON evaluation metrics
        self.json_eval['gt'] = true_cls_list
        self.json_eval['pred'] = pred_cls_list
        self.json_eval['accuracy'] = accuracy_list
        self.json_eval['current_accuracy'] = current_accuracy_list

    def dump_eval_online_result(self, is_train_offline=False):
        """
        Save online evaluation results to JSON file.
        
        Args:
            is_train_offline (bool): Whether to perform offline training evaluation
        """
        if is_train_offline:
            feats, cls, dls = self.target_train_set
            batchsize = conf.args.opt['batch_size']
            for num_sample in range(0, len(feats), batchsize):
                current_sample = feats[num_sample:num_sample + batchsize], cls[num_sample:num_sample + batchsize], dls[
                                                                                                                num_sample:num_sample + batchsize]
                self.evaluation_online(num_sample + batchsize,
                                    [list(current_sample[0]), list(current_sample[1]), list(current_sample[2])])

        json_file = open(self.write_path + 'online_eval.json', 'w')
        json = self.json_eval | self.json_active
        json_subsample = {key: json[key] for key in json.keys() - {'extracted_feat'}}
        json_file.write(to_json(json_subsample))
        json_file.close()

    def validation(self, epoch):
        """
        Validate the model performance.
        
        Args:
            epoch (int): Current epoch
            
        Returns:
            tuple: (class_accuracy, loss)
        """
        class_accuracy_of_test_data, loss, _ = self.evaluation(epoch, 'valid')
        return class_accuracy_of_test_data, loss

    def test(self, epoch):
        """
        Test the model performance.
        
        Args:
            epoch (int): Current epoch
            
        Returns:
            tuple: (class_accuracy, loss)
        """
        class_accuracy_of_test_data, loss, cm_class = self.evaluation(epoch, 'test')
        return class_accuracy_of_test_data, loss

    def add_instance_to_memory(self, current_sample, mem):
        """
        Add a sample instance to the specified memory buffer.
        
        Args:
            current_sample: Sample to add (features, class, domain)
            mem: Memory buffer to add the sample to
        """
        with torch.no_grad():
            self.net.eval()

            if isinstance(mem, FIFO):
                mem.add_instance(current_sample)
            else:
                f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)
                y_pred, y_conf, y_entropy, y_energy, y_embeddings, y_pred_softmax, _ = self.model_inference(
                    f.unsqueeze(0))

                if isinstance(mem, ActivePriorityFIFO):
                    mem.add_u_instance([f, c.item(), d, y_entropy.item()])
                else:
                    raise NotImplementedError

    def train_online(self, current_num_sample):
        """
        Perform online training on current sample.
        
        Args:
            current_num_sample (int): Current sample number being processed
            
        Returns:
            int: Training status (TRAINED, SKIPPED, or FINISHED)
        """
        if current_num_sample > len(self.target_train_set[0]):
            return FINISHED

        batch_data, cls, dls = self.target_train_set
        current_sample = batch_data[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]

        self.add_instance_to_memory(current_sample, self.fifo)
        self.add_instance_to_memory(current_sample, self.mem)
        if conf.args.enable_bitta:
            self.add_instance_to_memory(current_sample, self.active_mem)

        if current_num_sample % conf.args.update_every_x != 0:
            if not (current_num_sample == len(self.target_train_set[0]) and conf.args.update_every_x >= current_num_sample):
                    return SKIPPED
        
        self.evaluation_online(current_num_sample, self.fifo.get_memory())

        if conf.args.no_adapt:
            return TRAINED
        
        self.pre_active_sample_selection()
        
        if isinstance(self.mem, ActivePriorityFIFO):
            self.active_sample_selection(self.mem, current_num_sample)
        elif conf.args.enable_bitta:
            self.active_sample_selection(self.active_mem, current_num_sample)

        prev_wall_time = time.time()
        self.test_time_adaptation()
        curr_wall_time = time.time()

        if conf.args.wandb:
            import wandb
            wandb.log({
                'num_batch_adapt': self.num_batch_adapt,
                'wall_clock_time_per_batch': curr_wall_time - prev_wall_time
            })

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)
        self.num_batch_adapt += 1

        return TRAINED

    @abstractmethod
    def pre_active_sample_selection(self):
        """
        Perform any necessary preparation before active sample selection.
        This method will be implemented by subclasses.
        """
        pass

    def dropout_inference(self, x, n_iter, dropout, net=None, temperature=1.0):
        """
        Perform Monte Carlo dropout inference for uncertainty estimation.
        
        Args:
            x: Input tensor
            n_iter (int): Number of dropout iterations
            dropout (float): Dropout rate
            net: Neural network (uses self.net if None)
            temperature (float): Temperature for softmax scaling
            
        Returns:
            tuple: (predictions, mean_predictions, std_predictions)
        """
        if net is None:
            net = self.net
        net = self.net.module if isinstance(self.net, nn.DataParallel) or isinstance(self.net, nn.parallel.DistributedDataParallel) else net

        if dropout < 0:
            if conf.args.dataset == "pacs":
                dropout = 0.4
            elif conf.args.dataset == "tiny-imagenet":
                dropout = 0.3
            elif conf.args.dataset == "cifar10":
                dropout = 0.3
            else:
                raise NotImplementedError

        predictions = []
        for _ in range(n_iter):
            pred = net[1]((net[0](x)), dropout=dropout)
            pred = F.softmax(pred, dim=1) / temperature
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)
        mean_pred = torch.mean(predictions, dim=1)
        return mean_pred

    def _fit_platt(self):
        """C1: fit p_hat = sigmoid(a*logit(C) + b) on the observed (confidence, binary) pairs.

        The binary feedback is already paid for by the BFA loss; BiTTA never reuses it to improve
        selection. This is that free signal: feedback -> calibration -> better queries.

        Caveat that belongs in the write-up: pairs only ever come from QUERIED samples, so the
        buffer is a censored slice of the C distribution and the map extrapolates poorly outside
        it. --explore_eps is the mitigation.
        """
        n = len(self.platt_conf_buf)
        if n < conf.args.platt_min_samples:
            return

        c = torch.tensor(list(self.platt_conf_buf), dtype=torch.float32, device=device)
        z = torch.logit(c.clamp(1e-6, 1 - 1e-6))
        y = torch.tensor(list(self.platt_bin_buf), dtype=torch.float32, device=device)

        # Platt target smoothing: with only ~3 pairs/batch the buffer is often linearly
        # separable, which would drive a -> infinity.
        n_pos = float(y.sum().item())
        n_neg = float(n - n_pos)
        y = torch.where(y > 0.5,
                        torch.full_like(y, (n_pos + 1.0) / (n_pos + 2.0)),
                        torch.full_like(y, 1.0 / (n_neg + 2.0)))

        a = torch.tensor([self.platt_a], device=device, requires_grad=True)
        b = torch.tensor([self.platt_b], device=device, requires_grad=True)
        opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=50)
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce(a * z + b, y)
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except Exception:
            return  # keep the previous map rather than poisoning selection

        a_v, b_v = float(a.item()), float(b.item())
        if not (np.isfinite(a_v) and np.isfinite(b_v)):
            self.platt_n_rejected += 1
            return

        # Reject a non-positive slope instead of clamping it to 0. a == 0 makes p_hat constant,
        # so h_b(p_hat) is constant too and torch.topk then returns simply the first k indices --
        # selection would silently become arbitrary, i.e. worse than the mc_conf it replaced.
        # a < 0 is an inverted map (higher confidence -> lower P(correct)), which mis-ranks
        # systematically. In both cases the previous map is the safer thing to keep.
        if a_v <= self.PLATT_A_MIN:
            self.platt_n_rejected += 1
            return

        self.platt_a = float(np.clip(a_v, self.PLATT_A_MIN, 50.0))
        self.platt_b = b_v
        self.platt_fitted = True

    def _binary_entropy_score(self, dropout_confidences):
        """C1: h_b(p_hat) -- the expected information gain of one binary-feedback query.

        By the grouping rule of entropy, H(y|x) = h_b(p1) + (1-p1)*H(q). A query resolves to
        H=0 with prob p1 and to H(q) with prob 1-p1, so the expected reduction is exactly
        h_b(p1), maximized at p1 = 0.5 -- NOT at p1 -> 0 as BiTTA's Eq.6 assumes.
        """
        z = torch.logit(dropout_confidences.clamp(1e-6, 1 - 1e-6))
        p = torch.sigmoid(self.platt_a * z + self.platt_b).clamp(1e-6, 1 - 1e-6)
        return -(p * p.log() + (1 - p) * (1 - p).log())

    def _log_selection_diagnostics(self, selected_index, dropout_confidences, k):
        """Record what the selection rule actually did, so a null result can be diagnosed.

        A flat accuracy delta has two very different explanations that need opposite fixes:
        (a) the scoring rule keeps picking the SAME samples mc_conf would have picked, so C1 is a
            no-op -- visible as overlap_mc_conf ~ 1.0;
        (b) the rule picks different samples but the calibration is wrong -- visible as a healthy
            overlap below 1.0 together with an implausible platt_a / a high platt_n_rejected.
        Without these counters the two are indistinguishable from the accuracy column alone.

        Called BEFORE --explore_eps is applied, so it measures the scoring rule itself rather
        than the random perturbation layered on top of it.
        """
        mc_conf_pick = torch.topk(dropout_confidences, k, largest=False).indices
        overlap = len(set(selected_index.tolist()) & set(mc_conf_pick.tolist())) / max(k, 1)

        self.json_active['platt_a'] += [float(self.platt_a)]
        self.json_active['platt_b'] += [float(self.platt_b)]
        self.json_active['platt_fitted'] += [int(self.platt_fitted)]
        self.json_active['platt_n_pairs'] += [len(self.platt_conf_buf)]
        self.json_active['platt_n_rejected'] += [int(self.platt_n_rejected)]
        self.json_active['overlap_mc_conf'] += [float(overlap)]
        self.json_active['sel_mean_conf'] += [float(dropout_confidences[selected_index].mean())]

    def _epsilon_explore(self, selected_index, n_pool):
        """C1: with prob --explore_eps, spend a query slot on a uniformly random sample instead.

        Without this, we only ever observe feedback in the region we already choose to query, and
        the Platt map is fit on a censored sample of the confidence distribution.
        """
        sel = selected_index.tolist()
        sel_set = set(sel)
        for i in range(len(sel)):
            if np.random.random() >= conf.args.explore_eps:
                continue
            available = [j for j in range(n_pool) if j not in sel_set]
            if not available:
                break
            replacement = int(np.random.choice(available))
            sel_set.discard(sel[i])
            sel_set.add(replacement)
            sel[i] = replacement
        return torch.tensor(sel, dtype=selected_index.dtype, device=selected_index.device)

    @staticmethod
    def _has_spread(x, eps=1e-9):
        """True if a score vector actually discriminates between candidates.

        torch.topk on a constant vector silently returns the first k indices, so a collapsed
        score would look like a working selection rule while behaving arbitrarily. Callers use
        this to fall back to mc_conf instead.
        """
        return float(x.max() - x.min()) > eps

    @staticmethod
    def _greedy_diverse_topk(embeddings, k):
        """
        Hybrid Uncertainty - Diversity-aware selection (Greedy k-Center / Max-Min distance).
        embeddings[0] mặc định là mẫu uncertain nhất, luôn được chọn đầu tiên.

        Args:
            embeddings (torch.Tensor): (N, D), đã sắp xếp theo confidence tăng dần.
            k (int): số mẫu cần chọn (= n_active_sample).

        Returns:
            list[int]: local indices trong `embeddings` của các mẫu được chọn.
        """
        n_candidates = embeddings.size(0)
        if k >= n_candidates:
            return list(range(n_candidates))

        selected = [0]
        min_dist = torch.cdist(embeddings, embeddings[0:1]).squeeze(1)
        min_dist[0] = -float('inf')

        for _ in range(1, k):
            next_idx = int(torch.argmax(min_dist).item())
            selected.append(next_idx)
            new_dist = torch.cdist(embeddings, embeddings[next_idx:next_idx + 1]).squeeze(1)
            min_dist = torch.minimum(min_dist, new_dist)
            min_dist[next_idx] = -float('inf')

        return selected
    
    def active_sample_selection(self, mem, current_num_sample):
        """
        Select samples for active learning based on uncertainty measures.
        
        Args:
            mem: Memory buffer containing candidate samples
            current_num_sample (int): Current sample number
        """
        assert isinstance(mem, ActivePriorityFIFO)
    
        if conf.args.memory_size == 1:
            self.count_bs1 += 1
            self.count_bs1 %= 64
            
            if self.count_bs1 == 0:
                self.random_indexes_bs1 = np.arange(64)
                np.random.shuffle(self.random_indexes_bs1)
                self.random_indexes_bs1 = self.random_indexes_bs1[:3]

            if self.count_bs1 not in self.random_indexes_bs1:
                return
            
        selected_feats, selected_labels, selected_domains = [], [], []

        n_active_sample = conf.args.n_active_sample
        if conf.args.active_sample_per_n_step:
            if self.num_batch_adapt % conf.args.active_sample_per_n_step != 0:
                return

        self.net.train()

        if isinstance(mem, ActivePriorityFIFO):
            feats, gt_labels, domains, entropies = mem.get_u_memory()
        else:
            raise NotImplementedError

        # Stays None for `random`, which has no confidence signal to calibrate on.
        dropout_confidences = None

        if conf.args.sample_selection == "mc_conf":
            with torch.no_grad():
                dropout_softmax_mean = self.dropout_inference(torch.stack(feats), n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate, net=self.net)
                y_pred, y_conf, y_entropy, y_energy, _, _, _ = self.model_inference(torch.stack(feats), self.net)
                dropout_confidences = dropout_softmax_mean[:, y_pred].diagonal()
            selected_index = torch.topk(dropout_confidences, n_active_sample, largest=False).indices
        elif conf.args.sample_selection == "random":
            selected_index = torch.randperm(len(gt_labels))[:n_active_sample]
        elif conf.args.sample_selection == "mc_conf_diverse":
            # === Hybrid Uncertainty: MC-dropout uncertainty + Diversity-aware selection ===
            with torch.no_grad():
                dropout_softmax_mean = self.dropout_inference(torch.stack(feats), n_iter=conf.args.n_dropouts, dropout=conf.args.dropout_rate, net=self.net)
                y_pred, y_conf, y_entropy, y_energy, y_embeddings, _, _ = self.model_inference(torch.stack(feats), self.net, get_embedding=True)
                dropout_confidences = dropout_softmax_mean[:, y_pred].diagonal()

            n_candidates = min(max(conf.args.diversity_topn, n_active_sample), len(dropout_confidences))
            candidate_index = torch.topk(dropout_confidences, n_candidates, largest=False).indices

            cand_conf = dropout_confidences[candidate_index]
            order = torch.argsort(cand_conf)          # sắp xếp uncertain nhất -> ít uncertain hơn
            candidate_index = candidate_index[order]

            candidate_embeddings = y_embeddings[candidate_index].reshape(len(candidate_index), -1).float()
            local_selected = self._greedy_diverse_topk(candidate_embeddings, n_active_sample)
            selected_index = candidate_index[local_selected]
        elif conf.args.sample_selection == "info_max":
            # === C1: information-maximizing selection ===
            # Query the samples whose binary answer carries the most information about the label,
            # i.e. argmax h_b(P(correct|x)), instead of the least-confident ones (BiTTA Eq.6).
            n_it = conf.args.n_dropouts if conf.args.n_dropouts_select < 0 else conf.args.n_dropouts_select
            with torch.no_grad():
                dropout_softmax_mean = self.dropout_inference(torch.stack(feats), n_iter=n_it, dropout=conf.args.dropout_rate, net=self.net)
                y_pred, y_conf, y_entropy, y_energy, _, _, _ = self.model_inference(torch.stack(feats), self.net)
                dropout_confidences = dropout_softmax_mean.gather(1, y_pred.view(-1, 1)).squeeze(1)

            if not self.platt_fitted:
                # Cold start MUST fall back to mc_conf ranking, not to an identity Platt map:
                # with a=1,b=0 we would score h_b(C), which peaks at C=0.5 and therefore ranks
                # the most uncertain samples LOWEST -- inverting selection exactly where it matters.
                selected_index = torch.topk(dropout_confidences, n_active_sample, largest=False).indices
            else:
                scores = self._binary_entropy_score(dropout_confidences)
                if not self._has_spread(scores):
                    # Degenerate map -> every score equal -> topk would just return the first k
                    # indices, i.e. an arbitrary pick. mc_conf is strictly better than arbitrary.
                    selected_index = torch.topk(dropout_confidences, n_active_sample, largest=False).indices
                else:
                    selected_index = torch.topk(scores, n_active_sample, largest=True).indices

            self._log_selection_diagnostics(selected_index, dropout_confidences, n_active_sample)

            if conf.args.explore_eps > 0:
                # Keep the calibration buffer from collapsing onto the region we already query.
                selected_index = self._epsilon_explore(selected_index, len(dropout_confidences))
        else:
            raise NotImplementedError

        self.budget += n_active_sample
        self.json_active['budgets'] += [self.budget]

        if conf.args.wandb:
            import wandb
            wandb.log(
                {
                    'num_batch_adapt': self.num_batch_adapt,
                    'budget': self.budget,
                }
            )
                
        # add active samples to memory
        if n_active_sample <= 0:
            return

        selected_index = selected_index.sort(descending=True).values
        # Gathered inside this loop, not before it: the sort above reorders selected_index, so a
        # separately-gathered confidence vector would misalign with selected_feats.
        selected_confs = []
        for idx in selected_index:
            selected_feats += [feats[idx]]
            selected_labels += [gt_labels[idx]]
            selected_domains += [domains[idx]]
            if dropout_confidences is not None:
                selected_confs += [dropout_confidences[idx].item()]

            mem.remove_u_instance_by_index(idx)  # remove index from candidate pool

        if conf.args.active_full_label:  # full active TTA
            for correct_data_i in range(len(selected_feats)):
                data = [selected_feats[correct_data_i],
                        selected_labels[correct_data_i],
                        selected_domains[correct_data_i],
                        0.0]
                mem.add_correct_instance(data)
            return

        # binary TTA
        self.net.eval()
        selected_feats_ = torch.stack(selected_feats).to(device)
        selected_labels_ = torch.Tensor(selected_labels).to(device)

        with torch.no_grad():
            y_logit = self.net(selected_feats_)
            y_entropy = softmax_entropy(y_logit)
            y_pred = y_logit.max(1, keepdim=False)[1]

        mask_correct: torch.Tensor = y_pred == selected_labels_

        for match_i in range(len(mask_correct)):
            label = selected_labels_[match_i].item()
            
            if conf.args.feedback_error_rate > 0 and np.random.random() < conf.args.feedback_error_rate:
                mask_correct[match_i] = ~mask_correct[match_i]

            # C1: record the pair AFTER the noise flip -- a deployed calibrator only ever sees
            # the oracle's (possibly wrong) answer, never the clean one.
            if selected_confs:
                self.platt_conf_buf.append(selected_confs[match_i])
                self.platt_bin_buf.append(1 if bool(mask_correct[match_i]) else 0)

            data = [selected_feats[match_i],
                    y_pred[match_i].item(),
                    label,
                    y_entropy[match_i].item()]

            if mask_correct[match_i]:  # correct
                mem.add_correct_instance(data)
                self.num_correct += 1
            else:  # wrong
                mem.add_wrong_instance(data)
                self.num_wrong += 1

        # C1: refit the calibration map on the accumulated feedback.
        if conf.args.sample_selection == "info_max":
            self.platt_n_calls += 1
            if self.platt_n_calls % conf.args.platt_refit_every == 0:
                self._fit_platt()

    def disable_running_stats(self):
        """
        Disable running statistics tracking in BatchNorm layers.
        """
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = 0

    def enable_running_stats(self):
        """
        Enable running statistics tracking in BatchNorm layers.
        """
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum

    def set_seed(self):
        """
        Set random seeds for reproducibility.
        """
        torch.manual_seed(conf.args.seed)
        np.random.seed(conf.args.seed)
        random.seed(conf.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def get_bitta_ssl_loss(self):
        """
        Compute **TTA with binary feedback** baseline loss functions using correct and wrong samples.
        (This is NOT a BiTTA's algorithm)
        
        Returns:
            torch.Tensor: Combined loss from correct and wrong predictions
        """
        assert conf.args.enable_bitta
        self.disable_running_stats()

        loss = 0.0

        correct_feats, correct_preds, _, _ = self.active_mem.get_correct_memory()
        wrong_feats, wrong_preds, wrong_gt_labels, _ = self.active_mem.get_wrong_memory()

        if correct_preds is not None and len(correct_preds) > 0:
            if len(correct_preds) == 1:
                if not conf.args.use_learned_stats:
                    correct_feats = correct_feats + correct_feats
                else:
                    self.net.eval()

            correct_logits = self.net(torch.stack(correct_feats).to(device)).softmax(1)[:len(correct_preds)]
            correct_preds = torch.tensor(correct_preds).to(device)
            correct_loss = F.cross_entropy(correct_logits, correct_preds.detach())
            loss += correct_loss

        if wrong_preds is not None and len(wrong_preds) > 0:
            if len(wrong_preds) == 1:
                if not conf.args.use_learned_stats:
                    wrong_feats = wrong_feats + wrong_feats
                else:
                    self.net.eval()

            wrong_logits = self.net(torch.stack(wrong_feats).to(device)).softmax(1)[:len(wrong_preds)]
            wrong_preds = torch.tensor(wrong_preds).to(device)
            wrong_loss = complement_CrossEntropyLoss(wrong_logits, wrong_preds.detach())
            loss += wrong_loss

        self.enable_running_stats()
        return loss
