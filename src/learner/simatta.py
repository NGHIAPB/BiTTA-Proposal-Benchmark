import copy

import torch.nn
from torch import optim
from torch.utils.data import TensorDataset

import conf
from utils.memory import FIFO
from .dnn import DNN
from utils.loss_functions import *
from utils.atta_dataloader import InfiniteDataLoader
from munch import Munch
from sklearn.metrics import pairwise_distances_argmin_min

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


def cross_entropy_with_logit(y_pred, y_true, **kwargs):
    """F.cross_entropy wrapper -- toan bo ham mat mat SimATTA can tu Metric
    goc (BiTTA/utils/atta_metric.py), cac ham con lai cua lop do (roc_auc,
    f1, mcc, ...) khong duoc SimATTA dung nen khong port vao day."""
    return F.cross_entropy(y_pred, y_true, **kwargs)


class SimATTA(DNN):
    """
    SimATTA (Gui et al., "Active Test-Time Adaptation: Theoretical Analyses
    and An Algorithm", ICLR 2024), chuyen sang phan hoi nhi phan.

    Ported tu BiTTA/learner/simatta_bin.py -- bien the ma chinh tac gia
    BiTTA dung lam baseline so sanh, da chuyen oracle cua SimATTA tu "tra
    ve nhan lop day du" sang "mot bit: du doan top-1 cua mo hinh dung hay
    sai" (co the them nhieu qua --feedback_error_rate), giong het giao
    thuc TENT*/SAR*/EATA*/BiTTA dang dung trong benchmark nay.

    Co che chinh moi batch: chon mau "target" (entropy cao, dang nghi ngo,
    duoc hoi oracle) va mau "source-like" (entropy thap, mo hinh nguon dong
    bang con tu tin) bang nguong entropy + incremental k-means clustering
    (sample_select); mau source-like duoc gan pseudo-label boi ban sao dong
    bang cua mo hinh truoc khi thich nghi de chong catastrophic forgetting;
    ca hai tap duoc huan luyen bang ham mat mat CE (mau dung)/complement-CE
    (mau sai), voi trong so theo ty le kich thuoc hai tap (cluster_train).
    """

    def __init__(self, *args, **kwargs):
        super(SimATTA, self).__init__(*args, **kwargs)
        assert (conf.args.memory_type == "FIFO")

    def init_learner(self):
        self.atta_variables = {}
        self.atta_variables['freeze_bn'] = True

        for param in self.net.parameters():  # turn on requires_grad for all
            param.requires_grad = True

        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        self.atta_variables['teacher'] = copy.deepcopy(self.net.to('cpu'))
        self.atta_variables['teacher'].eval()

        # change train mode: giu BN o eval() ngay ca khi goi self.net.train()
        self.net.original_train = self.net.train
        self.net.train = self.train_mode_freeze_bn
        self.freeze_bn()

        self.net.to(device)
        self.atta_variables['teacher'].to(device)
        self.update_teacher(0)  # copy student to teacher

        optimizer = optim.SGD(self.net.parameters(), lr=conf.args.opt['learning_rate'], momentum=0.9)

        self.atta_variables['budgets'] = 0
        self.atta_variables['anchors'] = None
        self.atta_variables['source_anchors'] = None
        self.atta_variables['n_clusters'] = 10
        self.atta_variables['nc_increase'] = conf.args.atta_inc_rate
        self.atta_variables['source_n_clusters'] = 100

        self.atta_variables['cold_start'] = conf.args.atta_cold_start

        self.atta_variables['accumulate_weight'] = True
        self.atta_variables['target_cluster'] = True
        self.atta_variables['LE'] = True  # Label-Efficient: cung chon mau "source-like" de chong CF

        # Nguong eh/el/steps/train_bs lay theo dataset cua paper goc (pacs/tiny-imagenet/
        # imagenet*); cac dataset khac cua benchmark nay (cifar10/cifar100/colored-mnist/
        # waterbirds) chua duoc tune nen dung mac dinh qua CLI (--atta_upper_th/--atta_lower_th/
        # --atta_batch_size) -- cung mo hinh "chua tune -> fallback CLI" ma EATA/BiTTA da dung
        # cho cac dataset chua co trong paper goc cua chung.
        if conf.args.dataset == "pacs":
            self.atta_variables['eh'] = 1e-2
            self.atta_variables['el'] = 1e-4
            self.atta_variables['steps'] = 30
            self.atta_variables['train_bs'] = 16
            self.atta_variables['gpu_clustering'] = False
        elif conf.args.dataset in ["tiny-imagenet", "imagenetR"]:
            self.atta_variables['eh'] = 1e-1
            self.atta_variables['el'] = 1e-1
            self.atta_variables['steps'] = 10
            self.atta_variables['train_bs'] = 32
            # gpu_clustering=True trong ban goc doi hoi utils/fast_pytorch_kmeans (khong
            # duoc port vao benchmark nay); giu False -> dung sklearn.cluster.KMeans tren CPU.
            self.atta_variables['gpu_clustering'] = False
        else:
            self.atta_variables['eh'] = conf.args.atta_upper_th
            self.atta_variables['el'] = conf.args.atta_lower_th
            self.atta_variables['steps'] = 30
            self.atta_variables['train_bs'] = conf.args.atta_batch_size
            self.atta_variables['gpu_clustering'] = False

        self.atta_variables['num_workers'] = 0
        self.atta_variables['stop_tol'] = 5
        self.atta_variables['loss_func'] = cross_entropy_with_logit

        self.atta_variables['count_budget'] = 0
        self.atta_variables['count_src_like'] = 0

        self.atta_variables['current_batch_budget'] = (
            self.atta_variables['n_clusters'] - conf.args.atta_limit_batch_active_sample
            if conf.args.atta_limit_batch_active_sample is not None else None
        )

        return optimizer

    def test_time_adaptation(self):
        assert isinstance(self.mem, FIFO)
        feats, labels, _ = self.mem.get_memory()

        feats = torch.stack(feats).to(device)
        labels = torch.Tensor(labels).type(torch.long).to(device)

        budget_limit = conf.args.atta_budget if conf.args.atta_budget != -1 else float('inf')

        if conf.args.memory_size != 1:
            if self.count_skip == conf.args.enable_skip:
                self.count_skip = 0

                if self.atta_variables['count_budget'] < budget_limit:  # normal active sample selection
                    outputs, closest, self.atta_variables['anchors'] = self.sample_select(
                        self.net, feats, labels,
                        self.atta_variables['anchors'],
                        int(self.atta_variables['n_clusters']),
                        1, ent_bound=self.atta_variables['eh'],
                        incremental_cluster=self.atta_variables['target_cluster'])
                else:
                    outputs, closest = [], []
            else:
                self.count_skip += 1
                closest = []

            if self.atta_variables['LE']:  # src-like data selection
                _, _, self.atta_variables['source_anchors'] = self.sample_select(
                    self.atta_variables['teacher'],
                    feats, labels, self.atta_variables['source_anchors'],
                    self.atta_variables['source_n_clusters'], 0,
                    use_pseudo_label=True, ent_bound=self.atta_variables['el'],
                    incremental_cluster=False)
            else:
                self.atta_variables['source_anchors'] = self.update_anchors(None, torch.tensor([]), None, None, None, None)

        else:
            self.count_bs1 += 1
            self.count_bs1 %= 64

            self.atta_variables['source_anchors'] = self.update_anchors(None, torch.tensor([]), None, None, None, None)

            if self.count_bs1 == 0:
                self.random_indexes_bs1 = np.arange(64)
                np.random.shuffle(self.random_indexes_bs1)
                self.random_indexes_bs1 = self.random_indexes_bs1[:3]

            if self.count_bs1 in self.random_indexes_bs1:
                with torch.no_grad():
                    outputs = self.net(feats)

                    pseudo_label = outputs.argmax(1).cpu().detach()
                    feats = feats.cpu().detach()
                    labels = labels.cpu().detach()
                    binary = pseudo_label == labels

                    self.atta_variables['anchors'] = self.update_anchors(None, feats, pseudo_label, None, binary, 0.0)

                self.atta_variables['count_budget'] += 1
            closest = []

        if not self.atta_variables['target_cluster']:
            self.atta_variables['n_clusters'] = 0
        self.atta_variables['source_n_clusters'] = 100

        self.atta_variables['budgets'] += len(closest)
        if self.count_skip == 0:
            self.atta_variables['n_clusters'] += self.atta_variables['nc_increase']
        self.atta_variables['source_n_clusters'] += 1

        if conf.args.memory_size != 1:
            if conf.args.atta_limit_mem_size is not None:
                self.atta_variables['anchors'].data = self.atta_variables['anchors'].data[-conf.args.atta_limit_mem_size:]
                self.atta_variables['anchors'].target = self.atta_variables['anchors'].target[-conf.args.atta_limit_mem_size:]
                self.atta_variables['anchors'].feats = self.atta_variables['anchors'].feats[-conf.args.atta_limit_mem_size:]
                self.atta_variables['anchors'].binary = self.atta_variables['anchors'].binary[-conf.args.atta_limit_mem_size:]
                self.atta_variables['anchors'].weight = self.atta_variables['anchors'].weight[-conf.args.atta_limit_mem_size:]

                self.atta_variables['source_anchors'].data = self.atta_variables['source_anchors'].data[-conf.args.atta_limit_mem_size:]
                self.atta_variables['source_anchors'].target = self.atta_variables['source_anchors'].target[-conf.args.atta_limit_mem_size:]
                self.atta_variables['source_anchors'].feats = self.atta_variables['source_anchors'].feats[-conf.args.atta_limit_mem_size:]
                self.atta_variables['source_anchors'].binary = self.atta_variables['source_anchors'].binary[-conf.args.atta_limit_mem_size:]
                self.atta_variables['source_anchors'].weight = self.atta_variables['source_anchors'].weight[-conf.args.atta_limit_mem_size:]

                assert self.atta_variables['anchors'].num_elem() <= conf.args.atta_limit_mem_size
                assert self.atta_variables['source_anchors'].num_elem() <= conf.args.atta_limit_mem_size

                if self.atta_variables['anchors'].num_elem() == conf.args.atta_limit_mem_size:
                    self.atta_variables['n_clusters'] = self.atta_variables['anchors'].num_elem() + self.atta_variables['nc_increase']

            print(self.atta_variables['anchors'].num_elem(), self.atta_variables['source_anchors'].num_elem(),
                  self.atta_variables['count_budget'], self.atta_variables['count_src_like'])

            if self.atta_variables['source_anchors'].num_elem() > 0:
                self.cluster_train(self.atta_variables['anchors'], self.atta_variables['source_anchors'], all_binary=False)
            else:
                self.cluster_train(self.atta_variables['anchors'], self.atta_variables['anchors'], all_binary=True)

            if self.atta_variables['anchors'].num_elem() < budget_limit:
                self.atta_variables['anchors'] = self.update_anchors_feats(self.atta_variables['anchors'])

        else:
            if self.atta_variables['anchors'] is not None:
                self.cluster_train(self.atta_variables['anchors'], self.atta_variables['anchors'], all_binary=True)

    def update_teacher(self, alpha_teacher):
        for t_param, s_param in zip(self.atta_variables['teacher'].parameters(), self.net.parameters()):
            t_param.data[:] = alpha_teacher * t_param[:].data[:] + (1 - alpha_teacher) * s_param[:].data[:]
        if not self.atta_variables['freeze_bn']:
            for tm, m in zip(self.atta_variables['teacher'].modules(), self.net.modules()):
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    tm.running_mean = alpha_teacher * tm.running_mean + (1 - alpha_teacher) * m.running_mean
                    tm.running_var = alpha_teacher * tm.running_var + (1 - alpha_teacher) * m.running_var

    @torch.enable_grad()
    def cluster_train(self, target_anchors, source_anchors, all_binary):
        self.net.train()

        source_binary = source_anchors.binary if all_binary else torch.ones(source_anchors.target.shape) == 1

        # weights=trong so deu (khong phai None): InfiniteDataLoader voi weights=None dung
        # RandomSampler(replacement=True), ma "mot vong" cua no dai dung len(dataset) mau, nen
        # BatchSampler(drop_last=False) co the sinh batch cuoi vong CHI 1 mau moi khi len(dataset)
        # % train_bs du ra 1 -- va vi InfiniteDataLoader lap vo han, batch=1 se dinh ky xuat hien
        # giua luc self.net.train(), lam BatchNorm bao loi "Expected more than 1 value per
        # channel". Truyen weights deu -> WeightedRandomSampler(num_samples=batch_size) luon tra
        # dung batch_size moi vong, khong bao gio con du.
        source_loader = InfiniteDataLoader(TensorDataset(source_anchors.data, source_anchors.target, source_binary),
                                            weights=torch.ones(source_anchors.num_elem()),
                                            batch_size=self.atta_variables['train_bs'],
                                            num_workers=self.atta_variables['num_workers'])

        target_loader = InfiniteDataLoader(TensorDataset(target_anchors.data, target_anchors.target, target_anchors.binary),
                                            weights=torch.ones(target_anchors.num_elem()),
                                            batch_size=self.atta_variables['train_bs'], num_workers=self.atta_variables['num_workers'])

        alpha = target_anchors.num_elem() / (target_anchors.num_elem() + source_anchors.num_elem())

        # neu src-like qua it -> gioi han dong gop cua no o 0.2
        if source_anchors.num_elem() < self.atta_variables['cold_start']:
            alpha = min(0.2, alpha)

        ST_loader = iter(zip(source_loader, target_loader))
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=conf.args.opt['learning_rate'], momentum=0.9)

        loss_window = []
        tol = 0
        lowest_loss = float('inf')
        for i, ((S_data, S_targets, S_binaries), (T_data, T_targets, T_binaries)) in enumerate(ST_loader):
            S_data, S_targets, S_binaries = S_data.to(device), S_targets.to(device), S_binaries.to(device)
            T_data, T_targets, T_binaries = T_data.to(device), T_targets.to(device), T_binaries.to(device)
            L_T = self.one_step_train(S_data, S_targets, S_binaries, T_data, T_targets, T_binaries, alpha, self.optimizer, all_binary)
            if len(loss_window) < self.atta_variables['stop_tol']:
                loss_window.append(L_T.item())
            else:
                mean_loss = np.mean(loss_window)
                tol += 1
                if mean_loss < lowest_loss:
                    lowest_loss = mean_loss
                    tol = 0
                if tol > 5:
                    break
                loss_window = []
            if i > self.atta_variables['steps']:
                break

    def one_step_train(self, S_data, S_targets, S_binaries, T_data, T_targets, T_binaries, alpha, optimizer, all_binary):
        S_out = self.net(S_data)
        T_out = self.net(T_data)

        if all_binary:
            L_S = self.atta_variables['loss_func'](S_out[S_binaries], S_targets[S_binaries], reduction="mean")
            L_S += complement_CrossEntropyLoss(S_out[~S_binaries], S_targets[~S_binaries])
        else:
            L_S = self.atta_variables['loss_func'](S_out, S_targets)

        L_T = self.atta_variables['loss_func'](T_out[T_binaries], T_targets[T_binaries], reduction="mean")
        L_T += complement_CrossEntropyLoss(T_out[~T_binaries], T_targets[~T_binaries])

        loss = (1 - alpha) * L_S + alpha * L_T
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return L_T

    def softmax_entropy(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        if y is None:
            if x.shape[1] == 1:
                x = torch.cat([x, -x], dim=1)
            return -(x.softmax(1) * x.log_softmax(1)).sum(1)
        else:
            return - 0.5 * (x.softmax(1) * y.log_softmax(1)).sum(1) - 0.5 * (y.softmax(1) * x.log_softmax(1)).sum(1)

    def update_anchors(self, anchors, data, target, feats, binary, weight):
        if anchors is None:
            anchors = Munch()
            anchors.data = data
            anchors.target = target
            anchors.feats = feats
            anchors.weight = weight
            anchors.binary = binary
            anchors.num_elem = lambda: len(anchors.data)
        else:
            anchors.data = torch.cat([anchors.data, data])
            anchors.target = torch.cat([anchors.target, target])
            anchors.feats = torch.cat([anchors.feats, feats])
            anchors.binary = torch.cat([anchors.binary, binary])
            anchors.weight = torch.cat([anchors.weight, weight])
        return anchors

    def update_anchors_feats(self, anchors):
        # Chia batch thu cong thay vi FastDataLoader(drop_last=False): khi
        # track_running_stats=False (cau hinh TTA kieu TENT dung xuyen suot
        # benchmark nay), BatchNorm LUON tinh thong ke theo batch hien tai du
        # da goi self.net.eval() (khong co running stats de fallback), nen mot
        # batch cuoi chi con dung 1 mau se lam torch.nn.functional.batch_norm
        # nem ValueError "Expected more than 1 value per channel". Neu phep
        # chia deu cho batch_size de lai phan du dung 1, gop phan du do vao
        # batch truoc thay vi tach rieng.
        n = len(anchors.data)
        batch_size = 32
        sizes = []
        remaining = n
        while remaining > 0:
            take = min(batch_size, remaining)
            if remaining - take == 1:
                take += 1
            sizes.append(take)
            remaining -= take

        anchors.feats = None
        self.net.eval()
        idx = 0
        with torch.no_grad():
            for s in sizes:
                data = anchors.data[idx: idx + s].to(device)
                idx += s
                _, feats = self.net[1](self.net[0](data), get_embedding=True)
                feats = feats.cpu().detach()
                anchors.feats = feats if anchors.feats is None else torch.cat([anchors.feats, feats])
        return anchors

    @torch.no_grad()
    def sample_select(self, model, data, target, anchors, n_clusters, ent_beta, use_pseudo_label=False, ent_bound=1e-2, incremental_cluster=False):
        model.eval()

        outputs, feats = model[1](model[0](data), get_embedding=True)

        pseudo_label = outputs.argmax(1).cpu().detach()
        data = data.cpu().detach()
        feats = feats.cpu().detach()
        target = target.cpu().detach()
        # Phan hoi nhi phan: oracle chi tra loi dung/sai cho du doan top-1 cua mo hinh,
        # khong tra ve nhan that -- giong het giao thuc cua TENT*/SAR*/EATA*/BiTTA.
        binary = pseudo_label == target

        for binary_i in range(len(binary)):
            if conf.args.feedback_error_rate > 0 and np.random.random() < conf.args.feedback_error_rate:
                binary[binary_i] = ~binary[binary_i]

        entropy = self.softmax_entropy(outputs).cpu()

        if not incremental_cluster:
            entropy = entropy.numpy()
            if ent_beta == 0:
                closest = np.argsort(entropy)[: n_clusters]
                closest = closest[entropy[closest] < ent_bound]
            elif ent_beta == 1:
                closest = np.argsort(entropy)[- n_clusters:]
                closest = closest[entropy[closest] >= ent_bound]
            else:
                raise NotImplementedError
            weights = torch.zeros(len(closest), dtype=torch.float)
        else:
            if ent_beta == 0:
                sample_choice = entropy < ent_bound
            elif ent_beta == 1:
                sample_choice = entropy >= ent_bound
            else:
                raise NotImplementedError

            # defensive programming: neu nguong entropy qua chat khien so ung vien
            # it hon n_clusters, lay top-k theo entropy thay vi loi
            current_num_samples = sample_choice.sum() + len(anchors.feats) if anchors else sample_choice.sum()
            if current_num_samples < n_clusters:
                target_num_samples_ = n_clusters - len(anchors.feats) if anchors else n_clusters
                sample_choice = entropy.topk(k=target_num_samples_, largest=True).indices

            data = data[sample_choice]
            target = target[sample_choice]
            feats = feats[sample_choice]
            binary = binary[sample_choice]
            pseudo_label = pseudo_label[sample_choice]

            if anchors:
                feats4cluster = torch.cat([anchors.feats, feats])
                sample_weight = torch.cat([anchors.weight, torch.ones(len(feats), dtype=torch.float)])
            else:
                feats4cluster = feats
                sample_weight = torch.ones(len(feats), dtype=torch.float)

            if self.atta_variables['gpu_clustering']:
                from utils.fast_pytorch_kmeans import KMeans
                from joblib import parallel_backend
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, device=device).fit(
                    feats4cluster.to(device),
                    sample_weight=sample_weight.to(device))
                with parallel_backend('threading', n_jobs=8):
                    raw_closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats4cluster)
                kmeans_labels_ = kmeans.labels_
            else:
                from joblib import parallel_backend
                from sklearn.cluster import KMeans
                with parallel_backend('threading', n_jobs=8):
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10, algorithm='elkan', random_state=conf.args.seed).fit(
                        feats4cluster, sample_weight=sample_weight)
                    raw_closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats4cluster)
                kmeans_labels_ = kmeans.labels_

            if anchors:
                num_anchors = anchors.num_elem()
                prev_anchor_cluster = torch.tensor(kmeans_labels_[:num_anchors], dtype=torch.long)

                if self.atta_variables['accumulate_weight']:
                    # tich luy trong so cua anchor cu neu chia se cung cluster voi mau moi
                    num_prev_anchors_per_cluster = prev_anchor_cluster.unique(return_counts=True)
                    num_prev_anchors_per_cluster_dict = torch.zeros(len(raw_closest), dtype=torch.long)
                    num_prev_anchors_per_cluster_dict[num_prev_anchors_per_cluster[0].long()] = num_prev_anchors_per_cluster[1]

                    num_newsample_per_cluster = torch.tensor(kmeans.labels_).unique(return_counts=True)
                    num_newsample_per_cluster_dict = torch.zeros(len(raw_closest), dtype=torch.long)
                    num_newsample_per_cluster_dict[num_newsample_per_cluster[0].long()] = num_newsample_per_cluster[1]
                    assert (num_prev_anchors_per_cluster_dict[prev_anchor_cluster] == 0).sum() == 0
                    anchors.weight = anchors.weight + num_newsample_per_cluster_dict[prev_anchor_cluster] / \
                        num_prev_anchors_per_cluster_dict[prev_anchor_cluster].float()

                anchored_cluster_mask = torch.zeros(len(raw_closest), dtype=torch.bool).index_fill_(
                    0, prev_anchor_cluster.unique().long(), True)
                new_cluster_mask = ~ anchored_cluster_mask

                closest = raw_closest[new_cluster_mask] - num_anchors
                if (closest < 0).sum() != 0:
                    # Mau gan centroid nhat cua mot cluster co the khong thuoc chinh cluster do -> loai
                    new_cluster_mask = torch.where(new_cluster_mask)[0]
                    new_cluster_mask = new_cluster_mask[closest >= 0]
                    closest = closest[closest >= 0]

                weights = torch.tensor(kmeans.labels_).unique(return_counts=True)[1][new_cluster_mask]
            else:
                num_anchors = 0
                closest = raw_closest
                weights = torch.tensor(kmeans.labels_).unique(return_counts=True)[1]

        if ent_beta == 1:
            if conf.args.atta_limit_batch_active_sample is not None:
                current_batch_budget = conf.args.atta_limit_batch_active_sample + self.atta_variables['current_batch_budget']
                if len(closest) < current_batch_budget:
                    self.atta_variables['current_batch_budget'] = current_batch_budget - len(closest)
                else:
                    self.atta_variables['current_batch_budget'] = 0
                perm_index = np.random.permutation(len(closest))
                closest = closest[perm_index[:current_batch_budget]]
                weights = weights[perm_index[:current_batch_budget]]

            self.atta_variables['count_budget'] += len(closest)
        elif ent_beta == 0:
            self.atta_variables['count_src_like'] += len(closest)

        anchors = self.update_anchors(anchors, data[closest], pseudo_label[closest], feats[closest], binary[closest], weights)

        return outputs, closest, anchors

    def enable_bn(self, model):
        if not self.atta_variables['freeze_bn']:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.momentum = 0.1

    def train_mode_freeze_bn(self, mode=True):
        """Ghi de train() mac dinh de giu nguyen cac lop BatchNorm o eval()."""
        self.net.original_train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
