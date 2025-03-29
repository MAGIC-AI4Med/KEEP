import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import json

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

def mask_contrastive(logits, labels, device):
    batch_size = logits.shape[0]
    mask =  np.tile(np.array(labels+1),(batch_size,1))
    mask = mask - mask.T
    mask = mask + np.eye(batch_size)
    mask[mask!=0] = 1.
    mask = torch.tensor(mask,dtype=logits.dtype).to(device)
    
    logits_sum = torch.exp(logits).mul(mask).sum(1)
    logits_norm = torch.exp(torch.diag(logits))/logits_sum
    loss = -1*torch.log(logits_norm)
    loss = loss.mean()
    return loss
    

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, loss_weight = None, input_labels = None, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        
        if input_labels is None:
            labels = self.get_ground_truth(device, logits_per_image.shape[0])
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2
        else:
            labels = np.array(input_labels)
            total_loss = (
                mask_contrastive(logits_per_image, labels, device) +
                mask_contrastive(logits_per_text, labels, device)
            ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class HyMetricLoss(object):
    """
    SP loss using HARD example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, temp=0.04, loss_type = 'lhp-hn', caption_num = 32, knowledge_root = None):
        self.temp = temp
        self.loss_type = loss_type
        self.caption_num = caption_num
        
        if knowledge_root:
            with open(knowledge_root) as f:
                all_do_nodes = json.load(f)
            self.node_graph = dict()
            for kk,vv in all_do_nodes.items():
                self.node_graph[kk] = vv['parent']
    
    def metric_loss(self, sim_mat, scale, N_id, N_ins, device):
        
        sf_sim_scale = sim_mat*scale
        
        rows,cols = sim_mat.shape[0],sim_mat.shape[1]
        sf_sim_qq = sf_sim_scale[:rows,:rows]
        if rows != cols:
            sf_sim_neg = sf_sim_scale[:rows,rows:]

        right_factor = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).to(device)
        pos_mask = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).to(device)
        left_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((1,N_ins)))).to(device)
        
        ## hard-hard mining for pos and neg
        mask_HH = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).to(device)
        mask_HH[mask_HH==0]=1.

        ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))
        ID_sim_HH = ID_sim_HH.mm(right_factor)
        ID_sim_HH = left_factor.mm(ID_sim_HH)

        pos_mask_id = torch.eye(N_id).to(device)
        pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
        pos_sim_HH[pos_sim_HH==0]=1.
        pos_sim_HH = 1./pos_sim_HH
        ID_sim_HH = ID_sim_HH.mul(1-pos_mask_id) + pos_sim_HH.mul(pos_mask_id)
        
        ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH,p = 1, dim = 1)   
        
        ################################
        
        ## hard-easy mining for negs, hard-hard for pos
        mask_HE_neg = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).to(device)
        mask_HE_neg[mask_HE_neg==0]=1.

        ID_sim_HH_neg = torch.exp(sf_sim_qq.mul(mask_HE_neg))
        ID_sim_HH_neg = ID_sim_HH_neg.mm(right_factor)
        
        neg_sim_HH = ID_sim_HH_neg.mul(1-pos_mask)
        neg_sim_HH[neg_sim_HH==0]=1.
        neg_sim_HH = 1./neg_sim_HH
        ID_sim_HH_neg = neg_sim_HH.mul(1-pos_mask) + ID_sim_HH_neg.mul(pos_mask)
        
        ID_sim_HH_neg = left_factor.mm(ID_sim_HH_neg)
        # pos_mask_id = torch.eye(N_id).to(device)
        ID_sim_HH_neg = 1./ID_sim_HH_neg
        
        ID_sim_HH_neg_L1 = nn.functional.normalize(ID_sim_HH_neg,p = 1, dim = 1) 
        
        #########################################
        
        ## hard-easy mining for pos, hard-hard for neg
        mask_HE = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).to(device)
        mask_HE[mask_HE==0]=1.

        ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))
        ID_sim_HE = ID_sim_HE.mm(right_factor)

        pos_sim_HE = ID_sim_HE.mul(pos_mask)
        pos_sim_HE[pos_sim_HE==0]=1.
        pos_sim_HE = 1./pos_sim_HE
        ID_sim_HE = ID_sim_HE.mul(1-pos_mask) + pos_sim_HE.mul(pos_mask)

        # hard-hard for neg
        ID_sim_HE = left_factor.mm(ID_sim_HE)

        ## additional neg
        if rows != cols:
            mask_additional_neg = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,N_ins)))).to(device)
            additional_neg = torch.exp(sf_sim_neg.mul(mask_additional_neg))
            temp = additional_neg.mm(right_factor)
            temp = left_factor.mm(temp)
            add_neg = torch.diag(temp).unsqueeze(1)
            ID_sim_HE = torch.cat((ID_sim_HE,add_neg),dim=1)
        
        
        ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE,p = 1, dim = 1)
            
        ###################################
        
        ## hard-easy mining for pos and hard
        mask_HE_pos_neg = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).to(device)
        mask_HE_pos_neg[mask_HE_pos_neg==0]=1.

        ID_sim_HE_pos_neg = torch.exp(sf_sim_qq.mul(mask_HE_pos_neg))
        ID_sim_HE_pos_neg = ID_sim_HE_pos_neg.mm(right_factor)

        # pos_sim_HE = ID_sim_HE.mul(pos_mask)
        # pos_sim_HE[pos_sim_HE==0]=1.
        ID_sim_HE_pos_neg = 1./ID_sim_HE_pos_neg
        # ID_sim_HE = ID_sim_HE.mul(1-pos_mask) + pos_sim_HE.mul(pos_mask)

        ID_sim_HE_pos_neg = left_factor.mm(ID_sim_HE_pos_neg)
        pos_mask_id = torch.eye(N_id).to(device)
        ID_sim_HE_pos_neg = (1./ID_sim_HE_pos_neg).mul(1-pos_mask_id) + ID_sim_HE_pos_neg.mul(pos_mask_id)

        ID_sim_HE_pos_neg_L1 = nn.functional.normalize(ID_sim_HE_pos_neg,p = 1, dim = 1)    
        
        #######################################

        loss_HH = -1*torch.log(torch.diag(ID_sim_HH_L1)).mean()
        loss_HE = -1*torch.log(torch.diag(ID_sim_HE_L1)).mean()
        # loss_adaptive = -1*torch.log(torch.diag(adaptive_sim_mat_L1)).mean()
        
        loss_HE_neg = -1*torch.log(torch.diag(ID_sim_HH_neg_L1)).mean()
        loss_HE_pos_neg = -1*torch.log(torch.diag(ID_sim_HE_pos_neg_L1)).mean()
        
        if self.loss_type == 'hp-hn':
            loss = loss_HH.mean()
        elif self.loss_type == 'lhp-hn':
            loss = loss_HE.mean()
        # elif self.loss_type == 'adasp':
        #     loss = loss_adaptive
        elif self.loss_type == 'hp-lhn':
            loss = loss_HE_neg
        elif self.loss_type == 'lhp-lhn':
            loss = loss_HE_pos_neg

        return loss
    
    def node_reachable(self, graph, start, end):
        if end not in graph or start not in graph:
            return False
        if start ==end:
            return True
        cur_nodes = [end]
        while True:
            connect_nodes = []
            for node in cur_nodes:
                connect_nodes.extend(graph[node])
            if len(connect_nodes) == 0:
                break
            if start in connect_nodes:
                return True
            cur_nodes = connect_nodes
            
        return False

    def __call__(self, image_features, text_features, cap_labels, logit_scale, output_dict=False):
        
        img_feat_norm = nn.functional.normalize(image_features, dim=1)
        txt_feat_norm = nn.functional.normalize(text_features, dim=1)
        
        bs_size = image_features.size(0)
        # N_id = len(torch.unique(targets))
        N_id = self.caption_num
        N_ins = bs_size // self.caption_num
        device = image_features.device
        
        imgtext_sim_mat = torch.matmul(img_feat_norm, txt_feat_norm.T)
        
        unique_label = cap_labels[::N_ins]
        node_connection = torch.ones(len(unique_label),len(unique_label), device = device)
        for i in range(len(unique_label)):
            for j in range(len(unique_label)):
                if i==j:
                    continue
                if self.node_reachable(self.node_graph,unique_label[i],unique_label[j]) or self.node_reachable(self.node_graph,unique_label[j], unique_label[i]):
                    node_connection[i,j] = -1
        
        node_mask = torch.kron(node_connection, torch.ones(N_ins,N_ins,device = device))
        
        
        ## additional neg
        if imgtext_sim_mat.shape[0] != imgtext_sim_mat.shape[1]:
            unknown_node = torch.ones(len(unique_label),len(unique_label), device = device)
            for i in range(len(unique_label)):
                for j in range(len(unique_label)):
                    if i!=j:
                        unknown_node[i,j] = -1
                    elif unique_label[i] == 'unknown':
                        unknown_node[i,j] = -1
            
            unknown_mask = torch.kron(unknown_node, torch.ones(N_ins,N_ins,device = device))
        
            node_mask = torch.cat((node_mask,unknown_mask),dim=1)
        
        imgtext_sim_mat[node_mask==-1] = -1
        
        it_loss = self.metric_loss(imgtext_sim_mat, logit_scale, N_id, N_ins, device)/2
        ti_loss = self.metric_loss(imgtext_sim_mat[0:bs_size,0:bs_size].T, logit_scale, N_id, N_ins, device)/2
        
        loss = it_loss + ti_loss
        
        return {"metric_loss": loss} if output_dict else loss
