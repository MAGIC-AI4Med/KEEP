
import logging
import copy
import torch

class FreezeScheduler():
    def __init__(self,):
        self.visual_freeze_flag = False
        self.text_freeze_flag = False
        self.knowledge_freeze_flag = False
    def update(self, cfg, model, epoch):
        # freeze or open image encoder
        if epoch < cfg.SOLVER.FREE_VISUAL_EPOCHS and not self.visual_freeze_flag:
            logging.info('Freeze the visual encoder')
            for name, param in model.visual.named_parameters():
                param.requires_grad = False
            self.visual_freeze_flag = True

        elif epoch >= cfg.SOLVER.FREE_VISUAL_EPOCHS and self.visual_freeze_flag:
            logging.info('Open the visual encoder')
            for name, param in model.visual.named_parameters():
                param.requires_grad = True
            self.visual_freeze_flag = False

        # freese text encoder
        if epoch < cfg.SOLVER.FREE_BERT_EPOCHS and not self.text_freeze_flag:
            if cfg.MODEL.KNOWLEDGE_BERT:    
                logging.info('Freeze the text encoder')
                for name, param in model.text.named_parameters():
                    param.requires_grad = False
            else:
                logging.info('Freeze the bert encoder')
                for name, param in model.text.named_parameters():
                    if 'mlp_embed' in name:
                        continue
                    param.requires_grad = False
            self.text_freeze_flag = True

        elif epoch >= cfg.SOLVER.FREE_BERT_EPOCHS and self.text_freeze_flag:
            logging.info('Open the text encoder')
            for name, param in model.text.named_parameters():
                param.requires_grad = True
            self.text_freeze_flag = False
        
        # freese knowledge encoder
        if epoch < cfg.SOLVER.FREE_KNOWLEDGE_EPOCHS and not self.knowledge_freeze_flag:
            logging.info('Freeze the knowledge encoder')
            for name, param in model.knowledge.named_parameters():
                param.requires_grad = False
            self.knowledge_freeze_flag = True

        elif epoch >= cfg.SOLVER.FREE_KNOWLEDGE_EPOCHS and self.knowledge_freeze_flag:
            logging.info('Open the knowledge encoder')
            for name, param in model.knowledge.named_parameters():
                param.requires_grad = True
            self.knowledge_freeze_flag = False


def freeze_scheduler(epoch, cfg, model):
    # freeze image encoder
    if epoch < cfg.SOLVER.FREE_VISUAL_EPOCHS:
        logging.info('Freeze the visual encoder')
        for name, param in model.visual.named_parameters():
            param.requires_grad = False
    elif epoch == cfg.SOLVER.FREE_VISUAL_EPOCHS:
        logging.info('Open the visual encoder')
        for name, param in model.visual.named_parameters():
            param.requires_grad = True

    # freese text encoder
    if epoch < cfg.SOLVER.FREE_BERT_EPOCHS:
        logging.info('Freeze the text encoder')
        for name, param in model.text.named_parameters():
            param.requires_grad = False
    elif epoch == cfg.SOLVER.FREE_BERT_EPOCHS:
        logging.info('Open the text encoder')
        for name, param in model.text.named_parameters():
            param.requires_grad = True

class FreezeChecker():
    def __init__(self, model):
        self.before = dict()
        for name, kid in model.named_children():
            self.before[name] = {}
            for subname in kid.state_dict():
                if 'weight' in subname:
                    self.before[name][subname] =  copy.deepcopy(kid.state_dict()[subname])
                    break

    def freeze_checker(self, model):
        weight_diff = dict()
        for name, kid in model.named_children():
            if name in self.before:
                weight_diff[name] = 0.
                for subname in kid.state_dict():
                    if subname in self.before[name]:
                        weight_diff[name] =  torch.norm(self.before[name][subname] - kid.state_dict()[subname],p=2)
                        if weight_diff[name] > 0:
                            logging.info(f'{name} module is open.')
                        else:
                            logging.info(f'{name} module is freezed.')
                        break

