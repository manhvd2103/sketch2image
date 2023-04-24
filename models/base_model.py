import torch
import os
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks



class BaseModel(ABC):
    '''This class is an abstract base class (ABC) for model'''

    def __init__(self, opt):
        '''Initialize the BaseModel class
        
        Parameters:
            opt (Option class): stores all the experiment flags
        '''
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.preprocess != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        '''Add new model-specific options and rewrite default values for existing options
        
        Parameters:
            parser : original option parser
            is_train (bool): whether training phase or test phase
        
        Return:
            the modified parser
        '''
        return parser
    
    @abstractmethod
    def set_input(self, input):
        '''Unpack input data from the dataloader and perform necessary pre-process steps
        
        Parameters:
            input (dict): includes the data and its metadata information
        '''
        pass

    @abstractmethod
    def forward(self):
        '''Run forward pass'''
        pass

    @abstractmethod
    def optimize_parameters(self):
        '''Calculate losses, gradients and update network weights (call in every training iteration)'''
        pass

    def setup(self, opt):
        '''Load and print networks, create scheduler
        
        Parameters:
            opt (Option class): store all the experiment flags
        '''
        if self.isTrain:
            self.schedulers = [networks.get_schedulers(optimizer, opt) for optimizer in self.optimizers]
        self.print_networks()
    
    def eval(self):
        '''Make models eval mode during test time'''
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
    
    def test(self):
        '''Forward function used in test time'''
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        '''Calculate additional output images'''
        pass

    def get_image_paths(self):
        '''Return image paths that are used to load current data'''
        return self.image_paths

    def update_learning_rate(self):
        '''Update learinig rates for all the networks (call at the end of every epoch)'''
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('Learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        '''Return visualization images'''
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        '''Return training losses / errors'''
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        '''Save all the networks to the disk
        
        Parameters:
            epoch (int): current epoch
        '''
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
    
    def _path_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        '''Fix InstanceNorm checkpoints incompatibility'''
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startwith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self._path_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
    
    def print_networks(self):
        '''Print the total number of parameters in the network and network architecture'''
        print('------------ Networks initialized ------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print('[Network %s] Total number of parameters: %.3f M' % (name, num_params / 1e6))
        print('----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        '''Set requires_grad=False for all the networks
        
        Parameters:
            nets (network list): a list of network
            requires_grad (bool): whether the networks require gradients or not
        '''
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad