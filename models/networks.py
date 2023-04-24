import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import functools


def get_schedulers(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    elif opt.lr_policy == 'cosin':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('Learning reate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(norm_type='batch'):
    '''Return a normalization layer
    
    Parameters:
        norm_type(str): the name of the normalization layer: batch | instance | none

    '''
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return nn.Identity()
    else:
        raise NotImplementedError('Normalization [%s] is not found' % norm_type)
    
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    '''Initialize network weights
    
    Parameters:
        net(network): the network to initialized
        init_type(str): the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain(float): scaling factor for normal, xavier and orthogonal
    '''
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    print('Initialize network with %s' % init_type)
    return net

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    '''Initialize a network:
    1. Register CPU/GPU device
    2. Initialize the network wights
    
    Parameters:
        net(network): the network to be initialized
        init_type(str): the name of an initialization method: normal | xavier |kaiming | orthogonal
        init_gain(float): scaling factor for normal, xavier and orthogonal
        gpu_ids(list): which GPUs the network runs on
    '''
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return init_weights(net, init_type, init_gain)


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    '''Create a generator network
    
    Parameters:
        input_nc(int): the number of channels in input images
        output_nc(int): the number of channels in output images
        ngf(int): the number of filters in the last conv layer
        netG(str): the architecture's name: resnet_9blocks | unet_256 | resunet
        norm(str): the name of the normalization layers used in the network: batch | instance | none
        use_dropout(bool): if use dropout layer
        init_type(str): the name of the initialization method
        init_gain(float): scaling factor for normal, xavier and orthogonal method
        gpu_ids(list): which GPUs the network runs on
    
    Returns a generator network
    '''

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            n_blocks=9,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
        )
    
    elif netG == 'unet_512':
        net = UnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            num_downs=9,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout
        )
    
    elif netG == 'resunet':
        net = ResunetGenerator(
            channel=input_nc
        )
    
    else:
        raise NotImplementedError('Generator model name [%s] is not implemented' % netG)
    
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    '''Create a discriminator
    
    Parameters:
        input_nc (int): the number of channels in input images
        ndf (int): the number of filters in the first conv layer
        netD (str): the architecture's name: basic | n_layers | pixel
        n_layers_D (int): the number of conv layers in the discriminator (effective when netD=='n_layers')
        norm (str): the type of normalization layers used in the network
        init_gain (float): scaling factor for normal, xavier, orthogonal
        gpu_ids (list): which GPUs the network runs on
    
    Returns a discriminator network
    '''

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=3,
            norm_layer=norm_layer
        )
    elif netD == 'n_layers':
        net = NLayerDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers_D,
            norm_layer=norm_layer
        )
    elif netD == 'pixel':
        net = PixelDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            norm_layer=norm_layer
        )
    else:
        raise NotImplementedError('Discriminator model name [%s] is not implemented' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_lable=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_lable))
        self.gan_model = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('Gan mode %s not implemented' % gan_mode)
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_model in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
        elif self.gan_model == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
    

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    if lambda_gp > 0.0:
        if type == 'real':
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data[0].contiguous().view(*real_data.shape))
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv, grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
        return gradient_penalty, gradients
    else:
        return 0.0, None



class ResnetBlock(nn.Module):
    '''Define a Resnet block'''
    
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        '''Initialize the Resnet block'''
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        '''Construct a convolutional block.
        
        Parameters:
            dim(int): the number of channels in the conv layer
            padding_type(str): the name of padding layer: reflect | replicate | zero
            norm_layer: normalization layer
            use_dropout(bool): if use dropout layer
            use_bias(bool): if the conv layer use bias or not
        
        Returns a block(with a conv layer, a normalization layer and a non-linearity layer (ReLU))
        '''
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('Padding [%s] is not implemented' % padding_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim), 
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('Padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim)]
        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    '''Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations'''

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        '''Construct a Resnet-based generator
        
        Parameters:
            input_nc (int): the number of channels in input images
            output_nc (int): the number of channels in output images
            ngf (int): number of filters in the last conv layer
            norm_layer: the normalization layer 
            use_dropout (bool): if use dropout layer
            n_blocks (int): number of Resnet blocks
            padding_type (str): the name of padding layer in conv layer: reflect | replicate | zero
        '''
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func ==nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias, use_dropout=use_dropout)]
        
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    

class UnetSkipConnectionBlock(nn.Module):
    '''Define the Unet submodule with skip connection'''

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        '''Construct a Unet submodule with skip connections.
        
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.outer_nc (int): the number of the 
        '''
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = norm_layer(inner_nc)

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down= [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    '''Create a Unet-base generator'''

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        '''Construct a Unet generator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        '''
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, norm_layer=norm_layer, outermost=True)

    def forward(self, x):
        return self.model(x)
    
class ResidualConv(nn.Module):
    '''Define the residual convolution block'''

    def __init__(self, input_nc, output_nc, stride, padding):
        '''Construct the residual convolution block
        
        Parameters:
            input_nc (int): number of channels in the input images
            output_nc (int): number of channels in the output images
            stride (int): stride value for Conv2d layer
            padding (int): padding value for Conv2d layer
        '''
        super(ResidualConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_nc),
            nn.ReLU(),
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_nc),
            nn.ReLU(),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1)
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_nc)
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    '''Define upsample block'''

    def __init__(self, input_nc, output_nc, kernel_size, stride):
        '''Construct upsample block
        
        Parameters:
            input_nc (int): input of channels in the input images
            output_nc (int): output of channels in the output images
            kernel_size (int): value of kernel
            stride (int): value of stride
        '''
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride)
    
    def forward(self, x):
        return self.upsample(x)



class ResunetGenerator(nn.Module):
    '''Define the ResUnet-based generator'''

    def __init__(self, channel, filters=[64, 128, 256, 512, 1024]):
        '''Construct ResUnet-based generator
        
        Parameters:
            channel (int): number of channels in input images
            fillters (int): number of hidden filters
        '''
        super(ResunetGenerator, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        )

        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.upsample_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.upsample_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.upsample_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.upsample_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):

        # Encoder
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)

        # Bridge
        x5 = self.bridge(x4)

        # Decoder
        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)

        x7 = self.upsample_residual_conv1(x6)

        x7 = self.upsample_2(x7)
        x8 = torch.cat([x7, x3], dim=1)

        x9 = self.upsample_residual_conv2(x8)

        x9 = self.upsample_3(x9)
        x10 = torch.cat([x9, x2], dim=1)

        x11 = self.upsample_residual_conv3(x10)

        x11 = self.upsample_4(x11)
        x12 = torch.cat([x11, x1], dim=1)

        x13 = self.upsample_residual_conv4(x12)
        output = self.output_layer(x13)

        return output
    

class NLayerDiscriminator(nn.Module):
    '''Defines a PatchGAN discriminator'''

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        '''Construct a PatchGAN discriminator
        
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        '''
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
    
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
    
class PixelDiscriminator(nn.Module):
    '''Defines a 1x1 PatchGAN discriminator (pixelGAN)'''

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        '''Construct a 1x1 PatchGAN discriminator
        
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        '''
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        return self.net(x)
    

if __name__ == '__main__':

    input = torch.rand(1, 3, 512, 512).cuda()
    # net = define_G(
    #     input_nc=3,
    #     output_nc=3,
    #     ngf=64,
    #     netG='unet_512',
    #     norm='instance'
    # ).cuda()

    net = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG='resnet_9blocks',
        norm='instance'
    ).cuda()

    # net = define_G(
    #     input_nc=3,
    #     output_nc=3,
    #     ngf=64,
    #     netG='resunet',
    # ).cuda()
    print(net(input).shape)
