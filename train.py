import time
import torch
import argparse
from data import dataset
from models.pix2pix import Pix2PixModel
def train(opt):
    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define the network
    model = Pix2PixModel(opt)
    model.setup(opt)

    _data = dataset.CustomDatasetDataLoader(opt)
    _dataset = _data.load_data()
    print('The number of training images = %d' % len(_dataset))

    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        model.update_learning_rate()
        for i, data in enumerate(_dataset):
            iter_data_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_data_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add network parameters
    parser.add_argument('--input_nc', type=int, default=3, help='The number of  input image channel: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3, help='The number of output image channel: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='The number of gen filters in the last conv layers')
    parser.add_argument('--ndf', type=int, default=64, help='The number of disc filters in the first conv layer')
    parser.add_argument('--netG', type=str, default='unet_512', help='Specify generator architecture [resnet_9blocks | unet_512 | resunet]')
    parser.add_argument('--netD', type=str, default='basic', help='Specify discriminator architecture [basic | n_layers | pixel]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='Only use if  netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance', help='Instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='Network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='Scaling factor for normal, xavier and orthogonal')
    parser.add_argument('--no_dropout', action='store_true', help='No dropout for generator')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu_ids: 0 0, 1,2, use -1 for CPU')

    # Add train parameters
    parser.add_argument('--dataroot', type=str, default='data/city_512', help='path to data')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--max_dataset_size', type=int, default=float('inf'), help='Max number of samples allowed per dataset')
    parser.add_argument('--no_flip', action='store_true', help='if specified , do not flip the images for data augmentation')
    parser.add_argument('--load_size', type=int, default=286, help='Scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop images to this size')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop] | none')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', type=int, default=4, help='the threads for loading data')
    parser.add_argument('--lr', type=float, default=0.0002, help='Optimizer learning rate')
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective [vanilla | lsgan | wgangp]')
    parser.add_argument('--phase', type=str, default='train', help='train, val,test, ...')
    parser.add_argument('--isTrain', action='store_true', help='train or not')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoint', help='path to save models')
    parser.add_argument('--name', type=str, default='pix2pix', help='name of the experiment')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy [linear | step | plateau | cosine]')
    parser.add_argument('--epoch_count', type=int, default=1, help='The starting epoch count (save model by <epoch_count> , <epoch_count> + <save_lastest_freq>)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')

    opt = parser.parse_args()

    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    train(opt)