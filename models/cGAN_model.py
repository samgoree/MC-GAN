################################################################################
# MC-GAN
# Glyph Network Model
# By Samaneh Azadi
################################################################################

import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from scipy import misc
import random



class cGANModel(BaseModel):
    def name(self):
        return 'cGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        if self.opt.conv3d:
            self.netG_3d = networks.define_G_3d(opt.input_nc, opt.input_nc, norm=opt.norm, groups=opt.grps, gpu_ids=self.gpu_ids)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, opt.use_dropout, gpu_ids=self.gpu_ids)
        
        disc_ch = opt.input_nc

        disc_ch_for_missing_data = opt.input_nc - opt.auxiliarymissingcharacters
        self.disc_ch_for_missing_data = disc_ch_for_missing_data
            
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.opt.conditional:
                if opt.which_model_preNet != 'none':
                    self.preNet_A = networks.define_preNet(
                        disc_ch+disc_ch, disc_ch+disc_ch, which_model_preNet=opt.which_model_preNet,
                        norm=opt.norm, gpu_ids=self.gpu_ids
                    )
                    self.preNet_A_missing_data = networks.define_preNet(
                        disc_ch_for_missing_data+disc_ch_for_missing_data, 
                        disc_ch_for_missing_data+disc_ch_for_missing_data,
                        which_model_preNet=opt.which_model_preNet,
                        norm=opt.norm, gpu_ids=self.gpu_ids
                    )
                nif = disc_ch+disc_ch
                nif_missing_data = disc_ch_for_missing_data + disc_ch_for_missing_data
                
                netD_norm = opt.norm

                self.netD = networks.define_D(nif, opt.ndf,
                                             opt.which_model_netD,
                                             opt.n_layers_D, netD_norm, use_sigmoid, gpu_ids=self.gpu_ids)
                self.netD_for_missing_data = networks.define_D(nif_missing_data, opt.ndf,
                                             opt.which_model_netD,
                                             opt.n_layers_D, netD_norm, use_sigmoid, gpu_ids=self.gpu_ids)

            else:
                self.netD = networks.define_D(disc_ch, opt.ndf,
                                             opt.which_model_netD,
                                             opt.n_layers_D, opt.norm, use_sigmoid, gpu_ids=self.gpu_ids)
                self.netD_for_missing_data = networks.define_D(disc_ch_for_missing_data, opt.ndf,
                                                               opt.which_model_netD,
                                                               opt.n_layers_D, opt.norm, use_sigmoid, gpu_ids=self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            if self.opt.conv3d:
                 self.load_network(self.netG_3d, 'G_3d', opt.which_epoch)
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                if opt.which_model_preNet != 'none':
                    self.load_network(self.preNet_A, 'PRE_A', opt.which_epoch)
                    self.load_network(self.preNet_A_missing_data, 'PRE_A_MISSING_DATA', opt.which_epoch)
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netD_for_missing_data, 'D_missing_data', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.fake_AB_pool_no_missing = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()


            # initialize optimizers
            if self.opt.conv3d:
                 self.optimizer_G_3d = torch.optim.Adam(self.netG_3d.parameters(),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.which_model_preNet != 'none':
                self.optimizer_preA = torch.optim.Adam(self.preNet_A.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_preA_missing_data = torch.optim.Adam(self.preNet_A_missing_data.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_missing_data = torch.optim.Adam(self.netD_for_missing_data.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            if self.opt.conv3d:
                networks.print_network(self.netG_3d)
            networks.print_network(self.netG)
            if opt.which_model_preNet != 'none':
                networks.print_network(self.preNet_A)
                networks.print_network(self.preNet_A_missing_data)
            networks.print_network(self.netD)
            networks.print_network(self.netD_for_missing_data)
            print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']
        if 'missing_data_mask' in input:
            self.missing_data_mask = input['missing_data_mask']

    def forward(self):
        self.real_A = Variable(self.input_A)
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
            self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))
        else:
            self.fake_B = self.netG.forward(self.real_A)

        
        self.real_B = Variable(self.input_B)
        real_B = util.tensor2im(self.real_B.data)
        real_A = util.tensor2im(self.real_A.data)
    
    def add_noise_disc(self,real):
        #add noise to the discriminator target labels
        #real: True/False? 
        if self.opt.noisy_disc:
            rand_lbl = random.random()
            if rand_lbl<0.6:
                label = (not real)
            else:
                label = (real)
        else:  
            label = (real)
        return label
            
                

    
    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
            self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))

        else:
            self.fake_B = self.netG.forward(self.real_A)
            
        self.real_B = Variable(self.input_B, volatile=True)

    #get image paths
    def get_image_paths(self):
        return self.image_paths

    
    def backward_D(self):
        import matplotlib.pyplot as plt
        # Fake
        # stop backprop to the generator by detaching fake_B
        label_fake = self.add_noise_disc(False)

        b,c,m,n = self.fake_B.size()
        rgb = 3 if self.opt.rgb else 1

        self.fake_B_reshaped = self.fake_B
        self.real_A_reshaped = self.real_A
        self.real_B_reshaped = self.real_B

        if self.opt.conditional:

            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A_reshaped, self.fake_B_reshaped), 1))
            
            fake_B_for_missing_data = self.fake_B.detach()[:, :self.disc_ch_for_missing_data]
            A_for_missing_data = self.real_A.detach()[:, :self.disc_ch_for_missing_data]
            fake_AB_no_missing = self.fake_AB_pool_no_missing.query(
                torch.cat((A_for_missing_data, fake_B_for_missing_data), 1)
            )
            
            # train the two language discriminator on fake output
            self.pred_fake_patch = self.netD.forward(fake_AB.detach())

            self.loss_D_fake = self.criterionGAN(self.pred_fake_patch, label_fake)
            
            # train the one language discriminator on fake output
            self.pred_fake_patch_for_missing_data = self.netD_for_missing_data.forward(fake_AB_no_missing.detach())
            self.loss_D_fake_for_missing_data = self.criterionGAN(self.pred_fake_patch_for_missing_data, label_fake)
            
            # train the preNet
            if self.opt.which_model_preNet != 'none':
                #transform the input
                transformed_AB = self.preNet_A.forward(fake_AB.detach())
                self.pred_fake = self.netD.forward(transformed_AB)
                self.loss_D_fake += self.criterionGAN(self.pred_fake, label_fake)
                
                transformed_AB_no_missing = self.preNet_A_missing_data.forward(fake_AB_no_missing.detach())
                self.pred_fake_for_missing_data = self.netD_for_missing_data.forward(transformed_AB_no_missing)
                self.loss_D_fake += self.criterionGAN(self.pred_fake_for_missing_data, label_fake)
                
                            
        else:
            # train the two language discriminator on fake output
            self.pred_fake = self.netD.forward(self.fake_B.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake, label_fake)

            # train the one-language discriminator on fake output
            fake_B_for_missing_data = self.fake_B.detach()[:, :self.disc_ch_for_missing_data]
            self.pred_fake_for_missing_data = self.netD_for_missing_data.forward(fake_B_for_missing_data)
            self.loss_D_fake_for_missing_data = self.criterionGAN(self.pred_fake_for_missing_data, label_fake)

        # Real
        label_real = self.add_noise_disc(True)
        # no idea how to do the conditional thing with missing data
        if self.opt.conditional:
            
            # figure out the mask and shape for the new first half of the data
            missing_data_mask_first_half = self.missing_data_mask[:self.real_B.shape[0]//2]

            real_A_first_half = self.real_A[:self.real_A.shape[0]//2]
            real_B_first_half = self.real_B[:self.real_B.shape[0]//2]
            new_shape = list(real_A_first_half.shape)
            new_shape[1] = -1
            
            # mask the first half of the input
            real_A_first_half_no_missing = real_A_first_half[~missing_data_mask_first_half].reshape(new_shape)
            # and the output
            real_B_first_half_no_missing = real_B_first_half[~missing_data_mask_first_half].reshape(new_shape)
            
            # append them
            real_AB_first_half = torch.cat((real_A_first_half_no_missing, real_B_first_half_no_missing), 1)#.detach()
            self.pred_real_patch = self.netD.forward(real_AB_first_half)
            self.loss_D_real = self.criterionGAN(self.pred_real_patch, label_real)
            
            # and figure it out for the second half
            real_A_second_half = self.real_A[self.real_A.shape[0]//2:]
            missing_data_mask_second_half = self.missing_data_mask[self.real_B.shape[0]//2:]
            real_A_second_half_no_missing = real_A_second_half[~missing_data_mask_second_half].reshape(new_shape)
            # and the output
            real_B_second_half = self.real_B[self.real_B.shape[0]//2:]
            
            real_B_second_half_no_missing = real_B_second_half[~missing_data_mask_second_half].reshape(new_shape)
            
            # append them
            real_AB_second_half = torch.cat((real_A_second_half_no_missing, real_B_second_half_no_missing), 1)#.detach()
            self.pred_real_patch_for_missing_data = self.netD_for_missing_data.forward(real_AB_second_half)
            self.loss_D_real_for_missing_data = self.criterionGAN(self.pred_real_patch_for_missing_data, label_real)

            if self.opt.which_model_preNet != 'none':
                #transform the input
                transformed_A_real = self.preNet_A.forward(real_AB_first_half)
                self.pred_real = self.netD.forward(transformed_A_real)
                self.loss_D_real += self.criterionGAN(self.pred_real, label_real)
                
                transformed_A_real_no_missing = self.preNet_A_missing_data.forward(real_AB_second_half)
                self.pred_real_for_missing_data = self.netD_for_missing_data.forward(transformed_A_real_no_missing)
                self.loss_D_real_for_missing_data += self.criterionGAN(self.pred_real_for_missing_data, label_real)
                            
        else:
            # don't test the discriminator on missing data
            # the first half of the real examples are used for the two-language discriminator
            real_B_first_half = self.real_B[:self.real_B.shape[0]//2]
            missing_data_mask_first_half = self.missing_data_mask[:self.real_B.shape[0]//2]
            new_shape = list(real_B_first_half.shape)
            new_shape[1] = -1
            # print('real B first half:', real_B_first_half.shape, 'mask sum', np.sum(missing_data_mask_first_half.numpy()))
            real_B_first_half_no_missing = real_B_first_half[~missing_data_mask_first_half].reshape(new_shape)
            
            self.pred_real = self.netD.forward(real_B_first_half_no_missing)
            self.loss_D_real = self.criterionGAN(self.pred_real, label_real)

            # the second half can only be used for the one-language discriminator
            real_B_second_half = self.real_B[self.real_B.shape[0]//2:]
            missing_data_mask_second_half = self.missing_data_mask[self.real_B.shape[0]//2:]
            real_B_second_half_no_missing = real_B_second_half[~missing_data_mask_second_half].reshape(new_shape)
            
            self.pred_real_for_missing_data = self.netD_for_missing_data.forward(real_B_second_half_no_missing)
            self.loss_D_real_for_missing_data = self.criterionGAN(self.pred_real_for_missing_data, label_real)

        # Combined loss
        self.loss_D = (self.loss_D_fake * 0.5 + self.loss_D_real) * 0.5
        self.loss_D_for_missing_data = (self.loss_D_fake_for_missing_data * 0.5 + self.loss_D_real_for_missing_data) * 0.5

        self.loss_D.backward()
        self.loss_D_for_missing_data.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.conditional:
            # two-language discriminator output
            fake_AB = torch.cat((self.real_A_reshaped, self.fake_B_reshaped), 1)
            pred_fake_patch = self.netD.forward(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake_patch, True)
            
            # one-language discriminator output
            fake_AB_for_missing_data = torch.cat((self.real_A_reshaped[:, :self.disc_ch_for_missing_data], self.fake_B_reshaped[:, :self.disc_ch_for_missing_data]), 1)
            pred_fake_patch_for_missing_data = self.netD_for_missing_data.forward(fake_AB_for_missing_data)
            self.loss_G_GAN_for_missing_data = self.criterionGAN(pred_fake_patch_for_missing_data, True)
            if self.opt.which_model_preNet != 'none':
                #global disc
                transformed_A = self.preNet_A.forward(fake_AB)
                pred_fake = self.netD.forward(transformed_A)
                self.loss_G_GAN += self.criterionGAN(pred_fake, True)

                transformed_A_for_missing_data = self.preNet_A_missing_data.forward(fake_AB_for_missing_data)
                pred_fake_for_missing_data = self.netD_for_missing_data.forward(transformed_A_for_missing_data)
                self.loss_G_GAN_for_missing_data += self.criterionGAN(pred_fake_for_missing_data, True)
        
        else:
            # train the generator on the two-language discriminator output
            pred_fake = self.netD.forward(self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # train the generator on the one-language discriminator output
            fake_B_for_missing_data = self.fake_B[:, :self.disc_ch_for_missing_data]
            pred_fake_for_missing_data = self.netD_for_missing_data.forward(fake_B_for_missing_data)
            self.loss_G_GAN_for_missing_data = self.criterionGAN(self.pred_fake_for_missing_data, True)
        
        # L1 loss for the first half
        real_B_first_half = self.real_B[:self.real_B.shape[0]//2]
        missing_data_mask_first_half = self.missing_data_mask[:self.real_B.shape[0]//2]
        new_shape = list(real_B_first_half.shape)
        new_shape[1] = -1
        real_B_first_half_no_missing = real_B_first_half[~missing_data_mask_first_half].reshape(new_shape)

        fake_B_first_half = self.fake_B[:self.fake_B.shape[0]//2]
        fake_B_first_half_no_missing = fake_B_first_half[~missing_data_mask_first_half].reshape(new_shape)

        self.loss_G_L1 = self.criterionL1(fake_B_first_half_no_missing, real_B_first_half_no_missing) * self.opt.lambda_A

        # L1 loss for the second half
        real_B_second_half = self.real_B[self.real_B.shape[0]//2:]
        missing_data_mask_second_half = self.missing_data_mask[self.real_B.shape[0]//2:]
        real_B_second_half_no_missing = real_B_second_half[~missing_data_mask_second_half].reshape(new_shape)

        fake_B_second_half = self.fake_B[self.fake_B.shape[0]//2:]
        fake_B_second_half_no_missing = fake_B_second_half[~missing_data_mask_second_half].reshape(new_shape)

        self.loss_G_L1_for_missing_data = self.criterionL1(fake_B_second_half_no_missing, real_B_second_half_no_missing) * self.opt.lambda_A
        

        self.loss_G = (self.loss_G_GAN + self.loss_G_GAN_for_missing_data + self.loss_G_L1 + self.loss_G_L1_for_missing_data) * 0.5

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.optimizer_D_missing_data.zero_grad()
        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_D_missing_data.step()
        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.step()


        self.optimizer_G.zero_grad()
        if self.opt.conv3d:
            self.optimizer_G_3d.zero_grad()

        self.backward_G()
        self.optimizer_G.step()
        if self.opt.conv3d:
            self.optimizer_G_3d.step()
        
    

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                ('G_L1', self.loss_G_L1.data.item()),
                ('D_real_two_language', self.loss_D_real.item()),
                ('D_fake_two_language', self.loss_D_fake.item()),
                ('D_real_one_language', self.loss_D_real_for_missing_data.item()),
                ('D_fake_one_language', self.loss_D_fake_for_missing_data.item())
        ])


    def get_current_visuals(self):
        if self.real_A.shape[0] > 1:
            ind = np.random.randint(len(self.real_A))
            real_A = util.tensor2im(self.real_A.data[ind:ind+1])
            fake_B = util.tensor2im(self.fake_B.data[ind:ind+1])
            real_B = util.tensor2im(self.real_B.data[ind:ind+1])
        else:
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        if self.opt.conv3d:
             self.save_network(self.netG_3d, 'G_3d', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netG, 'G', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netD, 'D', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netD_for_missing_data, 'D_missing_data', label, gpu_ids=self.gpu_ids)
        if self.opt.which_model_preNet != 'none':
            self.save_network(self.preNet_A, 'PRE_A', label, gpu_ids=self.gpu_ids)
            

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        if self.opt.which_model_preNet != 'none':
            for param_group in self.optimizer_preA.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.conv3d:
            for param_group in self.optimizer_G_3d.param_groups:
                param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
