#=============================
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
#=============================



from data.image_folder import ImageFolder
from data.base_data_loader import BaseDataLoader
import random
import torch.utils.data
import torchvision.transforms as transforms
from builtins import object
import os
import numpy as np
from torch import LongTensor,index_select
from scipy import misc
import util.util as util
from PIL import Image
from torch.nn import UpsamplingBilinear2d
from torch.autograd import Variable
import warnings
import pickle


def normalize_stack(input,val=0.5):
    #normalize an tensor with arbitrary number of channels:
    # each channel with mean=std=val
    val=0.5
    len_ = input.size(1)
    mean = (val,)*len_
    std = (val,)*len_
    t_normal_stack = transforms.Compose([
        transforms.Normalize(mean,std)])
    return t_normal_stack(input)



def CreateDataLoader(opt):
    data_loader = None
    if opt.auxiliarydataroot != 'none':
        data_loader = MultiLanguageDataLoader()
    elif opt.stack:
        data_loader = StackDataLoader()
    elif opt.partial:
        data_loader = PartialDataLoader()
    else:
        data_loader = DataLoader()
    data_loader.initialize(opt)
    return data_loader


class FlatData(object):
    def __init__(self, data_loader, data_loader_base, fineSize, max_dataset_size, rgb, dict_test={},base_font=False, blanks=0.7):
        self.data_loader = data_loader
        self.data_loader_base = data_loader_base
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.blanks = blanks
        self.data_loader_base_iter = iter(self.data_loader_base)
        self.A_base,self.A_base_paths = next(self.data_loader_base_iter)
        self.base_font = base_font
        self.random_dict=dict_test
        self.dict = False
        if len(self.random_dict.keys()):
            self.dict = True


        
        
    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total/2)
        h = AB.size(2)
        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A =  torch.mean(AB,dim=1) #only one of the RGB channels    
        A = A[:,None,:,:] #(m,1,64,64*26)
        B =  torch.mean(AB,dim=1) #only one of the RGB channels    
        B = B[:,None,:,:] #(m,1,64,64*26)
        n_rgb = 3 if self.rgb else 1
        target_size = A.size(2)
        AA = A.clone() 
        if self.blanks != 0:       
            #randomly remove some of the glyphs

            if not self.dict:
                blank_ind = np.random.permutation(A.size(3)/target_size)[0:int(self.blanks*A.size(3)/target_size)]
            else:
                file_name = map(lambda x:x.split("/")[-1],AB_paths)
                blank_ind = self.random_dict[file_name][0:int(self.blanks*A.size(3)/target_size)]

            blank_ind = np.tile(range(target_size), len(blank_ind)) + np.repeat(blank_ind*target_size,target_size)
            AA.index_fill_(3,LongTensor(list(blank_ind)),1)
            # t_topil = transforms.Compose([
            #     transforms.ToPILImage()])

            # AA_ = t_topil(AA[0,0,:,:].unsqueeze_(0))
            # misc.imsave('./AA_0.png',AA_)            

        return {'A': AA, 'A_paths': AB_paths, 'B':B, 'B_paths':AB_paths}




class Data(object):
    def __init__(self, data_loader, fineSize, max_dataset_size, rgb, dict_test={}, blanks=0.7):
        self.data_loader = data_loader
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.blanks = blanks
        self.random_dict=dict_test
        self.dict = False
        if len(self.random_dict.keys()):
            self.dict = True

        
    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total / 2)
        h = AB.size(2)
        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
        B = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset: w_offset + self.fineSize]
        n_rgb = 3 if self.rgb else 1
        
        
        if self.blanks == 0:
            AA = A.clone()
        else: 
            # SG 12/4 --- Edit this to handle missing devanagari GT
            #randomly remove some of the glyphs in input
            if not self.dict:
                # raise RuntimeError('A:' + str(A) + '\n' + str(A.size(1)))
                blank_ind = np.repeat(np.random.permutation(A.size(1)//n_rgb)[0:int(self.blanks*A.size(1)//n_rgb)],n_rgb)
            else:
                file_name = list(map(lambda x:x.split("/")[-1],AB_paths))
                if len(file_name)>1:
                    raise Exception('batch size should be 1')
                file_name=file_name[0]
                blank_ind = self.random_dict[file_name][0:int(self.blanks*A.size(1)/n_rgb)]

            rgb_inds = np.tile(range(n_rgb),int(len(blank_ind)))
            # raise RuntimeError('rgb_inds ' + str(len(rgb_inds)) + ' blank_ind ' + str(len(blank_ind)))
            blank_ind = blank_ind*n_rgb + rgb_inds
            AA = A.clone()
            AA.index_fill_(1,LongTensor(list(blank_ind)),1)
            
        return {'A': AA, 'A_paths': AB_paths, 'B':B, 'B_paths':AB_paths}
    
    
class MultiLanguageData(object):
    def __init__(self, data_loader_both, data_loader_one, data_loader_one_missing_characters,
                 fineSize, max_dataset_size, rgb, dict_test={}, blanks=0.7, only_condition_on_one_language=False):
        self.data_loader_both = data_loader_both
        self.data_loader_one = data_loader_one
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.blanks = blanks
        self.random_dict=dict_test
        self.dict = False
        self.only_condition_on_one_language = only_condition_on_one_language
        
        self.n_missing_characters = data_loader_one_missing_characters # magic number for devanagari
        
        if len(self.random_dict.keys()):
            self.dict = True

        
    def __iter__(self):
        self.data_loader_both_iter = iter(self.data_loader_both)
        self.data_loader_one_iter = iter(self.data_loader_one)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        AB_both, AB_paths_both = next(self.data_loader_both_iter)
        AB_one, AB_paths_one = next(self.data_loader_one_iter)
        AB_paths = AB_paths_both + AB_paths_one

        assert AB_both.size(3) == AB_one.size(3) # our images should be the same shape
        assert AB_both.size(2) == AB_one.size(2)
        # handle the uneven batch size case
        if AB_both.size(0) != AB_one.size(0):
            min_size = min(AB_both.size(0), AB_one.size(0))
            AB_both = AB_both[:min_size]
            AB_one = AB_one[:min_size]
        # I think this is randomly offsetting each image for data augmentation
        w_total = AB_both.size(3)
        w = int(w_total / 2)
        h = AB_both.size(2)
        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A_both = AB_both[:, :, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
        B_both = AB_both[:, :, h_offset:h_offset + self.fineSize,
               w_offset: w_offset + self.fineSize]
        
        A_one = AB_one[:, :, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
        B_one = AB_one[:, :, h_offset:h_offset + self.fineSize,
               w_offset: w_offset + self.fineSize]
        n_rgb = 3 if self.rgb else 1
        
        
        if self.blanks == 0:
            AA = A.clone()
        else:
            #randomly remove some of the glyphs in input
            if not self.dict:
                # if there is no file, do it randomly
                # raise RuntimeError('A:' + str(A) + '\n' + str(A.size(1)))
                # get a permutation of the characters. Take the first self.blanks of them
                # and repeat it across the RGB axis
                if self.only_condition_on_one_language:
                    present_chars = A_both.size(1)//n_rgb - self.n_missing_characters
                    blank_ind_both = np.repeat(np.random.permutation(present_chars//n_rgb)[0:int(self.blanks*present_chars//n_rgb)],n_rgb)
                    blank_ind_both = np.concatenate([blank_ind_both, np.arange(present_chars, A_one.size(1), dtype='int32')])
                else:
                    blank_ind_both = np.repeat(np.random.permutation(A_both.size(1)//n_rgb)[0:int(self.blanks*A_both.size(1)//n_rgb)],n_rgb)
                # the number of present characters
                present_chars = A_one.size(1)//n_rgb - self.n_missing_characters
                blank_ind_one = np.repeat(np.random.permutation(present_chars//n_rgb)[0:int(self.blanks*present_chars//n_rgb)],n_rgb)
                blank_ind_one = np.concatenate([blank_ind_one, np.arange(present_chars, A_one.size(1), dtype='int32')])
            else:
                # otherwise get the blank indices from a provided file
                file_name = list(map(lambda x:x.split("/")[-1],AB_paths))
                if len(file_name)>1:
                    raise Exception('batch size should be 1')
                file_name=file_name[0]
                blank_ind_both = self.random_dict[file_name]
                blank_ind_one = blank_ind_both
                print('blank ind!', blank_ind_both)
            if self.only_condition_on_one_language:
                rgb_inds_both = np.tile(range(n_rgb),int(self.blanks*present_chars + self.n_missing_characters))
            else:
                rgb_inds_both = np.tile(range(n_rgb),int(self.blanks*A_both.size(1)/n_rgb))
            rgb_inds_one = np.tile(range(n_rgb),int(self.blanks*present_chars + self.n_missing_characters))
            # raise RuntimeError('rgb_inds ' + str(len(rgb_inds)) + ' blank_ind ' + str(len(blank_ind)))
            blank_ind_both = blank_ind_both*n_rgb + rgb_inds_both
            blank_ind_one = blank_ind_one*n_rgb + rgb_inds_one
            AA_both = A_both.clone()
            AA_both.index_fill_(1,LongTensor(list(blank_ind_both)),1)
            AA_one = A_one.clone()
            AA_one.index_fill_(1,LongTensor(list(blank_ind_one)),1)
            
            AA = torch.cat([AA_both, AA_one], 0)
            B = torch.cat([B_both, B_one], 0)
            # create a mask describing which characters are missing
            missing_data_mask = torch.zeros_like(B)
            missing_data_mask[B_both.size(0):,-self.n_missing_characters:] += 1
            missing_data_mask = missing_data_mask.bool()
            
            # print('AA:', AA.shape, 'B_both:', B_both.shape, 'B_one:', B_one.shape, 'B:', B.shape, missing_data_mask.shape)
        return {'A': AA, 'A_paths': AB_paths, 'B':B, 'B_paths':AB_paths, 'missing_data_mask':missing_data_mask}
    

class PartialData(object):
    def __init__(self, data_loader_A, data_loader_B, data_loader_base, fineSize, loadSize, max_dataset_size, phase, base_font=False, blanks=0):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.data_loader_base = data_loader_base
        self.fineSize = fineSize
        self.loadSize = loadSize
        self.max_dataset_size = max_dataset_size
        self.blanks = blanks

        self.base_font = base_font
        if base_font:        
            self.data_loader_base_iter = iter(self.data_loader_base)
            self.A_base,self.A_base_paths = next(self.data_loader_base_iter)
            self.A_base[0,:,:,:]=normalize_stack(self.A_base[0,:,:,:]) 
        else: 
            self.A_base = []
        self.phase =phase
        if self.phase=='train':

            t_tensor =  transforms.Compose([ 
                transforms.ToTensor(),])
            t_topil = transforms.Compose([
                transforms.ToPILImage()])             
                
            if self.base_font:
                for ind in range(self.A_base.size(1)):
                    A_base = t_topil(self.A_base[0,ind,:,:].unsqueeze(0))
    

    def __iter__(self):
        self.data_loader_iter_A = iter(self.data_loader_A)
        self.data_loader_iter_B = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration

        A, A_paths = next(self.data_loader_iter_A)
        B, B_paths = next(self.data_loader_iter_B)
        

        t_topil = transforms.Compose([
            transforms.ToPILImage()])
        t_scale = transforms.Compose([
            transforms.Scale(self.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        t_normal = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
         
        t_tensor =  transforms.Compose([ 
            transforms.ToTensor(),])                     
        
        
        for index in range(A.size(0)):
            A[index,:,:,:]=normalize_stack(A[index,:,:,:])
            B[index,:,:,:]=normalize_stack(B[index,:,:,:])
            BB=t_topil(B[index,:,:,:])
                         


            # remove more of the glyphs to make prediction harder  
            if self.blanks!=0: 
                gt_glyph = [int(A_paths[index].split('/')[-1].split('.png')[0].split('_')[-1])]
                observed_glyph = list(set(np.nonzero(1-A[index,:,:,:])[:,0]) - set(gt_glyph))
                observed_glyph = np.random.permutation(observed_glyph)
                blank_nums = 1
                for i in range(blank_nums):
                    A[index,observed_glyph[i],:,:] = 1
                    
        return {'A': A, 'A_paths': A_paths, 'B':B, 'B_paths':B_paths, 'A_base':self.A_base}
                      


class StackDataLoader(BaseDataLoader):
    """ a subset of the glyphs are observed and being used for transferring style
        train a pix2pix model conditioned on b/w glyphs
        to generate colored glyphs.
    """

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Scale(opt.loadSize),
            transforms.ToTensor(),
                                 ])
        dic_phase = {'train':'Train', 'test':'Test'}
	    # Dataset A
        dataset_A = ImageFolder(root=opt.dataroot +'A/'+ opt.phase,
                              transform=transform, return_paths=True, rgb=opt.rgb_in,
                              fineSize=opt.fineSize, loadSize=opt.loadSize,
                            font_trans=True, no_permutation=opt.no_permutation)
        len_A = len(dataset_A.imgs)
        shuffle_inds = np.random.permutation(len_A)
        
        dataset_B = ImageFolder(root=opt.dataroot  + 'B/'+ opt.phase,
                              transform=transform, return_paths=True, rgb=opt.rgb_out,
                              fineSize=opt.fineSize, loadSize=opt.loadSize,
                              font_trans=False, no_permutation=opt.no_permutation)


        if len(dataset_A.imgs)!=len(dataset_B.imgs):
            raise Exception("number of images in source folder and target folder does not match")

        if (opt.partial and (not self.opt.serial_batches)):
            dataset_A.imgs = [dataset_A.imgs[i] for i in shuffle_inds]
            dataset_B.imgs = [dataset_B.imgs[i] for i in shuffle_inds]
            dataset_A.img_crop = [dataset_A.img_crop[i] for i in shuffle_inds]
            dataset_B.img_crop = [dataset_B.img_crop[i] for i in shuffle_inds]
            shuffle = False 
        else:
            shuffle = not self.opt.serial_batches
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=self.opt.batchSize,
            shuffle=shuffle,
            num_workers=int(self.opt.nThreads))


        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=self.opt.batchSize,
            shuffle=shuffle,
            num_workers=int(self.opt.nThreads))

        if opt.base_font:
            
            #Read and apply transformation on the BASE font 
            dataset_base = ImageFolder(root=opt.base_root,
                                  transform=transform, return_paths=True, font_trans=True, rgb=opt.rgb,
                                   fineSize=opt.fineSize, loadSize=opt.loadSize) 
            data_loader_base = torch.utils.data.DataLoader(
                dataset_base,
                batch_size=1,
                shuffle=False,
                num_workers=int(self.opt.nThreads))
        else:
            data_loader_base=None



        self.dataset_A = dataset_A
        self._data = PartialData(data_loader_A,data_loader_B, data_loader_base, opt.fineSize, opt.loadSize, opt.max_dataset_size, opt.phase, opt.base_font, opt.blanks)
    def name(self):
        return 'StackDataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset_A), self.opt.max_dataset_size)





class PartialDataLoader(BaseDataLoader):
    """ a subset of the glyphs are observed and being used for training stlye
        train a pix2pix model conditioned on b/w glyphs
        to generate colored glyphs.
        In the pix2pix model it is simmilar to the unaligned data class.
    """

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Scale(opt.loadSize),
            transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5),
#                                  (0.5, 0.5, 0.5))
                                 ])
        dic_phase = {'train':'Train', 'test':'Test'}
        # Dataset A
        
        dataset_A = ImageFolder(root=opt.dataroot +'A/'+ opt.phase,
                              transform=transform, return_paths=True, rgb=opt.rgb, fineSize=opt.fineSize,
                              loadSize=opt.loadSize, font_trans=False, no_permutation=opt.no_permutation)
        len_A = len(dataset_A.imgs)
        if not opt.no_permutation:
            shuffle_inds = np.random.permutation(len_A)
        else:
            shuffle_inds = range(len_A)
        
        dataset_B = ImageFolder(root=opt.dataroot  + 'B/'+ opt.phase,
                              transform=transform, return_paths=True, rgb=opt.rgb, fineSize=opt.fineSize,
                              loadSize=opt.loadSize, font_trans=False, no_permutation=opt.no_permutation)

        if len(dataset_A.imgs)!=len(dataset_B.imgs):
            raise Exception("number of images in source folder and target folder does not match")

        if (opt.partial and (not self.opt.serial_batches)):
            dataset_A.imgs = [dataset_A.imgs[i] for i in shuffle_inds]
            dataset_B.imgs = [dataset_B.imgs[i] for i in shuffle_inds]
            dataset_A.img_crop = [dataset_A.img_crop[i] for i in shuffle_inds]
            dataset_B.img_crop = [dataset_B.img_crop[i] for i in shuffle_inds]
            shuffle = False 
        else:
            shuffle = not self.opt.serial_batches
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=self.opt.batchSize,
            shuffle=shuffle,
            num_workers=int(self.opt.nThreads))


        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=self.opt.batchSize,
            shuffle=shuffle,
            num_workers=int(self.opt.nThreads))
            
        if opt.base_font:
            #Read and apply transformation on the BASE font 
            dataset_base = ImageFolder(root=opt.base_root,
                                  transform=transform, return_paths=True, font_trans=True, rgb=opt.rgb,
                                   fineSize=opt.fineSize, loadSize=opt.loadSize) 
            data_loader_base = torch.utils.data.DataLoader(
                dataset_base,
                batch_size=1,
                shuffle=False,
                num_workers=int(self.opt.nThreads))
        else:
            data_loader_base = None



        self.dataset_A = dataset_A
        self._data = PartialData(data_loader_A,data_loader_B, data_loader_base, opt.fineSize, opt.loadSize, opt.max_dataset_size, opt.phase, opt.base_font)
    def name(self):
        return 'PartialDataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset_A), self.opt.max_dataset_size)



class DataLoader(BaseDataLoader):
    def initialize(self, opt):
        
        BaseDataLoader.initialize(self, opt)
        self.fineSize = opt.fineSize
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Scale(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        # Dataset A
        dataset = ImageFolder(root=opt.dataroot + '/' + opt.phase,
                              transform=transform, return_paths=True, font_trans=(not opt.flat), rgb=opt.rgb,
                               fineSize=opt.fineSize, loadSize=opt.loadSize) 
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))
            
       
        self.dataset = dataset
        dict_inds = {}
        test_dict = opt.dataroot+'/test_dict/dict.pkl'
        if opt.phase=='test':
            if os.path.isfile(test_dict):
                dict_inds = pickle.load(open(test_dict, 'rb'), encoding='latin1')
            else:
                warnings.warn('Blanks in test data are random. create a pkl file in ~/data_path/test_dict/dict.pkl including predifined random indices')



        if opt.flat:
            self._data = FlatData(data_loader, data_loader_base, opt.fineSize, opt.max_dataset_size, opt.rgb, dict_inds, opt.base_font, opt.blanks)
        else:
            self._data = Data(data_loader, opt.fineSize, opt.max_dataset_size, opt.rgb, dict_inds, opt.blanks)

    def name(self):
        return 'DataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    
class MultiLanguageDataLoader(BaseDataLoader):
    """
    New 12/4
    A dataloader to load a combination of bilingual fonts and latin-only fonts,
    each batch should be split 50/50 between them.
    """
    def initialize(self, opt):
        
        BaseDataLoader.initialize(self, opt)
        self.fineSize = opt.fineSize
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Scale(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        # Dataset A
        dataset_both = ImageFolder(root=opt.dataroot + '/' + opt.phase,
                                        transform=transform, return_paths=True, font_trans=(not opt.flat), rgb=opt.rgb,
                                        fineSize=opt.fineSize, loadSize=opt.loadSize)
        dataset_one = ImageFolder(root=opt.auxiliarydataroot + '/' + opt.phase,
                                   transform=transform, return_paths=True, font_trans=(not opt.flat), rgb=opt.rgb,
                                   fineSize=opt.fineSize, loadSize=opt.loadSize)
        
        if opt.batchSize % 2 != 0:
            raise ValueError('Batch Size must be even if split between two datasets')
            
        data_loader_one = torch.utils.data.DataLoader(
            dataset_one,
            batch_size=self.opt.batchSize//2,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))
            
        data_loader_both = torch.utils.data.DataLoader(
            dataset_both,
            batch_size=self.opt.batchSize//2,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))
       
        self.dataset = (dataset_one, dataset_both)
        dict_inds = {}
        test_dict = opt.dataroot+'/test_dict/dict.pkl'
        if opt.phase=='test':
            if os.path.isfile(test_dict):
                dict_inds = pickle.load(open(test_dict, 'rb'), encoding='latin1')
            else:
                warnings.warn('Blanks in test data are random. create a pkl file in ~/data_path/test_dict/dict.pkl including predifined random indices')



        """if opt.flat:
            self._data = FlatData(data_loader, data_loader_base, opt.fineSize, opt.max_dataset_size, opt.rgb, dict_inds, opt.base_font, opt.blanks)
        else:"""
        self._data = MultiLanguageData(data_loader_both, data_loader_one, opt.auxiliarymissingcharacters, opt.fineSize, 
                                        opt.max_dataset_size, opt.rgb, dict_inds, opt.blanks, only_condition_on_one_language=opt.onlyconditionononelanguage)

    def name(self):
        return 'MultiLanguageDataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset[0]), len(self.dataset[1]), self.opt.max_dataset_size)
    