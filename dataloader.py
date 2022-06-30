import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import h5py
import os
import json
import numpy as np



class Load_dataset(data.Dataset):
    def __init__(self, mode='train', config=None, dataset='vrd'):
        if mode == 'train':
            if dataset == 'vrd':
                self.data_dir = 'data/dataset-vrd/train/'
            elif dataset == 'vg':
                self.data_dir = 'data/dataset-vg/train/'
            elif dataset == 'clevr':
                self.data_dir = 'data/dataset-clevr/train/'
        elif mode == 'val':
            if dataset == 'vrd':
                self.data_dir = 'data/dataset-vrd/val/'
            elif dataset == 'vg':
                self.data_dir = 'data/dataset-vg/val/'
            elif dataset == 'clevr':
                self.data_dir = 'data/dataset-clevr/val/'
        elif mode == 'test':
            if dataset == 'vrd':
                self.data_dir = 'data/dataset-vrd/test/'
            elif dataset == 'vg':
                self.data_dir = 'data/dataset-vg/test/'
            elif dataset == 'clevr':
                self.data_dir = 'data/dataset-clevr/test/'

        self.config = config
        self.images = h5py.File(os.path.join(self.data_dir, 'images.hdf5'), 'r')
        self.dataset = h5py.File(os.path.join(self.data_dir, 'dataset.hdf5'), 'r')

        self.imgs = self.images['images']
        self.categories = self.dataset['categories']
        self.subjects = self.dataset['subject_locations']
        self.objects = self.dataset['object_locations']

        self.len = len(self.categories)
        self.target_size = self.config["output_dim"]*self.config["output_dim"]
    

    def __len__(self):
        return self.len
    

    def __getitem__(self, idx):
        
        # Create the batches.
        rel = self.categories[idx]
        s_regions = self.subjects[idx].reshape(self.target_size)
        #print(s_regions.max())
        o_regions = self.objects[idx].reshape(self.target_size)
        #print(o_regions.max())
        input_img = np.array(self.imgs[rel[3]])
        
        input_img[:, :, 2] -= 103.939
        input_img[:, :, 1] -= 116.779
        input_img[:, :, 0] -= 123.68
        
        input_img = transforms.ToTensor()(input_img)
        #input_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input_img)

        # Choose the inputs based on the parts of the relationship we will use.
        
        if self.config["use_subject"]:
            input_subj = rel[0]
        if self.config["use_predicate"]:
            input_rel = rel[1]
        if self.config["use_object"]:
            input_obj = rel[2]
        return input_img, input_subj, input_rel, input_obj, s_regions, o_regions


    def collate_fn(self, data):       
        input_img, input_subj, input_rel, input_obj, s_regions, o_regions = zip(*data)
        input_img = torch.stack(input_img)
        return input_img, input_subj, input_rel, input_obj, s_regions, o_regions



if __name__=='__main__':
    with open('config.json', 'r') as load_f:
        cfg = json.load(load_f)
    '''
    dataset_test = Load_dataset(mode='test', config=cfg)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=cfg['train_batch_size'],
                                                shuffle=False, num_workers=cfg['num_workers'], 
                                                collate_fn=dataset_test.collate_fn)

    for i, (input_img, input_subj, input_cate, input_obj, s_regions, o_regions) in enumerate(loader_test):
        print(input_img[0].shape, input_subj[0].shape, input_cate[0].shape, input_obj[0].shape, s_regions[0].shape, o_regions[0].shape)
        break
    '''
    dataset_test = Load_dataset(mode='train', config=cfg, dataset='clevr')
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                shuffle=False, 
                                                collate_fn=dataset_test.collate_fn)
    print(dataset_test.len)
    '''
    for i, (input_img, input_subj, input_cate, input_obj, s_regions, o_regions) in enumerate(loader_test):
        print('sample',i)
    '''
