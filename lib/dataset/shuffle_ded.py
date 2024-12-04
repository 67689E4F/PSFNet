import torch
import h5py
from torch.utils.data import Dataset
import os, cv2, random, json

import numpy as np
from pathlib import Path
from utils.trans_img import get_affine_transform, affine_transform

class Shuffle_DED(Dataset):
    def __init__(self, cfg, train_val_test, dataname="Shuffle_DED", load_all_data=True):
        self.path_shuffle = os.path.join(cfg.DATASET.PATH, "aberration/defocus_coco_seidel")
        self.path_ded = os.path.join(cfg.DATASET.PATH, "defocus/DED")
        self.train_val_test = train_val_test
        self.load_all_data = load_all_data
        self.path_img_shuffle = os.path.join(self.path_shuffle, train_val_test)

        self.path_img_ded = os.path.join(self.path_ded, 'image', train_val_test)
        self.path_label_ded = os.path.join(self.path_ded, 'label', train_val_test)



        self.imgfile_list_shuffle = os.listdir(self.path_img_shuffle)
        self.imgfile_list_ded = os.listdir(self.path_img_ded)

        self.num_shuffle = len(self.imgfile_list_shuffle)
        self.num_ded = len(self.imgfile_list_ded)


        self.cfg = cfg
        self.heatmap_size = cfg.MODEL.IMAGE_SIZE #w h
        if load_all_data:
            self.shuffledata_all = {}
            for fileName in self.imgfile_list_shuffle:
                path_data = os.path.join(self.path_img_shuffle, fileName)

                try:
                    with h5py.File(path_data, 'r') as hf:

                        self.shuffledata_all[fileName] = hf['67689e4f'][:,:,:]
                        hf.close()
                except:
                    print(path_data)

    def __len__(self,):
        return self.num_shuffle


    def __getitem__(self, idx):
        img_filename_shuffle = self.imgfile_list_shuffle[idx]
        if self.load_all_data:
            data = self.shuffledata_all[img_filename_shuffle]
        else:
            path_data = os.path.join(self.path_img_shuffle, img_filename_shuffle)

            try:
                with h5py.File(path_data, 'r') as hf:

                    data = hf['67689e4f'][:,:,:]
                    hf.close()
            except:
                print(path_data)

        img_shuffle = data[:3, :, :]
        img_shuffle = img_shuffle.transpose(1,2,0)
        label_shuffle = data[3, :, :]
        height_shuffle, width_shuffle = img_shuffle.shape[0], img_shuffle.shape[1]
        center_img_shuffle = np.array([img_shuffle.shape[1] / 2., img_shuffle.shape[0] / 2.], dtype=np.float32)
        s_shuffle = np.array([width_shuffle, height_shuffle], dtype=np.float32)
  

        # id_ded = idx % self.num_ded #这里应该改为随机数更好
        id_ded = random.randint(0, self.num_ded-1)

        img_filename_ded = self.imgfile_list_ded[id_ded]
        img_ded = cv2.imread(os.path.join(self.path_img_ded, img_filename_ded))
        label_ded = cv2.imread(os.path.join(self.path_label_ded, img_filename_ded.replace('image', "defocus").replace("jpg", "png")), 0)
        height_ded, width_ded = img_ded.shape[0], img_ded.shape[1]
        center_img_ded = np.array([img_ded.shape[1] / 2., img_ded.shape[0] / 2.], dtype=np.float32)
        s_ded = np.array([width_ded, height_ded], dtype=np.float32)

        if self.train_val_test == "train":
            # cv_blur = random.random()
            # if 0< cv_blur <0.2 :
            #     sigma_blur = random.randint(1,5)
            #     img = cv2.GaussianBlur(img,[255,255], sigmaX=sigma_blur, sigmaY=sigma_blur)
            #     label = ((label * 15)**2 + sigma_blur**2)**0.5 / 15.8114




            if 0 < random.random() <= self.cfg.FLIP:
                img_shuffle = img_shuffle[:, ::-1, :]
                label_shuffle = label_shuffle[:,::-1]
                center_img_shuffle[0] =  width_shuffle - center_img_shuffle[0] - 1

                img_ded = img_ded[:, ::-1, :]
                label_ded = label_ded[:,::-1]
                center_img_ded[0] =  width_ded - center_img_ded[0] - 1



            rotation = np.clip(np.random.randn()*self.cfg.ROTATION, -self.cfg.ROTATION, self.cfg.ROTATION) \
                if random.random() <= 0.6 else 0
            trans_shuffle = get_affine_transform(center_img_shuffle, s_shuffle, rotation, [width_shuffle, height_shuffle])
            rotation = np.clip(np.random.randn()*self.cfg.ROTATION, -self.cfg.ROTATION, self.cfg.ROTATION) \
                if random.random() <= 0.6 else 0
            trans_ded = get_affine_transform(center_img_ded, s_ded, rotation, [width_ded, height_ded])
            
            img_shuffle = cv2.warpAffine(img_shuffle, trans_shuffle, 
                         (width_shuffle, height_shuffle),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT
                         )
            label_shuffle = cv2.warpAffine(label_shuffle, trans_shuffle, 
                         (width_shuffle, height_shuffle),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT
                         )
            
        
        
            img_ded = cv2.warpAffine(img_ded, trans_ded, 
                         (width_ded, height_ded),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT
                         )
            label_ded = cv2.warpAffine(label_ded, trans_ded, 
                         (width_ded, height_ded),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT
                         )
            
        

            if img_shuffle.shape[1] - self.heatmap_size[0]>0 and img_shuffle.shape[0] - self.heatmap_size[1]>0:
                cut_w = np.random.randint(low=0, high=img_shuffle.shape[1] - self.heatmap_size[0])
                cut_h = np.random.randint(low=0, high=img_shuffle.shape[0] - self.heatmap_size[1])
            
                img_shuffle = img_shuffle[cut_h:cut_h+self.heatmap_size[1], cut_w:cut_w+self.heatmap_size[0]]
                label_shuffle = label_shuffle[cut_h:cut_h+self.heatmap_size[1], cut_w:cut_w+self.heatmap_size[0]]
            

            cut_w = np.random.randint(low=0, high=img_ded.shape[1] - self.heatmap_size[0])
            cut_h = np.random.randint(low=0, high=img_ded.shape[0] - self.heatmap_size[1])
            img_ded = img_ded[cut_h:cut_h+self.heatmap_size[1], cut_w:cut_w+self.heatmap_size[0]]
            label_ded = label_ded[cut_h:cut_h+self.heatmap_size[1], cut_w:cut_w+self.heatmap_size[0]]
        else:
            img_ded = self.cut_img(img_ded)
            label_ded = self.cut_img(label_ded)
            img_shuffle = self.cut_img(img_shuffle)
            label_shuffle = self.cut_img(label_shuffle)
        
        label_ded = np.expand_dims(label_ded, axis = 2)
        label_shuffle = np.expand_dims(label_shuffle, axis = 2)
        

        img_shuffle = (img_shuffle.astype(np.float32) / 255.)
        # img = (img - self.mean) / self.std
        img_shuffle = torch.from_numpy(img_shuffle.transpose(2, 0, 1))

        label_shuffle = label_shuffle.transpose(2, 0, 1)
        label_shuffle = torch.from_numpy(label_shuffle)

        img_ded = (img_ded.astype(np.float32) / 255.)
        # img = (img - self.mean) / self.std
        img_ded = torch.from_numpy(img_ded.transpose(2, 0, 1))

        label_ded = (label_ded.astype(np.float32) / 255.)
        label_ded = label_ded.transpose(2, 0, 1)
        label_ded = torch.from_numpy(label_ded)
        return img_shuffle, label_shuffle, img_ded, label_ded
    

    def cut_img(self, img, cont = 16):
        if len(img.shape) == 2:
            h, w = img.shape
            img = img[0:h-h%cont, 0:w-w%cont]
        else:
            h, w, c = img.shape
            img = img[0:h-h%cont, 0:w-w%cont, :]
        return img

        





