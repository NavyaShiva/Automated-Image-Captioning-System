from __future__ import print_function
import torch
import os
from os import listdir
import os.path
import errno
import numpy as np
import random
import sys
import nltk
import torch.utils.data as data
from PIL import Image
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

#import torch.utils.data as data
#from torchvision.datasets.utils import download_url, check_integrity

class DATALOAD():
    def __init__(self,root,url, captions,vocab, split='train',
                 transform=None, download=False):
        self.root = root#os.path.expanduser(root)
        self.trans = transform
        self.split = split  # train, val, or test
        self.url= url
        self.data=url
        self.captions=captions
        self.vocab=vocab

        if download:
            self.download()
        self.transform()

        if self.split == 'train': 
           self.train_data=self.data
           self.train_captions=self.captions
           self.train_url=self.url

        elif self.split == 'val':
           self.val_data=self.data
           self.val_captions=self.captions
           self.val_url=self.url

        elif self.split == 'test':
           self.test_data=self.data
           self.test_captions=self.captions
           self.test_url=self.url
    def __len__(self):
        #print("am i here ")
        if self.split == 'train': 
           return len(self.train_data)
        elif self.split == 'val':
           #print(len(self.val_data), len(self.val_captions))
           return len(self.val_data)
        elif self.split == 'test':
           return len(self.test_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #print("hahaha")
        if self.split == 'train':
            img,caption,url = self.train_data[index], self.train_captions[index],self.train_url[index]
            #print(index,self.train_data[index],self.train_captions[index])
        elif self.split == 'val':
           img,caption,url = self.val_data[index], self.val_captions[index],self.val_url[index]
           #print(" while enumerate",caption) 
        elif self.split == 'test':
           img, caption,url = self.test_data[index], self.test_captions[index],self.test_url[index]
        
        if self.trans is not None:
            img = self.trans(img)
        if  self.split !='test':  
            caption=torch.LongTensor(caption)
            
        return img,caption,url
    
    def download(self):
        # importing and extracting flickr images
        import tarfile
        tf = tarfile.open("/content/drive/My Drive/Colab Notebooks/576_project/shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar")
        tf.extractall(path=".")
       # importing the tokenized captions
        tf2 = tarfile.open("/content/drive/My Drive/Colab Notebooks/576_project/data/flickr30k.tar.gz")
        tf2.extractall(path=".")
        
    def transform(self):
        i=0
        dataset=[]
        captions=[]
      
        captions=loadcaptions(self.captions)
        
        index=[] 
        urls=[]
        print( ' we are here check 1')
        
        for img in self.url:
            #print(img)
            for a in range(len(captions)):
                for key,value in captions[a].items():
                    if key==img:
                        index.append(a)
            
            url1=self.root+"/images/"+img
            urls.append(url1)
            im=Image.open(self.root+"/images/"+img)
            dataset.append(im)
            
        print("check 2",urls)   
        self.url=urls
      
        new_captionlist=[]
        for k in index:
            new_captionlist.append(self.captions[k])
        
        cap2=[]
        cap2=loadcaptions(new_captionlist)
        #print(cap2)
        y=[]
       	images=[]
        target=[]
        new_urls=[]
         
        num=5
        for i in range(len(cap2)):
              if self.split !='test':
                 images.extend([dataset[i]]*num)
                 new_urls.extend([urls[i]]*num)
                 for j,k in cap2[i].items():
                     for m in range(5):
                         y.append(k[m])
              else:
                  #print(self.split)
                  images=dataset
                  for j,k in cap2[i].items():
                      y.append(k)
        print(self.split,len(y),y)
        if self.split !='test':
          self.url=new_urls
          for i in range(len(y)):            
              token = nltk.tokenize.word_tokenize(y[i].lower())
        #print(token)
              vec = []
              vocab=self.vocab
              vec.append(vocab.get_id('<start>'))
              vec.extend([vocab.get_id(word) for word in token])
              vec.append(vocab.get_id('<end>'))   
              target.append(vec)
        else:
              target=y
        
        self.data=images  
        self.captions=target
            
        
        return images,target

def collate_fn(data):
    # Sort a data list by caption length (descending order).
      #print("did you meet me here ")
      data.sort(key=lambda x: len(x[1]), reverse=True)
      images, captions,url = zip(*data)
      #print("check 7",captions, url)
    # Merge images (from tuple of 3D tensor to 4D tensor).
      images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
      lengths = [len(cap) for cap in captions]
      #print(lengths)
      targets = torch.zeros(len(captions), max(lengths)).long()
      for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end] 
      #print(targets)        
      return images, targets, lengths

def val_collate(data):
    images,captions,url=zip(*data)
    images = torch.stack(images, 0)
    return images,captions,url
"""
def tokenize(vocab,imgcap):
        
        final=[]
        for i in range(len(imgcap)):
            final.append(captions2ind(vocab,imgcap[i]))   
        return final

def captions2ind(vocab, caption):
        token = nltk.tokenize.word_tokenize(caption.lower())
        #print(token)
        vec = []
        vec.append(vocab.get_id('<start>'))
        vec.extend([vocab.get_id(word) for word in token])
        vec.append(vocab.get_id('<end>'))
        return vec 
"""
def loadcaptions(text):
        captions=[]
        import json 
        #if vocab:
        for i in  text:
          res = json.loads(i) 
          captions.append(res)
        """   
        else:
          for i in range(len(text)):
            res = json.loads(text[i]) 
            captions.append(res)
        """       
        return captions
        
 
    
        