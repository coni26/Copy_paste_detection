import numpy as np
from matplotlib import pyplot as plt
import mahotas
from tqdm import tqdm
from itertools import repeat
import os
from PIL import Image

def extract_patch(img, x, y, p):
    return img[x : x + p, y : y + p]

def extract_patch_zernike(img, x, y, p):
    return img[x, y, :]

def image_to_zernike(img, p):
    radius = p
    h, w = img.shape[0] - p, img.shape[1] - p
    img_zernike = np.zeros((h, w, 25))
    for x in range(h):
        for y in range(w):
            img_zernike[x, y] = mahotas.features.zernike_moments(img[x : x + p, y : y + p], radius, degree=8)
    return img_zernike

def image_to_zernike_3c(img, p):
    return np.swapaxes(np.swapaxes(np.array([image_to_zernike(img[:,:,i], p) for i in range(3)]), 0,2), 0, 1)

def l1(p1, p2):
    return np.sum(abs(p1 - p2))

def l1_3c(p1, p2):
    return np.sum(abs(p1 - p2))

def propagation(img, nnf_x, nnf_y, odd, p, loss_fn=l1):
    h, w = img.shape[0], img.shape[1]
    if odd:
        for x in range(h-2,-1,-1):
            for y in range(w-2,-1,-1):
                best_d = nnf_x[x, y], nnf_y[x, y]
                best_l = loss_fn(extract_patch_zernike(img, x, y, p), extract_patch_zernike(img, x + nnf_x[x, y], y + nnf_y[x, y], p))
                if 0 <= x + nnf_x[x+1, y] < h and 0 <= y + nnf_y[x+1, y] < w and loss_fn(extract_patch_zernike(img, x, y, p), extract_patch_zernike(img, x + nnf_x[x+1, y], y + nnf_y[x+1, y], p)) < best_l:
                    best_l = loss_fn(extract_patch_zernike(img, x, y, p), extract_patch_zernike(img, x + nnf_x[x+1, y], y + nnf_y[x+1, y], p))
                    best_d = nnf_x[x+1, y], nnf_y[x+1, y]
                if 0 <= x + nnf_x[x, y+1] < h and 0 <= y + nnf_y[x, y+1] < w and loss_fn(extract_patch_zernike(img, x + best_d[0], y + best_d[1], p), extract_patch_zernike(img, x + nnf_x[x, y+1], y + nnf_y[x, y+1], p)) < best_l:
                    best_l = loss_fn(extract_patch_zernike(img, x, y, p), extract_patch_zernike(img, x + nnf_x[x, y+1], y + nnf_y[x, y+1], p))
                    best_d = nnf_x[x, y+1], nnf_y[x, y+1]
                nnf_x[x, y], nnf_y[x, y] = best_d
    else:
        for x in range(1, h):
            for y in range(1,w):
                best_d = nnf_x[x, y], nnf_y[x, y]
                best_l = loss_fn(extract_patch_zernike(img, x, y, p), extract_patch_zernike(img, x + nnf_x[x, y], y + nnf_y[x, y], p))
                if 0 <= x + nnf_x[x-1, y] < h and 0 <= y + nnf_y[x-1, y] < w and loss_fn(extract_patch_zernike(img, x, y, p), extract_patch_zernike(img, x + nnf_x[x-1, y], y + nnf_y[x-1, y], p)) < best_l:
                    best_l = loss_fn(extract_patch_zernike(img, x, y, p), extract_patch_zernike(img, x + nnf_x[x-1, y], y + nnf_y[x-1, y], p))
                    best_d = nnf_x[x-1, y], nnf_y[x-1, y]
                if 0 <= x + nnf_x[x, y-1] < h and 0 <= y + nnf_y[x, y-1] < w and loss_fn(extract_patch_zernike(img, x + best_d[0], y + best_d[1], p), extract_patch_zernike(img, x + nnf_x[x, y-1], y + nnf_y[x, y-1], p)) < best_l:
                    best_l = loss_fn(extract_patch_zernike(img, x, y, p), extract_patch_zernike(img, x + nnf_x[x, y-1], y + nnf_y[x, y-1], p))
                    best_d = nnf_x[x, y-1], nnf_y[x, y-1]
                nnf_x[x, y], nnf_y[x, y] = best_d
    return nnf_x, nnf_y


def random_search(img, nnf_x, nnf_y, p, L=8, loss_fn=l1):
    dir_random = [(-1,-1), (0,1), (1,0), (0,-1), (-1,0), (1,-1), (-1,1), (1,1)]
    h, w = img.shape[0], img.shape[1]
    for x in range(h):
        for y in range(w):
            best_d = nnf_x[x, y], nnf_y[x, y]
            best_l = loss_fn(extract_patch_zernike(img, x, y, p), extract_patch_zernike(img, x + nnf_x[x, y], y + nnf_y[x, y], p))
            for i in range(L):
                dir_ = dir_random[np.random.randint(8)]
                while not(0 <= x + nnf_x[x ,y] + 2 ** i * dir_[0] < h) or not(0 <= y + nnf_y[x ,y] + 2 ** i * dir_[1] < w) or (nnf_x[x ,y] + 2 ** i * dir_[0] == 0 and nnf_y[x ,y] + 2 ** i * dir_[1]==0):
                    dir_ = dir_random[np.random.randint(8)]
                if loss_fn(extract_patch_zernike(img, x, y, p), extract_patch_zernike(img, int(x + nnf_x[x, y] + 2 ** i * dir_[0]), int(y + nnf_y[x, y] + 2 ** i * dir_[1]), p)) < best_l:
                    best_d = nnf_x[x ,y] + 2 ** i * dir_[0], nnf_y[x ,y] + 2 ** i * dir_[1]
            nnf_x[x, y], nnf_y[x, y] = best_d
    return nnf_x, nnf_y



def symetry_comparison(img, nnf_x, nnf_y, p, loss_fn=l1):
    h, w = img.shape[0], img.shape[1]
    for x in range(h):
        for y in range(w):
            x_, y_ = x + nnf_x[x, y], y + nnf_y[x, y]
            best_l = loss_fn(extract_patch_zernike(img, x_, y_, p), extract_patch_zernike(img, x_ + nnf_x[x_, y_], y_ + nnf_y[x_, y_], p))
            if loss_fn(extract_patch_zernike(img, x_, y_, p), extract_patch_zernike(img, x, y, p)) < best_l:
                nnf_x[x_, y_] = - nnf_x[x, y]
                nnf_y[x_, y_] = - nnf_y[x, y]
    return nnf_x, nnf_y


def remove_0(nnf_x, nnf_y, p):
    for i in range(nnf_x.shape[0]):
        for j in range(nnf_x.shape[1]):
            while nnf_x[i,j] == nnf_y[i,j] == 0:
                nnf_x[i,j] = np.random.randint(-i, nnf_x.shape[0] - i - p)
    return nnf_x, nnf_y

def remove_small(nnf_x, nnf_y, p):
    for i in range(nnf_x.shape[0]):
        for j in range(nnf_x.shape[1]):
            while (abs(nnf_x[i,j]) + abs(nnf_y[i,j])) <= 8:
                nnf_x[i,j] = np.random.randint(-i, nnf_x.shape[0] - i - p)
                nnf_y[i,j] = np.random.randint(-j, nnf_x.shape[1] - j - p)
    return nnf_x, nnf_y



def patchmatch(img, n_iters=100, p=8):
    img_zernike = image_to_zernike_3c(img, p)
    h, w = img.shape[0] - p, img.shape[1] - p
    x = np.array(np.linspace(0,h-1,h))
    y = np.array(np.linspace(0,w-1,w))
    yy, xx = np.meshgrid(y, x)
    xx = xx.astype(int)
    yy = yy.astype(int)
    nnf_x = np.random.randint(-xx, h-xx-p)
    nnf_y = np.random.randint(-yy, w-yy-p)
    nnf_x, nnf_y = remove_small(nnf_x, nnf_y, p) # to avoid null move
    for i in tqdm(range(n_iters)):
        nnf_x, nnf_y = random_search(img_zernike, nnf_x, nnf_y, p)
        nnf_x, nnf_y = propagation(img_zernike, nnf_x, nnf_y, i%2, p)
        nnf_x, nnf_y = symetry_comparison(img_zernike, nnf_x, nnf_y, p)
        nnf_x, nnf_y = remove_small(nnf_x, nnf_y, p)
    return nnf_x, nnf_y, img_zernike



def diff_nn(img, img_zernike, nnf_x, nnf_y, p=8, loss_fn=l1):
    h, w = nnf_x.shape
    diff = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            diff[x, y] = loss_fn(extract_patch_zernike(img_zernike, x, y, p), extract_patch_zernike(img_zernike, x + nnf_x[x, y], y + nnf_y[x, y], p))
    return diff      
    
    
    
def std_patch(img, p=8):
    h, w = img.shape[0] - p, img.shape[1] - p 
    std = np.zeros((h,w))
    for x in range(h):
        for y in range(w):
            std[x, y] = np.mean(np.std(extract_patch(img, x, y, p), axis=(0,1)))
    return std
            
            
def f_measure(pred, labels):
    tp = np.multiply(pred, (pred == labels)).sum()
    fn = np.multiply(1 - pred, (pred != labels)).sum()
    fp = np.multiply(pred, (pred != labels)).sum()
    return 2 * tp / (2 * tp + fn + fp)           
            
                
def find_neigh(ind, h, w):
    if ind[0] > 0:
        if ind[0] < h - 1:
            if ind[1] > 0:
                if ind[1] < w - 1:
                    return [(ind[0]-1,ind[1]), (ind[0]+1,ind[1]), (ind[0],ind[1]-1), (ind[0],ind[1]+1)]
                else:
                    return [(ind[0]-1,ind[1]), (ind[0]+1,ind[1]), (ind[0],ind[1]-1)]
            else:
                return [(ind[0]-1,ind[1]), (ind[0]+1,ind[1]), (ind[0],ind[1]+1)]
        else:
            if ind[1] > 0:
                if ind[1] < w - 1:
                    return [(ind[0]-1,ind[1]), (ind[0],ind[1]-1), (ind[0],ind[1]+1)]
                else:
                    return [(ind[0]-1,ind[1]), (ind[0],ind[1]-1)]
            else:
                return [(ind[0]-1,ind[1]), (ind[0],ind[1]+1)]
    else:
        if ind[1] > 0:
            if ind[1] < w - 1:
                return [(ind[0]+1,ind[1]), (ind[0],ind[1]-1), (ind[0],ind[1]+1)]
            else:
                return [(ind[0]+1,ind[1]), (ind[0],ind[1]-1)]
        else:
            return [(ind[0]+1,ind[1]), (ind[0],ind[1]+1)]      

                                     
                                     
                                     
                                     
                                     
