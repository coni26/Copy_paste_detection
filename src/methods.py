import numpy as np
from matplotlib import pyplot as plt
import mahotas
from tqdm import tqdm
from itertools import repeat
import os
from PIL import Image
from scipy.stats import binom
from scipy.ndimage import convolve, binary_dilation



def method_diff(nnf_x, nnf_y, diff, std, img, threshold=0.05):
    h, w = diff.shape
    img_copy = np.zeros((h, w, 3), dtype=int)
    pred = np.zeros((h+8, w+8), dtype=int)
    for x in range(diff.shape[0]):
            for y in range(diff.shape[1]):
                if diff[x, y] < threshold:
                    img_copy[x, y] = img[x,y]
                    pred[x, y] = 1
    return pred, img_copy

def method_diff_product(nnf_x, nnf_y, diff, std, img, threshold=0.1):
    h, w = diff.shape
    img_copy = np.zeros((h, w, 3), dtype=int)
    pred = np.zeros((h+8, w+8), dtype=int)
    for x in range(diff.shape[0]):
            for y in range(diff.shape[1]):
                if diff[x, y] * min(std[x,y], 3) < threshold:
                    img_copy[x, y] = img[x,y]
                    pred[x, y] = 1
    return pred, img_copy

def method_diff_std(nnf_x, nnf_y, diff, std, img, threshold_diff=0.05, threshold_std=5):
    h, w = diff.shape
    img_copy = np.zeros((h, w, 3), dtype=int)
    pred = np.zeros((h+8, w+8), dtype=int)
    for x in range(diff.shape[0]):
            for y in range(diff.shape[1]):
                if diff[x, y] < threshold_diff and std[x,y] > threshold_std:
                    img_copy[x, y] = img[x,y]
                    pred[x, y] = 1
    return pred, img_copy


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

def method_propagation(nnf_x, nnf_y, diff, std, img):
    h, w = diff.shape
    img_copy = np.zeros((h, w, 3), dtype=int)
    pred = np.zeros((h+8, w+8), dtype=int)
    seen = np.zeros((h+8, w+8), dtype=int)
    to_explore = []
    for x in range(h):
            for y in range(w):
                if diff[x, y] < 0.01 and std[x,y] > 5:
                    seen[x, y] = 1
                    to_explore.append((x,y))
    while len(to_explore) > 0:
        ind = to_explore.pop()
        neighs = find_neigh(ind, h, w)
        pred[ind] = 1
        img_copy[ind] = img[ind]
        for neigh in neighs:
            if not(seen[neigh]) and nnf_x[neigh] == nnf_x[ind] and nnf_y[neigh] == nnf_y[ind]:
                to_explore.append(neigh)
            seen[neigh] = 1
    return pred, img_copy

def method_propagation_dilated(nnf_x, nnf_y, diff, std, img):
    h, w = diff.shape
    img_copy = np.zeros((h, w, 3), dtype=int)
    pred = np.zeros((h+8, w+8), dtype=int)
    seen = np.zeros((h+8, w+8), dtype=int)
    to_explore = []
    for x in range(h):
            for y in range(w):
                if diff[x, y] < 0.01 and std[x,y] > 5:
                    seen[x, y] = 1
                    to_explore.append((x,y))
    while len(to_explore) > 0:
        ind = to_explore.pop()
        neighs = find_neigh(ind, h, w)
        pred[ind] = 1
        img_copy[ind] = img[ind]
        for neigh in neighs:
            if not(seen[neigh]) and nnf_x[neigh] == nnf_x[ind] and nnf_y[neigh] == nnf_y[ind]:
                to_explore.append(neigh)
            seen[neigh] = 1
    pred = binary_dilation(pred, iterations=7)
    return pred, img_copy



def in_bounds(a, b, bounds):
    for bound in bounds:
        if bound[0] <= a <= bound[1] and bound[2] <= b <= bound[3]:
            return True
    return False

def method_histo(nnf_x, nnf_y, diff, std, img, threshold=100):
    h, w = diff.shape
    img_copy = np.zeros((h, w, 3), dtype=int)
    pred = np.zeros((h+8, w+8), dtype=int)
    ind = np.where(abs(nnf_x.flatten()) + abs(nnf_y.flatten()) > 10)
    counts, bins1, bins2, _ = plt.hist2d(nnf_x.flatten()[ind], nnf_y.flatten()[ind], bins=200)
    ind_ = np.where(counts > threshold * np.mean(counts.flatten()))
    bounds = []
    for i in range(len(ind_[0])):
        bounds.append([bins1[ind_[0][i]], bins1[ind_[0][i]+1], bins2[ind_[1][i]], bins2[ind_[1][i]+1]])
    for x in range(h):
            for y in range(w):
                if in_bounds(nnf_x[x, y], nnf_y[x, y], bounds):
                    img_copy[x, y] = img[x,y]
                    pred[x, y] = 1
    return pred, img_copy

def method_nfa(nnf_x, nnf_y, diff, std, img, threshold=2):
    h, w = diff.shape
    mat_count = np.zeros((2 * h - 1, 2 * w - 1), dtype=int)
    img_copy = np.zeros((h, w, 3), dtype=int)
    pred = np.zeros((h+8, w+8), dtype=int)
    for x in range(h):
        for y in range(w):
            mat_count[h + nnf_x[x, y] - 1, w + nnf_y[x, y] - 1] += 1
    l_dir = []
    for x in range(mat_count.shape[0]):
        for y in range(mat_count.shape[1]):
            p = min(x + 1, 2 * h - 1 - x) * min(y + 1, 2 * w - 1 - y) / (h * w) ** 2
            if mat_count[x, y] > 64 and binom.cdf(mat_count[x, y], h * w, p) >= 1 - threshold / (4 * h * w):
                l_dir.append((x - h + 1, y - w + 1))
    for x in range(h):
            for y in range(w):
                if (nnf_x[x, y], nnf_y[x, y]) in l_dir:
                    img_copy[x, y] = img[x,y]
                    pred[x, y] = 1
    return pred, img_copy


def method_nfa_kernel(nnf_x, nnf_y, diff, std, img, threshold=2):
    h, w = diff.shape
    mat_count = np.zeros((2 * h - 1, 2 * w - 1), dtype=int)
    img_copy = np.zeros((h, w, 3), dtype=int)
    pred = np.zeros((h+8, w+8), dtype=int)
    for x in range(h):
        for y in range(w):
            mat_count[h + nnf_x[x, y] - 1, w + nnf_y[x, y] - 1] += 1
    l_dir = []
    for x in range(mat_count.shape[0]):
        for y in range(mat_count.shape[1]):
            p = min(x + 1, 2 * h - 1 - x) * min(y + 1, 2 * w - 1 - y) / (h * w) ** 2
            if mat_count[x, y] > 64 and binom.cdf(mat_count[x, y], h * w, p) >= 1 - threshold / (4 * h * w):
                l_dir.append((x - h + 1, y - w + 1))
    for x in range(h):
            for y in range(w):
                if (nnf_x[x, y], nnf_y[x, y]) in l_dir:
                    pred[x, y] = 1
    kernel = np.ones((64,64), dtype=int)
    c = convolve(pred, kernel, mode='constant')
    for x in range(h):
            for y in range(w):
                if pred[x, y] and c[x,y] > 512:
                    img_copy[x, y] = img[x,y]
                else:
                    pred[x, y] = 0
    return binary_dilation(pred, iterations=7), img_copy
    
