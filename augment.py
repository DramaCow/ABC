import numpy as np
import os
import cv2
from glob import iglob
from os.path import basename, normpath, join

"""
A standalone script for augmenting Omniglot image data.
"""

def load_alphabets(path):            
    return { basename(normpath(apath)) : [[cv2.imread(filename, cv2.IMREAD_GRAYSCALE) for filename in iglob(join(cpath, '*'))] for cpath in iglob(join(apath, '*'))] for apath in iglob(join(path, '*')) }

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__=="__main__":
    DATAPATH = '_data'

    # training data
    background = load_alphabets(join(os.getcwd(), DATAPATH, 'images_background'))
    rpath = mkdir(join(DATAPATH, 'images_background_test'))
    for name, alphabet in background.items():
        apath = mkdir(join(rpath, name))
        for i, character in enumerate(alphabet):
            cpath1 = mkdir(join(apath, '%03d' % (4 * i + 0)))
            cpath2 = mkdir(join(apath, '%03d' % (4 * i + 1)))
            cpath3 = mkdir(join(apath, '%03d' % (4 * i + 2)))
            cpath4 = mkdir(join(apath, '%03d' % (4 * i + 3)))
            for j, image in enumerate(character):
                image = cv2.resize(~image, (28, 28), interpolation=cv2.INTER_AREA)
                cv2.imwrite(join(cpath1, '%03d.png' % j), image)
                cv2.imwrite(join(cpath2, '%03d.png' % j), cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
                cv2.imwrite(join(cpath3, '%03d.png' % j), cv2.rotate(image, cv2.ROTATE_180))
                cv2.imwrite(join(cpath4, '%03d.png' % j), cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    
    # validation data
    evaluation = load_alphabets(join(os.getcwd(), DATAPATH, 'images_evaluation'))
    rpath = mkdir(join(DATAPATH, 'images_evaluation_test'))
    for name, alphabet in evaluation.items():
        apath = mkdir(join(rpath, name))
        for i, character in enumerate(alphabet, 1):
            cpath = mkdir(join(apath, '%03d' % i))
            for j, image in enumerate(character, 1):
                image = cv2.resize(~image, (28, 28), interpolation=cv2.INTER_AREA)
                cv2.imwrite(join(cpath, '%03d.png' % j), image)