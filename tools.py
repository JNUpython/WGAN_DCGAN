# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 22:05
# @Author  : kean
# @Email   : 
# @File    : utils.py
# @Software: PyCharm

import os
import numpy as np
import scipy.misc as misc
import tensorflow as tf


def image2array(path, re_width=None, re_height=None, grayscale=False):
    if grayscale:
        image = misc.imread(path, flatten=True).astype(np.float32)
    if not grayscale:
        image = misc.imread(path, flatten=False).astype(np.float32)
    # print(image.shape)
    width, high = image.shape[0], image.shape[1]
    if re_height or re_height:
        image = misc.imresize(image, size=[re_width, re_height])
    return image


def array2image(array, path):
    misc.imsave(path, array)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images + 1.) / 2.

def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w



def batch_norm(x, name, train, reuse=False):
    # 关于tensorflow有3种BN方法：https://www.cnblogs.com/hrlnw/p/7227447.html
    # return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=train, scope=name)
    with tf.variable_scope(name, reuse=reuse):
        return tf.layers.batch_normalization(inputs=x, training=train, scale=True) # inference mode training ：False



def next_batch(num_sample, batch_size, n_critc=1):
    num_batches = num_sample // (batch_size * n_critc)
    indices = np.arange(0, num_batches * batch_size * n_critc)
    np.random.shuffle(indices)
    for i in range(num_batches - 1):
        yield indices[i * batch_size * n_critc: (i + 1) * batch_size * n_critc]


if __name__ == '__main__':
    paths = [os.path.join("portraits/portraits", _) for _ in os.listdir("portraits/portraits")]
    print(len(paths))
    sample = paths[0]
    print(sample)
    image2array(sample, grayscale=True)
