# -*- coding: utf-8 -*-
# @Time    : 2018/10/30 20:10
# @Author  : kean
# @Email   : 
# @File    : DCGAN.py
# @Software: PyCharm

import numpy as np
import os
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib import slim

from logger import logger
from tools import batch_norm, next_batch, array2image, image2array, save_images, image_manifold_size


class DCGAN:

    def __init__(self, image_size, z_size, color_chanel=3, optimazer=tf.train.RMSPropOptimizer, check_point="./checks",
                 lr_G=0.0002, lr_D=0.0002, clip_value=0.01, wasserstein=True, stddev=0.02, batch_size=64, num_samples=100):
        self.image_size = image_size  # 真实图片的size list
        self.z_size = z_size  # G的输入 int
        self.image = tf.placeholder(dtype=tf.float32, name="image", shape=[None] + image_size + [color_chanel])
        self.z = tf.placeholder(dtype=tf.float32, name="z", shape=[None, z_size])
        self.color_chanel = color_chanel  # default： RGB
        self.optimazer_G = optimazer(lr_G)  # 两者学习速率研究
        self.optimazer_D = optimazer(lr_D)
        self.checks = check_point
        self.clip_value = clip_value
        self.stddev = stddev
        self.batch_size = batch_size
        self.num_samples = num_samples
        # 初始化模型
        self.G = self.generator(self.z, reuse=False, train=True)
        self.D_logits = self.discriminator(self.image, reuse=False, train=True)  # 调用两次dis reuse？
        self.G_logits = self.discriminator(self.G, reuse=True, train=True)

        self.G_sample = self.generator(self.z, reuse=True, train=False, num_z_input=self.num_samples)

        # 普通的定义loss,
        if not wasserstein:
            self.loss_D = tf.add(
                    tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                                    labels=tf.ones_like(self.D_logits)),
                            name="loos_D_real"),  # 正样本
                    tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G_logits,
                                                                    labels=tf.zeros_like(self.G_logits)),
                            name="loos_D_fake"),  # 负样本
                    name="loss_D"
            )
            self.loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.G_logits, labels=tf.ones_like(self.G_logits)), name="loss_G")  # 对于G来说都是正样本
        if wasserstein:
            self.loss_D = - tf.add(
                    tf.reduce_mean(self.D_logits, name="loos_D_real"),  # 正样本
                    -tf.reduce_mean(self.G_logits, name="loos_D_fake"),  # 负样本
                    name="loss_D"
            )
            self.loss_G = - tf.reduce_mean(self.G_logits, name="loss_G")  # 对于G来说都是正样本

        self.optimazer_D = self.optimazer_D.minimize(self.loss_D)
        if wasserstein:
            with tf.control_dependencies([self.optimazer_D]):
                #  先执行上面的内容 再执行包含在下面的内容
                d_var = tf.trainable_variables(scope="DIS")
                logger.info("cliped vars")
                logger.info(d_var)
                # assign 人在 计算过程中对 DIS设计的所有变量进行值域限制，
                # group 对返回一个 operation 执行这些操作 When this op finishes, all ops in inputs， An Operation that executes all its inputs.
                self.optimazer_D = tf.group(*(tf.assign(var, tf.clip_by_value(var, -clip_value, clip_value)) for var in d_var))
                # 执行 self.optimazer_D  这里等于先求梯度 更新权重 然后再对权重进行clip操作

        self.optimazer_G = self.optimazer_G.minimize(self.loss_G)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        # self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.session = tf.Session()


        # show
        train_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(train_vars, print_info=True)
        # initial
        self.session.run(tf.global_variables_initializer())  # 所有变量已经初始化

    def generator(self, z_input, num_z_input=None, num_layers=4, stride=2, z_first_output_deep=512, reuse=True, train=True):
        """
        :param layers_size:
        :return:
        """
        # try:
        #     num_sample = z_input.get_shape()[0] if z_input.get_shape()[0] == tf.Dimension(None) else 64  # 解决bug
        # except AttributeError:
        #     num_sample = z_input.shape[0]
        # logger.info(z_input.get_shape()[0]._value)
        # num_sample = z_input.get_shape()[0]._value if z_input.get_shape()[0]._value else 64  # 解决bug
        # logger.info(num_sample)
        if num_z_input == None:
            num_z_input = self.batch_size

        with tf.variable_scope("GEN", reuse=reuse):
            hidden_output_size = [np.ceil(np.array(self.image_size) / (stride ** i)).astype(np.int).tolist()
                                  for i in range(0, num_layers + 1)]  # 所有层的output size
            hidden_output_size = list(reversed(hidden_output_size))
            # logger.info(
            #         hidden_output_size)  # [array([6, 6]), array([12, 12]), array([24, 24]), array([48, 48]), array([96, 96])]
            # 输出层采用全连接层: 线性全连接层
            h0_size = (hidden_output_size[0][0] * hidden_output_size[0][
                1]) * z_first_output_deep  # width * height * chanel
            # weight_0 = tf.Variable(initial_value=tf.random_normal(
            #         shape=[self.z_size, h0_size],
            #         stddev=0.2,
            #         dtype=tf.float32),
            #         name="weight_input_layer"
            # )
            # bias_0 = tf.Variable(initial_value=np.zeros(shape=[h0_size]), dtype=tf.float32, name="bias_input_layer")
            weight_0 = tf.get_variable(
                    shape=[self.z_size, h0_size],
                    initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                    dtype=tf.float32,
                    name="h0_weight"
            )
            bias_0 = tf.get_variable(shape=[h0_size],
                                     dtype=tf.float32,
                                     name="h0_bias",
                                     initializer=tf.zeros_initializer()
                                     )
            h0 = tf.matmul(z_input, weight_0) + bias_0
            h0_reshape = tf.reshape(h0, shape=[num_z_input, hidden_output_size[0][0], hidden_output_size[0][1], z_first_output_deep],
                            name="h0_reshape")  # reshape project
            # logger.info(h0.get_shape())  # (?, 6, 6, 1024)
            h0_bn = batch_norm(h0_reshape, name="h0_bn", train=train, reuse=reuse)
            h0_lrelu = tf.nn.leaky_relu(h0_bn, alpha=0.2, name="h0_lrelu")
            hm_lrelu = None
            # 定义反卷积层
            for i in range(1, len(hidden_output_size) - 1):
                with tf.variable_scope("deconv_%d" % i, reuse=reuse):
                    # 第一层 不许卷积， 最后一层不需要BN
                    # width, height = hidden_output_size[i]
                    filter_shape = [5, 5,  # 认为设定 5 width height
                                    int(z_first_output_deep / 2 ** (i)),  # output chanel
                                    int(z_first_output_deep / 2 ** (i - 1))]  # input chanel
                    logger.info(filter_shape)
                    # bug：必须指定batch的具体数量
                    # output_shape = [z_input.get_shape()[0]._value, hidden_output_size[i+1][0], hidden_output_size[i+1][0], filter_shape[-1]]
                    output_shape = [num_z_input, hidden_output_size[i][0], hidden_output_size[i][0], filter_shape[-2]]
                    # logger.info(output_shape)
                    # filter = tf.Variable(
                    #         initial_value=tf.random_normal(
                    #                 dtype=tf.float32,
                    #                 shape=filter_shape
                    #         ),
                    #         name="deconv_filter_%d" % i
                    # )  # 反卷积操作的共享卷积权重矩阵
                    # bias = tf.get_variable(name="deconv_bias_%d" % i,
                    #                        shape=[z_first_output_deep / 2 ** (i)],
                    #                        dtype=tf.float32,
                    #                        initializer=tf.zeros_initializer()
                    #                        )  # filter对应的偏置，shape取决于filter的output chanel 即filter的数量
                    filter = tf.get_variable(
                            dtype=tf.float32,
                            shape=filter_shape,
                            initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                            name="deconv%d_filter_weight" % i
                    )  # 反卷积操作的共享卷积权重矩阵
                    bias = tf.get_variable(name="deconv%d_filter_bias" % i,
                                           shape=[z_first_output_deep / 2 ** (i)],  # output chanel
                                           dtype=tf.float32,
                                           initializer=tf.zeros_initializer()
                                           )  # filter对应的偏置，shape取决于filter的output chanel 即filter的数量
                    if hm_lrelu == None:
                        hm_lrelu = h0_lrelu
                    hm = tf.add(tf.nn.conv2d_transpose(value=hm_lrelu,
                                                       filter=filter,
                                                       strides=[1, stride, stride, 1],
                                                       output_shape=output_shape
                                                       ),
                                bias, name="h%d" % i
                                )
                    hm_bn = batch_norm(hm, name="h%d_bn" % i, train=train, reuse=reuse)
                    hm_lrelu = tf.nn.leaky_relu(hm_bn, name="h%d_lrelu" % i)
                    # logger.info(hm.get_shape())

            # 定义输出层
            # filter = tf.Variable(
            #         initial_value=tf.random_normal(
            #                 shape=[5, 5, self.color_chanel, hm_lrelu.get_shape()[-1]._value],  # 获取Dimension 的值
            #                 dtype=tf.float32
            #         ),
            #         name="weight_output_filter"
            # )
            # logger.info(filter.get_shape())
            # bias = tf.Variable(initial_value=np.zeros(shape=[self.color_chanel]), dtype=tf.float32,
            #                    name="bias_output_filter")
            filter = tf.get_variable(
                    shape=[5, 5, self.color_chanel, hm_lrelu.get_shape()[-1]._value],  # 获取Dimension 的值
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                    name="output_filter_weight"
            )
            # logger.info(filter.get_shape())
            bias = tf.get_variable(shape=[self.color_chanel], dtype=tf.float32,
                                   name="output_filter_bias",
                                   initializer=tf.truncated_normal_initializer(stddev=self.stddev)
                                   )
            output_shape = [num_z_input, hidden_output_size[-1][0], hidden_output_size[-1][1], self.color_chanel]
            # logger.info(output_shape)  输出层的激活函数为tanh
            out_put = tf.nn.tanh(
                    tf.add(tf.nn.conv2d_transpose(
                            value=hm_lrelu, filter=filter, strides=[1, stride, stride, 1],
                            output_shape=output_shape),
                            bias,
                    ),
                    name="out_put"
            )  # 没有BN， active func： tanh
            # logger.info(out_put.get_shape())
            return out_put

    def discriminator(self, input, num_input=None, layer_size=4, stride=2, d_first_output_deep=64, reuse=True,
                      train=True):
        """
        :param layer_size:
        :return:
        """
        # D leak —relu， 没有pool层， 没有BN层
        if num_input == None:
            num_input = self.batch_size
        with tf.variable_scope("DIS", reuse=reuse) as dis:
            # tf.get_variable_scope().reuse_variables()
            # 定义没有池化层卷积层
            input_ = input
            for i in range(layer_size):
                with tf.variable_scope("conv_%d" % i, reuse=reuse) as conv:
                    # conv.reuse_variables()
                    # tf.get_variable_scope().reuse_variables()
                    filter = tf.get_variable(name='conv%d_filter_weight' % i,
                                             shape=[5, 5,  # 认为设定： 5
                                                    input_.get_shape()[-1]._value,  # input chanel
                                                    d_first_output_deep * (2 ** i)],  # output chanel
                                             initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                                             dtype=tf.float32
                                             )
                    # logger.info(filter.get_shape())
                    bias = tf.get_variable(name='conv%d_filter_bias' % i,
                                           shape=[d_first_output_deep * (2 ** i)],  # filter 的数量不断在提高
                                           initializer=tf.constant_initializer(0.0),
                                           dtype=tf.float32
                                           )
                    # if self.clip_value:
                    #     filter = tf.clip_by_value(filter, clip_value_min=-self.clip_value,
                    #                               clip_value_max=self.clip_value)
                    #     bias = tf.clip_by_value(bias, clip_value_min=-self.clip_value, clip_value_max=self.clip_value)
                    conv = tf.nn.conv2d(input_, filter=filter, strides=[1, stride, stride, 1], padding='SAME')
                    hm = tf.nn.bias_add(conv, bias, name="h%d" % i)
                    hm_bn = batch_norm(hm, name="h%d_bn" % i, train=train, reuse=reuse)  # 第一层的输入并没有采用BN
                    # input_ = tf.nn.leaky_relu(hm_bn, name="h%d_lrelu" % i)
                    input_ = tf.nn.relu(hm_bn, name="h%d_lrelu" % i)

                    logger.info(input_.get_shape())
            # 定义输出层： 输出采用全连接s
            output_layer_input = tf.layers.flatten(input_, name="output_layer_input")
            output_layer_input = tf.reshape(output_layer_input,
                                            shape=[num_input, output_layer_input.get_shape()[-1]._value])
            logger.info(output_layer_input.get_shape())
            weight = tf.get_variable(name='output_weitht',
                                     shape=[output_layer_input.get_shape()[-1]._value, 1],
                                     initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                                     dtype=tf.float32
                                     )
            bias = tf.get_variable(name='output_bias',
                                   shape=[1],
                                   initializer=tf.constant_initializer(0.0),
                                   dtype=tf.float32
                                   )
            # if self.clip_value:
            #     weight = tf.clip_by_value(weight, clip_value_min=-self.clip_value, clip_value_max=self.clip_value)
            #     bias = tf.clip_by_value(bias, clip_value_min=-self.clip_value, clip_value_max=self.clip_value)
            logits = tf.add(tf.matmul(output_layer_input, weight), bias, name="logits")
            logger.info(logits.get_shape())
            return logits  # 输出不采用是个sigmoid

    def train(self, images_path, n_critc=5, epoches=25, restore=False):
        assert isinstance(images_path, np.ndarray)
        logger.info("training...")
        # restore
        saver = tf.train.Saver()
        if restore:
            saver.restore(self.session, self.checks)

        z_samples = np.random.uniform(size=[self.num_samples, self.z_size]).astype(np.float32)  # 用生成num_sample 个随机样本
        num_images = len(images_path)
        counter = 0
        for epoch in range(epoches):
            if epoch < 25:
                n_critc = 25
            for indices in next_batch(num_images, self.batch_size, n_critc=n_critc):
                counter += 1
                batch_image = np.array([image2array(p) for p in images_path[indices]])
                batch_z = np.random.uniform(size=[self.batch_size, self.z_size]).astype(np.float32)
                # train D
                for i in range(n_critc):
                    _, loss_D, = self.session.run(
                            fetches=[self.optimazer_D, self.loss_D],
                            feed_dict={self.image: batch_image[i * self.batch_size: (i + 1) * self.batch_size],
                                       self.z: batch_z}
                    )
                # train G
                _, loss_G = self.session.run([self.optimazer_G, self.loss_G], feed_dict={self.z: batch_z})
                logger.info("epoche:(%d: %d) {loss_D: %f}-{loss_G: %f}" % (epoch, counter, loss_D, loss_G))

                # 生成样本图片
                # if np.mod(counter, 100) == 1:
                #     logger.info("save images>>>...")
                #     samples = self.session.run(fetches=[self.generator(z_samples, train=False, reuse=True)],
                #                                feed_dict={self.z: z_samples})
                #     os.makedirs("samples/%04d" % counter)
                #     for s_index, s in enumerate(samples[0]):
                #         logger.info(s.shape)
                #         array2image(s, "samples/%04d/%04d.jpg" % (counter, s_index + 1))
                if np.mod(counter, 100) == 1:
                    saver.save(self.session, self.checks)
                    samples = self.session.run(fetches=[self.G_sample], feed_dict={self.z: z_samples})[0]
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format("samples", epoch, counter))
                    logger.info('./{}/train_{:02d}_{:04d}.png'.format("samples", epoch, counter))
                    logger.info("[Sample] d_loss: %.8f, g_loss: %.8f" % (loss_D, loss_G))


if __name__ == '__main__':
    model = DCGAN(image_size=[96, 96], z_size=100, wasserstein=True)
    images_path = [os.path.join("portraits/portraits", _) for _ in os.listdir("portraits/portraits")]
    # logger.info("load images ...")
    # images = []
    # for p in tqdm(images_path):
    #     images.append(image2array(p))
    # images = np.array(images)
    model.train(np.array(images_path), restore=False)
