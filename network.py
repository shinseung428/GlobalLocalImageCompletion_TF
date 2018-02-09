import tensorflow as tf
import numpy as np

from ops import *
from architecture import *

class network():
    def __init__(self, args):

        self.batch_size = args.batch_size
        self.input_dim = args.input_dim 

        self.patch_size = args.patch_size
        self.m = args.margin

        self.alpha = args.alpha

        #prepare training data
        self.real_img, self.perturbed_img, self.mask, self.coord, self.data_count = load_train_data(args)
        self.build_model()
        self.build_loss()

        #summary
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss) 
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.input_img_sum = tf.summary.image("input_img", self.perturbed_img, max_outputs=5)
        self.real_img_sum = tf.summary.image("real_img", self.real_img, max_outputs=5)
        
        self.recon_img_sum = tf.summary.image("recon_img", self.recon_img, max_outputs=5)
        self.g_local_imgs_sum = tf.summary.image("g_local_imgs", self.g_local_imgs, max_outputs=5)
        self.r_local_imgs_sum = tf.summary.image("r_local_imgs", self.r_local_imgs, max_outputs=5)

    #structure of the model
    def build_model(self):
        self.C = tf.concat([self.perturbed_img, self.mask], -1)

        self.recon_img, self.g_nets = self.completion_net(self.C, name="completion_net")


        # self.r_local_img = tf.image.crop_to_bounding_box(self.real_img, self.coord[0]-self.m, self.coord[1]-self.m, self.patch_size+self.m*2, self.patch_size+self.m*2)
        # self.g_local_img = tf.image.crop_to_bounding_box(self.recon_img, self.coord[0]-self.m, self.coord[1]-self.m, self.patch_size+self.m*2, self.patch_size+self.m*2)
       
        self.r_local_imgs = []
        self.g_local_imgs = [] 
        for idx in range(0,self.real_img.shape[0]):
            self.r_local_imgs.append(tf.image.crop_to_bounding_box(self.real_img[idx], self.coord[idx,0]-self.m, self.coord[idx,1]-self.m, self.patch_size+self.m*2, self.patch_size+self.m*2))
            self.g_local_imgs.append(tf.image.crop_to_bounding_box(self.recon_img[idx], self.coord[idx,0]-self.m, self.coord[idx,1]-self.m, self.patch_size+self.m*2, self.patch_size+self.m*2))
        
        self.r_local_imgs = tf.convert_to_tensor(self.r_local_imgs)
        self.g_local_imgs = tf.convert_to_tensor(self.g_local_imgs)
        
        #global discriminator setting
        self.local_fake_d_logits, self.local_fake_d_net = self.local_discriminator(self.g_local_imgs, name="local_discriminator")
        self.local_real_d_logits, self.local_real_d_net = self.local_discriminator(self.r_local_imgs, name="local_discriminator", reuse=True)

        #local discriminator setting
        self.global_fake_d_logits, self.global_fake_d_net = self.global_discriminator(self.recon_img, name="global_discriminator")
        self.global_real_d_logits, self.global_real_d_net = self.global_discriminator(self.real_img, name="global_discriminator", reuse=True)

        self.fake_d_logits = tf.concat([self.local_fake_d_logits, self.global_fake_d_logits], axis=1)
        self.real_d_logits = tf.concat([self.local_fake_d_logits, self.global_fake_d_logits], axis=1)

        self.fake_loss = linear(self.fake_d_logits, 1, "fake_loss")
        self.real_loss = linear(self.real_d_logits, 1, "real_loss")

        trainable_vars = tf.trainable_variables()
        self.c_vars = []
        self.d_vars = []
        for var in trainable_vars:
            if "completion_net" in var.name:
                self.c_vars.append(var)
            else:
                self.d_vars.append(var)

    #loss function
    def build_loss(self):
        def calc_loss(logits, label):
            if label==1:
                y = tf.ones_like(logits)
            else:
                y = tf.zeros_like(logits)
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

        self.fake_d_loss = calc_loss(self.fake_loss, 0)
        self.real_d_loss = calc_loss(self.real_loss, 1)

        self.d_loss = self.fake_d_loss + self.real_d_loss

        self.g_loss = calc_loss(self.fake_loss, 1)
        
        #equation 2 in the paper
        self.recon_loss = tf.sqrt(tf.square(tf.multiply(self.mask, tf.subtract(self.real_img, self.recon_img))))
        
        self.loss = self.recon_loss + self.alpha*self.g_loss

    # completion network 
    def completion_net(self, X_r, name="generator"):
        input_shape = X_r.get_shape().as_list()
        nets = []
        with tf.variable_scope(name) as scope:

            #encode
            # conv1 = tf.contrib.layers.conv2d(X_r, 64, 5, 1,
            #                          padding="SAME",
            #                          activation_fn=None,
            #                          scope="conv1")
            conv= conv2d(X_r, 64,
                          kernel=5,
                          stride=1,
                          padding="SAME",
                          name="conv1"
                          )
            conv = batch_norm(conv, name="conv_bn1")
            conv = tf.nn.relu(conv)

            conv = conv2d(conv, 128,
                          kernel=3,
                          stride=2,
                          padding="VALID",
                          name="conv2"
                          )
            conv = batch_norm(conv, name="conv_bn2")
            conv = tf.nn.relu(conv)

            conv = conv2d(conv, 128,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv3"
                          )
            conv = batch_norm(conv, name="conv_bn3")
            conv = tf.nn.relu(conv)

            conv = conv2d(conv, 256,
                          kernel=3,
                          stride=2,
                          padding="VALID",
                          name="conv4"
                          )
            conv = batch_norm(conv, name="conv_bn4")
            conv = tf.nn.relu(conv)

            conv = conv2d(conv, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv5"
                          )
            conv = batch_norm(conv, name="conv_bn5")
            conv = tf.nn.relu(conv)

            conv = conv2d(conv, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv6"
                          )
            conv = batch_norm(conv, name="conv_bn6")
            conv = tf.nn.relu(conv)

            #Dilated conv from here
            dilate_conv = dilate_conv2d(conv, 
                                        [self.batch_size, conv.get_shape()[1], conv.get_shape()[2], 256],
                                        rate=2,
                                        name="dilate_conv1")

            dilate_conv = dilate_conv2d(dilate_conv, 
                                        [self.batch_size, dilate_conv.get_shape()[1], dilate_conv.get_shape()[2], 256],
                                        rate=4,
                                        name="dilate_conv2")

            dilate_conv = dilate_conv2d(dilate_conv, 
                                        [self.batch_size, dilate_conv.get_shape()[1], dilate_conv.get_shape()[2], 256],
                                        rate=8,
                                        name="dilate_conv3")

            dilate_conv = dilate_conv2d(dilate_conv, 
                                        [self.batch_size, dilate_conv.get_shape()[1], dilate_conv.get_shape()[2], 256],
                                        rate=16,
                                        name="dilate_conv4")                                                                                              

            #resize back
            conv = conv2d(conv, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv7"
                          )
            conv = batch_norm(conv, name="conv_bn7")
            conv = tf.nn.relu(conv)

            conv = conv2d(conv, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv8"
                          )
            conv = batch_norm(conv, name="conv_bn8")
            conv = tf.nn.relu(conv)


            deconv = deconv2d(conv, [self.batch_size, input_shape[1]/2, input_shape[2]/2, 128], name="deconv1")
            deconv = batch_norm(deconv, name="deconv_bn1")
            deconv = tf.nn.relu(deconv)


            conv = conv2d(deconv, 128,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv9"
                          )
            conv = batch_norm(conv, name="conv_bn9")
            conv = tf.nn.relu(conv)

            deconv = deconv2d(conv, [self.batch_size, input_shape[1], input_shape[1], 64], name="deconv2")
            deconv = batch_norm(deconv, name="deconv_bn2")
            deconv = tf.nn.relu(deconv)

            conv = conv2d(deconv, 32,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv10"
                          )
            conv = batch_norm(conv, name="conv_bn10")
            conv = tf.nn.relu(conv)

            conv = tf.contrib.layers.conv2d(conv, 3, 3, 1,
                                     padding="SAME",
                                     activation_fn=None,
                                     scope="conv11")
            conv = batch_norm(conv, name="conv_bn11")
            conv = tf.nn.relu(conv)

            return conv, nets

    # D network from DCGAN
    def local_discriminator(self, input, name="local_discriminator", reuse=False):
        nets = []
        with tf.variable_scope(name, reuse=reuse) as scope:
            conv1 = tf.contrib.layers.conv2d(input, 64, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv1")
            conv1 = batch_norm(conv1, name="bn1")
            conv1 = tf.nn.relu(conv1)
            nets.append(conv1)

            conv2 = tf.contrib.layers.conv2d(conv1, 128, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv2")
            conv2 = batch_norm(conv2, name="bn2")
            conv2 = tf.nn.relu(conv2)
            nets.append(conv2)

            conv3 = tf.contrib.layers.conv2d(conv2, 256, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv3")
            conv3 = batch_norm(conv1, name="bn3")
            conv3 = tf.nn.relu(conv3)
            nets.append(conv3)

            conv4 = tf.contrib.layers.conv2d(conv3, 512, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv4")
            conv4 = batch_norm(conv4, name="bn4")                                                                                                                           
            conv4 = tf.nn.relu(conv4)
            nets.append(conv4)

            conv5 = tf.contrib.layers.conv2d(conv4, 512, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv5")
            conv5 = batch_norm(conv5, name="bn5")                                                                                                                           
            conv5 = tf.nn.relu(conv5)
            nets.append(conv5)

            flatten = tf.contrib.layers.flatten(conv5)

            output = linear(flatten, 1024, name="linear")

            return output, nets



    def global_discriminator(self, input, name="global_discriminator", reuse=False):
        nets = []
        with tf.variable_scope(name, reuse=reuse) as scope:
            conv1 = tf.contrib.layers.conv2d(input, 64, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv1")
            conv1 = batch_norm(conv1, name="bn1")
            conv1 = tf.nn.relu(conv1)
            nets.append(conv1)

            conv2 = tf.contrib.layers.conv2d(conv1, 128, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv2")
            conv2 = batch_norm(conv2, name="bn2")
            conv2 = tf.nn.relu(conv2)
            nets.append(conv2)

            conv3 = tf.contrib.layers.conv2d(conv2, 256, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv3")
            conv3 = batch_norm(conv1, name="bn3")
            conv3 = tf.nn.relu(conv3)
            nets.append(conv3)

            conv4 = tf.contrib.layers.conv2d(conv3, 512, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv4")
            conv4 = batch_norm(conv4, name="bn4")                                                                                                                           
            conv4 = tf.nn.relu(conv4)
            nets.append(conv4)

            conv5 = tf.contrib.layers.conv2d(conv4, 512, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv5")
            conv5 = batch_norm(conv5, name="bn5")                                                                                                                           
            conv5 = tf.nn.relu(conv5)
            nets.append(conv5)

            conv6 = tf.contrib.layers.conv2d(conv5, 512, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv6")
            conv6 = batch_norm(conv6, name="bn6")                                                                                                                           
            conv6 = tf.nn.relu(conv6)
            nets.append(conv6)


            flatten = tf.contrib.layers.flatten(conv6)

            output = linear(flatten, 1024, name="linear")


            return output, nets
