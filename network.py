import tensorflow as tf
import numpy as np

from ops import *
from architecture import *

class network():
    def __init__(self, args):

        self.batch_size = args.batch_size
        self.input_dim = args.input_dim 

        self.local_width, self.local_height = args.local_input_width, args.local_input_height

        self.m = args.margin

        self.alpha = args.alpha

        #prepare training data
        self.real_img, self.perturbed_img, self.mask, self.coord, self.pads, self.data_count = load_train_data(args)
        # self.orig_img, self.test_img, self.test_mask, self.test_data_count = load_test_data(args)
        
        self.single_orig = tf.placeholder(tf.float32, (args.batch_size, args.input_height, args.input_width, 3))
        self.single_test = tf.placeholder(tf.float32, (args.batch_size, args.input_height, args.input_width, 3))
        self.single_mask = tf.placeholder(tf.float32, (args.batch_size, args.input_height, args.input_width, 3))

        self.build_model()
        self.build_loss()

        #summary
        self.recon_loss_sum = tf.summary.scalar("recon_loss", self.recon_loss) 
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss) 
        self.loss_all_sum = tf.summary.scalar("loss_all", self.loss_all)
        self.input_img_sum = tf.summary.image("input_img", self.perturbed_img, max_outputs=5)
        self.real_img_sum = tf.summary.image("real_img", self.real_img, max_outputs=5)
        
        self.recon_img_sum = tf.summary.image("recon_img", self.recon_img, max_outputs=5)
        self.g_local_imgs_sum = tf.summary.image("g_local_imgs", self.g_local_imgs, max_outputs=5)
        self.r_local_imgs_sum = tf.summary.image("r_local_imgs", self.r_local_imgs, max_outputs=5)

    #structure of the model
    def build_model(self):
        def rand_crop(img, coord, pads):
          cropped = tf.image.resize_images(tf.image.crop_to_bounding_box(img, coord[0]-self.m, coord[1]-self.m, pads[0]+self.m*2, pads[1]+self.m*2), (self.local_height, self.local_width))
          return cropped

        # self.C = tf.concat([self.perturbed_img, self.mask], -1)

        self.recon_img, self.g_nets = self.completion_net(self.perturbed_img, name="completion_net")
        self.recon_img = (1-self.mask)*self.real_img + self.mask*self.recon_img

        self.test_res_imgs, _ = self.completion_net(self.single_test, name="completion_net", reuse=True)
        self.test_res_imgs = (1-self.single_mask)*self.single_orig + self.single_mask*self.test_res_imgs

        self.r_local_imgs = []
        self.g_local_imgs = [] 
        for idx in range(0,self.real_img.shape[0]):
            r_cropped = rand_crop(self.real_img[idx], self.coord[idx], self.pads[idx])
            g_cropped = rand_crop(self.recon_img[idx], self.coord[idx], self.pads[idx])
            self.r_local_imgs.append(r_cropped)
            self.g_local_imgs.append(g_cropped)


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

        #loss to train the discriminator
        self.d_loss = self.alpha*(self.fake_d_loss + self.real_d_loss)

        self.g_loss = calc_loss(self.fake_loss, 1)
        
        #mse loss in the paper
        self.recon_loss = tf.reduce_mean(tf.nn.l2_loss(self.real_img - self.recon_img))
        
        self.loss_all = self.recon_loss + self.alpha*self.g_loss

    # completion network 
    def completion_net(self, input, name="generator", reuse=False):
        input_shape = input.get_shape().as_list()
        nets = []
        with tf.variable_scope(name, reuse=reuse) as scope:
            conv1 = conv2d(input, 64,
                          kernel=5,
                          stride=1,
                          padding="SAME",
                          name="conv1"
                          )
            conv1 = batch_norm(conv1, name="conv_bn1")
            conv1 = tf.nn.relu(conv1)
            
            conv2 = conv2d(conv1, 128,
                          kernel=3,
                          stride=2,
                          padding="SAME",
                          name="conv2"
                          )
            conv2 = batch_norm(conv2, name="conv_bn2")
            conv2 = tf.nn.relu(conv2)

            conv3 = conv2d(conv2, 128,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv3"
                          )
            conv3 = batch_norm(conv3, name="conv_bn3")
            conv3 = tf.nn.relu(conv3)

            conv4 = conv2d(conv3, 256,
                          kernel=3,
                          stride=2,
                          padding="SAME",
                          name="conv4"
                          )
            conv4 = batch_norm(conv4, name="conv_bn4")
            conv4 = tf.nn.relu(conv4)

            conv5 = conv2d(conv4, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv5"
                          )
            conv5 = batch_norm(conv5, name="conv_bn5")
            conv5 = tf.nn.relu(conv5)

            conv6 = conv2d(conv5, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv6"
                          )
            conv6 = batch_norm(conv5, name="conv_bn6")
            conv6 = tf.nn.relu(conv5)

            #Dilated conv from here
            dilate_conv1 = dilate_conv2d(conv6, 
                                        [self.batch_size, conv6.get_shape()[1], conv6.get_shape()[2], 256],
                                        rate=2,
                                        name="dilate_conv1")

            dilate_conv2 = dilate_conv2d(dilate_conv1, 
                                        [self.batch_size, dilate_conv1.get_shape()[1], dilate_conv1.get_shape()[2], 256],
                                        rate=4,
                                        name="dilate_conv2")

            dilate_conv3 = dilate_conv2d(dilate_conv2, 
                                        [self.batch_size, dilate_conv2.get_shape()[1], dilate_conv2.get_shape()[2], 256],
                                        rate=8,
                                        name="dilate_conv3")

            dilate_conv4 = dilate_conv2d(dilate_conv3, 
                                        [self.batch_size, dilate_conv3.get_shape()[1], dilate_conv3.get_shape()[2], 256],
                                        rate=16,
                                        name="dilate_conv4")                                                                                              

            #resize back
            conv7 = conv2d(dilate_conv4, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv7"
                          )
            conv7 = batch_norm(conv7, name="conv_bn7")
            conv7 = tf.nn.relu(conv7)

            conv8 = conv2d(conv7, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv8"
                          )
            conv8 = batch_norm(conv8, name="conv_bn8")
            conv8 = tf.nn.relu(conv8)

            deconv1 = deconv2d(conv8, [self.batch_size, input_shape[1]/2, input_shape[2]/2, 128], name="deconv1")
            deconv1 = batch_norm(deconv1, name="deconv_bn1")
            deconv1 = tf.nn.relu(deconv1)

            conv9 = conv2d(deconv1, 128,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv9"
                          )
            conv9 = batch_norm(conv9, name="conv_bn9")
            conv9 = tf.nn.relu(conv9)

            deconv2 = deconv2d(conv9, [self.batch_size, input_shape[1], input_shape[2], 64], name="deconv2")
            deconv2 = batch_norm(deconv2, name="deconv_bn2")
            deconv2 = tf.nn.relu(deconv2)

            conv10 = conv2d(deconv2, 32,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv10"
                          )
            conv10 = batch_norm(conv10, name="conv_bn10")
            conv10 = tf.nn.relu(conv10)

            conv11 = conv2d(conv10, 3,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv11"
                          )
            conv11 = batch_norm(conv11, name="conv_bn11")
            conv11 = tf.nn.tanh(conv11)

            return conv11, nets

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
