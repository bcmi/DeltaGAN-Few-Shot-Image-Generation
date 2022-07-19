import tensorflow as tf
from dagan_architectures_with_matchingclassifier import UResNetGenerator, Discriminator, Discriminator_latent, \
    Discriminator_images_latent
import numpy as np
import time
from utils.network_summary import count_parameters

setting_training = True


def rbf_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    x = tf.reduce_mean(x, [1, 2])
    y = tf.reduce_mean(y, [1, 2])
    x = tf.reshape(x, [x_size, tf.shape(x)[1]])
    y = tf.reshape(y, [y_size, tf.shape(y)[1]])
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    ###gaussian kernal
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def get_mmd(x, y, sigma_sqr=1.0):
    x_kernel = rbf_kernel(x, x)
    y_kernel = rbf_kernel(y, y)
    xy_kernel = rbf_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


def Hinge_loss(real_logits, fake_logits):
    D_loss = -tf.reduce_mean(tf.minimum(0., -1.0 + real_logits)) - tf.reduce_mean(tf.minimum(0., -1.0 - fake_logits))
    G_loss = -tf.reduce_mean(fake_logits)
    return D_loss, G_loss


class DAGAN:
    def __init__(self, input_x_i, input_x_j, input_y_i, input_y_j, input_global_y_i, input_global_y_j,
                 input_x_j_selected, input_global_y_j_selected, classes, dropout_rate, generator_layer_sizes,
                 discriminator_layer_sizes, generator_layer_padding, z_inputs, z_inputs_2, matching, fce,
                 full_context_unroll_k, average_per_class_embeddings, batch_size=100, z_dim=100,
                 num_channels=1, is_training=True, augment=True, discr_inner_conv=0, gen_inner_conv=0, num_gpus=1,
                 is_z2=True, is_z2_vae=True, is_attention=0, connection_layers=3,
                 use_wide_connections=False, selected_classes=5, support_num=5, loss_G=1, loss_D=1, loss_KL=0.0001,
                 loss_recons_B=0.01, loss_matching_G=0.01, loss_matching_D=0.01, loss_CLA=1, loss_FSL=1, loss_sim=0.01,
                 z1z2_training=True):

        """
        Initializes a DAGAN object.
        :param input_x_i: Input image x_i
        :param input_x_j: Input image x_j
        :param dropout_rate: A dropout rate placeholder or a scalar to use throughout the network
        :param generator_layer_sizes: A list with the number of feature maps per layer (generator) e.g. [64, 64, 64, 64]
        :param discriminator_layer_sizes: A list with the number of feature maps per layer (discriminator)
                                                                                                   e.g. [64, 64, 64, 64]
        :param generator_layer_padding: A list with the type of padding per layer (e.g. ["SAME", "SAME", "SAME","SAME"]
        :param z_inputs: A placeholder for the random noise injection vector z (usually gaussian or uniform distribut.)
        :param batch_size: An integer indicating the batch size for the experiment.
        :param z_dim: An integer indicating the dimensionality of the random noise vector (usually 100-dim).
        :param num_channels: Number of image channels
        :param is_training: A boolean placeholder for the training/not training flag
        :param augment: A boolean placeholder that determines whether to augment the data using rotations
        :param discr_inner_conv: Number of inner layers per multi layer in the discriminator
        :param gen_inner_conv: Number of inner layers per multi layer in the generator
        :param num_gpus: Number of GPUs to use for training
        """
        self.training = setting_training
        self.print = False
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.z_inputs = z_inputs
        self.z_inputs_2 = z_inputs_2
        self.num_gpus = num_gpus
        self.support_num = support_num
        self.loss_G = loss_G
        self.loss_D = loss_D
        self.loss_KL = loss_KL
        self.loss_CLA = loss_CLA
        self.loss_FSL = loss_FSL
        self.loss_matching_G = loss_matching_G
        self.loss_recons_B = loss_recons_B
        self.loss_matching_D = loss_matching_D
        self.loss_sim = loss_sim
        self.input_x_i = input_x_i
        self.input_x_j = input_x_j
        self.input_x_j_selected = input_x_j_selected
        self.input_y_i = input_y_i
        self.input_y_j = input_y_j
        self.input_global_y_i = input_global_y_i
        self.input_global_y_j = input_global_y_j
        self.input_global_y_j_selected = input_global_y_j_selected
        self.classes = classes
        self.selected_classes = selected_classes
        self.dropout_rate = dropout_rate
        self.training_phase = is_training
        self.augment = augment
        self.is_z2 = is_z2
        self.is_z2_vae = is_z2_vae
        self.z1z2_training = z1z2_training
        self.is_attention = is_attention
        self.connection_layers = connection_layers

        self.g = UResNetGenerator(batch_size=self.batch_size, layer_sizes=generator_layer_sizes,
                                  num_channels=num_channels, layer_padding=generator_layer_padding,
                                  inner_layers=gen_inner_conv, name="generator", matching=matching, fce=fce,
                                  full_context_unroll_k=full_context_unroll_k,
                                  average_per_class_embeddings=average_per_class_embeddings)

        # self.d = Discriminator(batch_size=self.batch_size, layer_sizes=discriminator_layer_sizes,
        #                        inner_layers=discr_inner_conv, use_wide_connections=use_wide_connections,
        #                        name="discriminator")
        # self.d_refer = Discriminator(batch_size=self.batch_size, layer_sizes=discriminator_layer_sizes,
        #                              inner_layers=discr_inner_conv, use_wide_connections=use_wide_connections,
        #                              name="discriminator_refer")
        # self.d_Z = Discriminator_latent(batch_size=self.batch_size, layer_sizes=discriminator_layer_sizes,
        #                                 inner_layers=discr_inner_conv, use_wide_connections=use_wide_connections,
        #                                 name="discriminator_z")
        self.d = Discriminator_images_latent(batch_size=self.batch_size, layer_sizes=discriminator_layer_sizes,
                                             inner_layers=discr_inner_conv, use_wide_connections=use_wide_connections,
                                             name="discriminator")

    def rotate_data(self, image_a, image_b):
        """
        Rotate 2 images by the same number of degrees
        :param image_a: An image a to rotate k degrees
        :param image_b: An image b to rotate k degrees
        :return: Two images rotated by the same amount of degrees
        """
        random_variable = tf.unstack(tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32, seed=None, name=None))
        image_a = tf.image.rot90(image_a, k=random_variable[0])
        image_b = tf.image.rot90(image_b, k=random_variable[0])
        return [image_a, image_b]

    def rotate_batch(self, batch_images_a, batch_images_b):
        """
        Rotate two batches such that every element from set a with the same index as an element from set b are rotated
        by an equal amount of degrees
        :param batch_images_a: A batch of images to be rotated
        :param batch_images_b: A batch of images to be rotated
        :return: A batch of images that are rotated by an element-wise equal amount of k degrees
        """
        shapes = map(int, list(batch_images_a.get_shape()))
        batch_size, x, y, c = shapes
        with tf.name_scope('augment'):
            batch_images_unpacked_a = tf.unstack(batch_images_a)
            batch_images_unpacked_b = tf.unstack(batch_images_b)
            new_images_a = []
            new_images_b = []
            for image_a, image_b in zip(batch_images_unpacked_a, batch_images_unpacked_b):
                rotate_a, rotate_b = self.augment_rotate(image_a, image_b)
                new_images_a.append(rotate_a)
                new_images_b.append(rotate_b)

            new_images_a = tf.stack(new_images_a)
            new_images_a = tf.reshape(new_images_a, (batch_size, x, y, c))
            new_images_b = tf.stack(new_images_b)
            new_images_b = tf.reshape(new_images_b, (batch_size, x, y, c))
            return [new_images_a, new_images_b]

    def generate(self, conditional_images, support_input, input_global_x_j_selected, input_y_i, input_y_j,
                 input_global_y_i, input_global_y_j_selected, selected_classes, support_num, classes, is_z2, is_z2_vae,
                 z_input11=None, z_input_22=None):
        """
        Generate samples with the DAGAN
        :param conditional_images: Images to condition DAGAN on.
        :param z_input: Random noise to condition the DAGAN on. If none is used then the method will generate random
        noise with dimensionality [batch_size, z_dim]
        :return: A batch of generated images, one per conditional image
        """
        if z_input11 is None:
            # z_input11 = tf.random_normal([self.batch_size, self.z_dim], mean=0, stddev=1)
            # z_input_21 = tf.random_normal([self.batch_size, self.z_dim], mean=0, stddev=1)
            z_input11 = tf.random_uniform(shape=[self.batch_size, self.z_dim], minval=-1, maxval=1)
            z_input_21 = tf.random_uniform(shape=[self.batch_size, self.z_dim], minval=-1, maxval=1)
        if self.training:
            gan_decoder_delta, gan_decoder_delta2, gan_recons_image1, gan_decoder_delta_trs, z_inputs, z_inputs_2, fake_delta, fake_delta2, real_delta_x1x2, decoder_layers, decoder_layers2, \
            mmd_delta, renconstruction_loss_image = self.g(
                z_input11, z_input_22,
                conditional_images, support_input, input_y_i, input_y_j, classes, support_num, is_z2,
                is_z2_vae,
                is_attention=self.is_attention,
                connection_layers=self.connection_layers,
                training=self.training_phase,
                dropout_rate=self.dropout_rate,
                z1z2_training=self.z1z2_training,
                z_dim=self.z_dim)
            return gan_decoder_delta, gan_decoder_delta2, gan_recons_image1, gan_decoder_delta_trs, z_inputs, z_inputs_2, fake_delta, fake_delta2, real_delta_x1x2, decoder_layers, decoder_layers2, \
                   mmd_delta, renconstruction_loss_image
        else:
            generated_samples, similarities, similarities_data, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake = self.g(
                z_input11, z_input_22,
                conditional_images, support_input, input_y_i, input_y_j, selected_classes, support_num, is_z2,
                is_z2_vae,
                training=self.training_phase,
                dropout_rate=self.dropout_rate,
                z1z2_training=self.z1z2_training,
                z_dim=self.z_dim)

            similarities_onehot = tf.cast((0) * tf.ones_like(similarities[:, 0]), dtype=tf.int32)
            similarities_onehot = tf.expand_dims(similarities_onehot, axis=-1)
            similarities_index = tf.expand_dims(similarities[:, 0], axis=-1)

            # g_same_class_outputs = self.d(generated_samples, similarities_onehot, similarities_index, input_global_x_j_selected, input_global_y_i,
            #                               input_global_y_j_selected, selected_classes, support_num, classes,
            #                               similarities, training=self.training_phase,
            #                               dropout_rate=self.dropout_rate)
            return generated_samples, similarities, z_input11, preds_fake

    def augment_rotate(self, image_a, image_b):
        r = tf.unstack(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, seed=None, name=None))
        rotate_boolean = tf.equal(0, r, name="check-rotate-boolean")
        [image_a, image_b] = tf.cond(rotate_boolean[0], lambda: self.rotate_data(image_a, image_b),
                                     lambda: [image_a, image_b])
        return image_a, image_b

    def data_augment_batch(self, batch_images_a, batch_images_b):
        [images_a, images_b] = tf.cond(self.augment, lambda: self.rotate_batch(batch_images_a, batch_images_b),
                                       lambda: [batch_images_a, batch_images_b])

        return images_a, images_b

    def save_features(self, name, features):
        for i in range(len(features)):
            shape_in = features[i].get_shape().as_list()
            channels = shape_in[3]
            y_channels = 8
            x_channels = channels / y_channels

            activations_features = tf.reshape(features[i], shape=(shape_in[0], shape_in[1], shape_in[2],
                                                                  y_channels, x_channels))

            activations_features = tf.unstack(activations_features, axis=4)
            activations_features = tf.concat(activations_features, axis=2)
            activations_features = tf.unstack(activations_features, axis=3)
            activations_features = tf.concat(activations_features, axis=1)
            activations_features = tf.expand_dims(activations_features, axis=3)
            # tf.summary.image('{}_{}'.format(name, i), activations_features)

    def loss(self, gpu_id):
        with tf.name_scope("losses_{}".format(gpu_id)):
            before_loss = time.time()
            epsilon = 1e-8
            input_a, input_b, input_y_a, input_y_b, input_global_y_a, input_global_y_b, input_b_selected, input_global_y_b_selected = \
                self.input_x_i[gpu_id], self.input_x_j[gpu_id], self.input_y_i[gpu_id], self.input_y_j[gpu_id], \
                self.input_global_y_i[gpu_id], self.input_global_y_j[gpu_id], self.input_x_j_selected[gpu_id], \
                self.input_global_y_j_selected[gpu_id]

            current_support = input_b

            x_g1, x_g2, gan_recons_image1, gan_decoder_delta_trs, z1, z2, fake_delta, fake_delta2, real_delta_x1x2, decoder_layers, decoder_layers2, mmd_delta, renconstruction_loss_image = \
                self.generate(input_a, input_b, input_b_selected, input_global_y_a, input_global_y_b, input_global_y_a,
                              input_global_y_b_selected, self.selected_classes, self.support_num, self.classes,
                              self.is_z2, self.is_z2_vae)

            #### fake image
            ##### distinguish real and false
            real_feature, d_real, d_realZ, t_classification_loss = self.d(input_b[:, 1], input_b[:, 0],input_global_y_b[:, 1],
                                                                          input_global_y_b_selected, self.classes,
                                                                          real_delta_x1x2,
                                                                          training=self.training_phase,
                                                                          dropout_rate=self.dropout_rate)
            fake_feature, d_fake, d_fakeZ1, g_classification_loss = self.d(x_g1, input_b[:, 0], input_global_y_b[:, 1],
                                                                           input_global_y_b_selected, self.classes,
                                                                           fake_delta,
                                                                           training=self.training_phase,
                                                                           dropout_rate=self.dropout_rate)

            fake_feature2, d_fake2, d_fakeZ2, g_classification_loss2 = self.d(x_g2,input_b[:, 0], input_global_y_b[:, 1],
                                                                              input_global_y_b_selected, self.classes,
                                                                              fake_delta2,
                                                                              training=self.training_phase,
                                                                              dropout_rate=self.dropout_rate)

            fake_feature_refer, d_fake_refer, d_fake_trans, g_classification_loss_refer = self.d(gan_decoder_delta_trs,input_b[:, 0],
                                                                                                 input_global_y_b[:, 1],
                                                                                                 input_global_y_b_selected,
                                                                                                 self.classes,
                                                                                                 real_delta_x1x2,
                                                                                                 training=self.training_phase,
                                                                                                 dropout_rate=self.dropout_rate)

            # d_realZ = self.d_Z(x_g2, input_global_y_b[:, 1],
            #                    input_global_y_b_selected, self.classes, real_delta_x1x2,
            #                    training=self.training_phase,
            #                    dropout_rate=self.dropout_rate)
            # d_fakeZ1 = self.d_Z(x_g2, input_global_y_b[:, 1],
            #                    input_global_y_b_selected, self.classes, fake_delta,
            #                    training=self.training_phase,
            #                    dropout_rate=self.dropout_rate)
            # d_fakeZ2 = self.d_Z(x_g2, input_global_y_b[:, 1],
            #                     input_global_y_b_selected, self.classes, fake_delta2,
            #                     training=self.training_phase,
            #                     dropout_rate=self.dropout_rate)

            d_fakeZ = tf.stack([d_fakeZ1, d_fakeZ2], axis=1)
            d_fakeZ = tf.reduce_mean(d_fakeZ, axis=1)

            #
            # real_feature_refer, d_real_refer, t_classification_loss_refer = self.d_refer(input_b[:, 1],
            #                                                                              input_global_y_b[:, 1],
            #                                                                              input_global_y_b_selected,
            #                                                                              self.classes, z1,
            #                                                                              training=self.training_phase,
            #                                                                              dropout_rate=self.dropout_rate)
            # fake_feature_refer, d_fake_refer, g_classification_loss_refer = self.d_refer(gan_decoder_delta_trs,
            #                                                                              input_global_y_b[:, 1],
            #                                                                              input_global_y_b_selected,
            #                                                                              self.classes, z1,
            #                                                                              training=self.training_phase,
            #                                                                              dropout_rate=self.dropout_rate)

            # d_real = tf.concat([d_real, d_real0],axis=0)
            # t_classification_loss = t_classification_loss + t_classification_loss_refer
            # g_classification_loss = g_classification_loss+ g_classification_loss_refer

            # d_fake = tf.concat([d_fake, d_fake2], axis=0)
            d_fake = tf.stack([d_fake, d_fake2], axis=1)
            d_fake = tf.reduce_mean(d_fake, axis=1)

            ###### reconstruct \tilde{x}_2 - x_2
            '''
            delta_image_recons1 = tf.abs( x_g1 - input_b[:,1])
            delta_image_recons2 = tf.abs( x_g2 - input_b[:,1])
            delta_image_recons = tf.concat([delta_image_recons1, delta_image_recons2],axis=0)
            '''
            delta_image_recons = tf.abs(gan_decoder_delta_trs - input_b[:, 1])
            ###### reconstruct \tilde{x}_2 - x_2

            ##### dsistance from x1
            x1_dis1 = -tf.abs(x_g1 - input_b[:, 0])
            x1_dis2 = -tf.abs(x_g2 - input_b[:, 0])
            x1_dis = tf.stack([x1_dis1, x1_dis2], axis=1)
            x1_dis = tf.reduce_mean(x1_dis, axis=1)

            ##### multiscale feature matching
            # feature_loss = []
            # for k in range(len(real_feature)):
            #     feature_loss.append(tf.reduce_mean(tf.abs(real_feature_refer[k] - fake_feature_refer[k]), axis=[1, 2, 3]))
            # feature_loss = tf.stack(feature_loss,axis=1)
            # feature_loss = tf.reduce_mean(feature_loss,axis=1)
            feature_loss = []
            for k in range(len(real_feature)):
                feature_loss.append(
                    tf.reduce_mean(tf.abs(real_feature[k] - fake_feature_refer[k]), axis=[1, 2, 3]))
            feature_loss = tf.stack(feature_loss, axis=1)
            feature_loss = tf.reduce_mean(feature_loss, axis=1)
            # feature_loss = tf.concat(feature_loss, axis=0)
            ##### multiscale feature matching

            ##### mmd on the discriminator
            # mmd_feature_loss = []
            mmd_feature_loss = 0
            for k in range(len(real_feature)):
                mmd_feature1 = get_mmd(real_feature[k], fake_feature[k])
                mmd_feature2 = get_mmd(real_feature[k], fake_feature2[k])
                mmd_feature = (mmd_feature1 + mmd_feature2) / 2
                mmd_feature_loss += mmd_feature
            mmd_feature_loss = mmd_feature_loss / len(real_feature)

            '''
            ##### normoalization feature
            fake_feature__ = tf.nn.l2_normalize(tf.reshape(fake_feature[-1],[fake_feature[-1].get_shape()[0], -1]),axis=1)
            fake_feature2__ = tf.nn.l2_normalize(tf.reshape(fake_feature2[-1],[fake_feature2[-1].get_shape()[0], -1]),axis=1)
            current_loss_diverification = -  (tf.reduce_mean(tf.abs(fake_feature__ - fake_feature2__)) / (tf.reduce_mean(tf.abs(z1 - z2)) + 1e-6))
            ##### normoalization feature
            '''

            def diverification_loss_function(x_g1, x_g2, fake_feature, fake_feature2, fake_delta, fake_delta2,
                                             decoder_layers, decoder_layers2, is_feature_level, is_z1z2distance,
                                             is_threshold):
                if is_feature_level:
                    feature_distance = []
                    ##### feature from discrimiantor
                    for k in range(len(real_feature)):
                        if fake_feature[k].shape.ndims > 2:
                            feature_distance.append(
                                tf.reduce_mean(tf.abs(fake_feature[k] - fake_feature2[k]), axis=[1, 2, 3]))
                        else:
                            feature_distance.append(
                                tf.reduce_mean(tf.abs(fake_feature[k] - fake_feature2[k]), axis=[1]))

                    # #### features from delta encoder
                    # feature_distance.append(tf.reduce_mean(tf.abs(fake_delta - fake_delta2),axis=[1,2,3]))
                    #
                    # #### features from the decoder of generator
                    # for k in range(len(decoder_layers)):
                    #     if k== len(decoder_layers)-1:
                    #         # print('here', decoder_layers[k])
                    #         feature_distance.append(tf.reduce_mean(tf.abs(decoder_layers[k] - decoder_layers2[k]),axis=[1,2,3]))

                    feature_distance = tf.stack(feature_distance, axis=1)
                    feature_distance = tf.reduce_mean(feature_distance, axis=1)
                    if is_z1z2distance:
                        '''current_loss_diverification = - tf.divide(
                            (tf.reduce_mean(tf.abs(fake_feature[-1] - fake_feature2[-1]), axis=[1, 2, 3])),
                            tf.reduce_mean(tf.abs(z1 - z2), axis=[1]) + 1e-6)'''
                        #### features from discriminator
                        current_loss_diverification_diverse = tf.divide(feature_distance,
                                                                        tf.reduce_mean(tf.abs(z1 - z2), axis=[1]))
                        current_loss_diverification = 1 / (current_loss_diverification_diverse + 1e-5)
                        current_conv_diverfication_diverse = tf.divide(feature_distance,
                                                                       tf.reduce_mean(tf.abs(fake_delta - fake_delta2),
                                                                                      axis=[1, 2, 3]))
                        current_conv_diverfication = 1 / (current_conv_diverfication_diverse + 1e-5)
                    else:
                        current_loss_diverification = -  (tf.reduce_mean(feature_distance))
                else:
                    if is_z1z2distance:
                        # current_loss_diverification =   - tf.divide(  tf.reduce_mean(tf.abs(x_g1 - x_g2),axis=[1,2,3]), tf.reduce_mean(tf.abs(z1 - z2),axis=[1]) + 1e-6  )
                        current_loss_diverification = tf.divide(tf.reduce_mean(tf.abs(z1 - z2), axis=[1]) + 1e-6,
                                                                tf.reduce_mean(tf.abs(x_g1 - x_g2), axis=[1, 2, 3]))
                        current_conv_diverfication = tf.divide(
                            tf.reduce_mean(tf.abs(fake_delta - fake_delta2), axis=[1, 2, 3]) + 1e-6,
                            tf.reduce_mean(tf.abs(x_g1 - x_g2), axis=[1, 2, 3]))
                    else:
                        current_loss_diverification = -  (tf.reduce_mean(tf.abs(x_g1 - x_g2), axis=[1, 2, 3]))
                ####### threshold for diverification loss
                thresholed_loss_diverification = tf.zeros(shape=current_loss_diverification.get_shape())
                if is_threshold:
                    condition = current_loss_diverification > -10
                else:
                    condition = current_loss_diverification > -100
                loss_diversification = tf.where(condition, current_loss_diverification, thresholed_loss_diverification)
                loss_conv_diverfication = current_conv_diverfication
                return loss_diversification, loss_conv_diverfication

            ##### control version #####
            is_threshold = False
            is_z1z2distance = True
            is_feature_level = True
            is_diverfication_loss_GD = False
            ##### control version #####

            loss_diversification, loss_conv_diverfication = diverification_loss_function(x_g1, x_g2, fake_feature,
                                                                                         fake_feature2, fake_delta,
                                                                                         fake_delta2, decoder_layers,
                                                                                         decoder_layers2,
                                                                                         is_feature_level,
                                                                                         is_z1z2distance, is_threshold)

            ##### without mask
            '''
            delta_image_recons = mask_loss(z_recons_loss, mask)
            z_recons_loss = mask_loss(z_recons_loss, mask)
            loss_diversification = mask_loss(loss_diversification, mask)
            feature_loss = mask_loss(feature_loss, mask)
            KL_loss = mask_loss(KL_loss, mask)
            renconstruction_loss_image = mask_loss(renconstruction_loss_image, mask)
            reconstruction_loss_feature = mask_loss(reconstruction_loss_feature, mask)
            g_classification_loss = mask_loss(g_classification_loss, mask)
            t_classification_loss = mask_loss(t_classification_loss, mask)
            CLASSIFICATION_loss = mask_loss(CLASSIFICATION_loss, mask)
            '''
            loss_conv_diverfication = tf.reduce_mean(loss_conv_diverfication)
            x1_dis = tf.reduce_mean(x1_dis)
            verification_loss = tf.reduce_mean(loss_diversification)
            loss_reconstruction_image = tf.reduce_mean(renconstruction_loss_image)
            # loss_reconstruction_feature = tf.reduce_mean(reconstruction_loss_feature)
            loss_feature = tf.reduce_mean(feature_loss)
            g_classification_loss = tf.reduce_mean(g_classification_loss)
            t_classification_loss = tf.reduce_mean(t_classification_loss)
            delta_image_recons = tf.reduce_mean(delta_image_recons)
            mmd_delta_loss = tf.reduce_mean(mmd_delta)
            mmd_feature_loss = tf.reduce_mean(mmd_feature_loss)

            #####hinge loss version for G and D
            # d_loss_pure_gen, G_loss_gen = Hinge_loss(d_real, d_fake)
            # d_loss_pure_refer, G_loss_refer = Hinge_loss(d_real_refer, d_fake_refer)
            # d_loss_pure =  d_loss_pure_gen +  d_loss_pure_refer
            # G_loss =  G_loss_gen +  G_loss_refer

            d_loss_pure_gen, G_loss_gen = Hinge_loss(d_real, d_fake)
            d_loss_pure_Z, G_loss_Z = Hinge_loss(d_realZ, d_fakeZ)

            d_loss_pure = d_loss_pure_gen +   d_loss_pure_Z
            # d_loss_pure = d_loss_pure_gen
            is_gradient_penalty = False
            if is_gradient_penalty:
                #### 2*self.batch_size: genrated images
                alpha = tf.random_uniform(shape=[2 * self.batch_size, 1, 1, 1], minval=0., maxval=1.)
                interpolates = alpha * tf.concat([x_g1, x_g2], axis=0) + (1 - alpha) * tf.concat(
                    [input_b[:, 1], input_b[:, 1]], axis=0)
                _, pre_grads, _ = self.d(interpolates,
                                         tf.concat([input_global_y_b[:, 1], input_global_y_b[:, 1]], axis=0),
                                         input_global_y_b_selected,
                                         self.classes, z1,
                                         training=self.training_phase,
                                         dropout_rate=self.dropout_rate)
                gradients = tf.gradients(pre_grads, [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                d_loss_pure = d_loss_pure + 5 * gradient_penalty

            G_loss = G_loss_gen +   G_loss_Z
            # G_loss = G_loss_gen

            ###### for feature-level diverfiv
            #### self.loss_recons_B: target stylecode recons
            #### self.loss_sim: cyclegan recons, maintain reference information
            #### self.loss_matching_D: maintain reference image information
            #### self.loss_FSL: make generated images close to target image
            #### self.loss_matching_G: style diverification loss
            g_loss_notraining = G_loss * self.loss_G + g_classification_loss * self.loss_CLA + \
                                self.loss_recons_B * x1_dis + self.loss_sim * mmd_feature_loss + \
                                self.loss_matching_D * loss_feature + self.loss_FSL * delta_image_recons + self.loss_matching_G * verification_loss

            g_loss_training = G_loss * self.loss_G + g_classification_loss * self.loss_CLA + self.loss_recons_B * x1_dis \
                              + self.loss_sim * mmd_feature_loss + self.loss_matching_D * loss_feature \
                              + self.loss_matching_G * verification_loss + self.loss_FSL * delta_image_recons

            g_loss = tf.cond(self.z1z2_training, lambda: g_loss_training,
                             lambda: g_loss_notraining)
            if is_feature_level and is_diverfication_loss_GD:
                d_loss = self.loss_D * (
                    d_loss_pure) + self.loss_CLA * t_classification_loss + self.loss_matching_G * verification_loss
            else:
                d_loss = self.loss_D * (d_loss_pure) + self.loss_CLA * t_classification_loss

                # tf.add_to_collection('fzl_losses',crossentropy_loss_real)

            tf.add_to_collection('g_losses', g_loss)
            tf.add_to_collection('d_losses', d_loss)
            tf.add_to_collection('c_losses', t_classification_loss)

            if is_gradient_penalty:
                tf.summary.scalar('Gradient_penalty', gradient_penalty)
            tf.summary.scalar('G_losses_gen', G_loss_gen)
            tf.summary.scalar('D_losses_gen', d_loss_pure_gen)
            tf.summary.scalar('G_losses_delta', G_loss_Z)
            tf.summary.scalar('D_losses_delta', d_loss_pure_Z)
            tf.summary.scalar('c_losses', g_classification_loss)
            tf.summary.scalar('x1_distance_losses', x1_dis)
            tf.summary.scalar('mmd_delta_loss', mmd_delta_loss)
            tf.summary.scalar('mmd_feature_loss', mmd_feature_loss)

            tf.summary.scalar('delta_image_recons_losses', delta_image_recons)
            # tf.summary.scalar('reconstruction_feature_losses', loss_reconstruction_feature)

            tf.summary.scalar('verification_losses', verification_loss)
            tf.summary.scalar('verification_conv_losses', loss_conv_diverfication)

            tf.summary.scalar('total_g_losses', g_loss)
            tf.summary.scalar('total_d_losses', d_loss)

            tf.summary.scalar('cyclegan_image_recons_losses', loss_reconstruction_image)
            tf.summary.scalar('matchingD_losses', loss_feature)

        return {
            "g_losses": tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            "d_losses": tf.add_n(tf.get_collection('d_losses'), name='total_d_loss'),
            "c_losses": tf.add_n(tf.get_collection('c_losses'), name='total_c_loss'),
            # "fzl_losses":tf.add_n(tf.get_collection('fzl_losses'),name='total_fzl_loss'),
            # "recons_losses":tf.add_n(tf.get_collection('recons_losses'),name='total_recons_loss'),
        }

    def train(self, opts, losses):

        """
        Returns ops for training our DAGAN system.
        :param opts: A dict with optimizers.
        :param losses: A dict with losses.
        :return: A dict with training ops for the dicriminator and the generator.
        """
        opt_ops = dict()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            ##### three stage
            opt_ops["g_opt_op"] = opts["g_opt"].minimize(losses["g_losses"], var_list=self.g.variables,
                                                         colocate_gradients_with_ops=True)

            ### first
            # opt_ops["g_opt_op"] = opts["g_opt"].minimize(losses["g_losses"],var_list=self.g.variables_E + self.g.variables_D ,colocate_gradients_with_ops=True)
            # count_parameters(self.g.variables_E + self.g.variables_D, name="current_update_generator_parameter_num")
            ### second
            # opt_ops["g_opt_op"] = opts["g_opt"].minimize(losses["g_losses"],var_list=self.g.variables_M1M2,colocate_gradients_with_ops=True)
            # count_parameters(self.g.variables_M1M2, name="current_update_generator_parameter_num")

            ### third
            # opt_ops["g_opt_op"] = opts["g_opt"].minimize(losses["g_losses"],var_list= self.g.variables_D,colocate_gradients_with_ops=True)
            # count_parameters(self.g.variables_D, name="current_update_generator_parameter_num")

            # opt_ops["d_opt_op"] = opts["d_opt"].minimize(losses["d_losses"],
            #                                              var_list=self.d.variables + self.d_refer.variables,
            #                                              colocate_gradients_with_ops=True)
            # count_parameters(self.d.variables , name="current_discriminator_gen")
            # count_parameters(self.d_refer.variables , name="current_discriminator_refer")
            opt_ops["d_opt_op"] = opts["d_opt"].minimize(losses["d_losses"],
                                                         var_list=self.d.variables,
                                                         colocate_gradients_with_ops=True)
            count_parameters(self.d.variables, name="current_discriminator_gen")

            opt_ops["c_opt_op"] = opts["c_opt"].minimize(losses['c_losses'], var_list=self.d.variables,
                                                         colocate_gradients_with_ops=True)

            # opt_ops["fzl_opt_op"] = opts["fzl_opt"].minimize(losses['fzl_losses'], var_list=self.g.variables_fzl,
            #                                              colocate_gradients_with_ops=True)
            # opt_ops["recons_opt_op"] = opts["recons_opt"].minimize(losses['recons_losses'], var_list=self.g.variables,
            #                                              colocate_gradients_with_ops=True)

        return opt_ops

    def init_train(self, learning_rate_g, learning_rate_d, learning_rate=1e-4, beta1=0.0, beta2=0.9):
        """
        Initialize training by constructing the summary, loss and ops
        :param learning_rate: The learning rate for the Adam optimizer
        :param beta1: Beta1 for the Adam optimizer
        :param beta2: Beta2 for the Adam optimizer
        :return: summary op, losses and training ops.
        """

        losses = dict()
        opts = dict()

        if self.num_gpus > 0:
            device_ids = ['/gpu:{}'.format(i) for i in range(self.num_gpus)]
        else:
            device_ids = ['/cpu:0']
        for gpu_id, device_id in enumerate(device_ids):
            with tf.device(device_id):
                total_losses = self.loss(gpu_id=gpu_id)
                for key, value in total_losses.items():
                    if key not in losses.keys():
                        losses[key] = [value]
                    else:
                        losses[key].append(value)

        for key in list(losses.keys()):
            losses[key] = tf.reduce_mean(losses[key], axis=0)
            if key == "g_losses":
                print('g_lr', learning_rate_g)
                opts[key.replace("losses", "opt")] = tf.train.AdamOptimizer(beta1=beta1, beta2=beta2,
                                                                            learning_rate=learning_rate_g)
            elif key == "d_losses":
                opts[key.replace("losses", "opt")] = tf.train.AdamOptimizer(beta1=beta1, beta2=beta2,
                                                                            learning_rate=learning_rate_d)
                print('d_lr', learning_rate_d)
            else:
                opts[key.replace("losses", "opt")] = tf.train.AdamOptimizer(beta1=beta1, beta2=beta2,
                                                                            learning_rate=learning_rate)

            # opts[key.replace("losses", "opt")] = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        summary = tf.summary.merge_all()
        apply_grads_ops = self.train(opts=opts, losses=losses)

        return summary, losses, apply_grads_ops

    def sample_same_images(self):
        """
        Samples images from the DAGAN using input_x_i as image




        conditional input and z_inputs as the gaussian noise.
        :return: Inputs and generated images
        """
        conditional_inputs = self.input_x_i[0]
        support_input = self.input_x_j[0]
        input_global_y_i = self.input_global_y_i[0]
        input_global_x_j_selected = self.input_x_j_selected[0]

        input_y_i = self.input_y_i[0]
        input_y_j = self.input_y_j[0]
        input_global_y_j = self.input_global_y_j[0]
        input_global_y_j_selected = self.input_global_y_j_selected[0]

        classes = self.classes
        #### calculating the d_loss for score of selected samples

        if self.training:
            generated, x_g2, gan_recons_image1, gan_decoder_delta_trs, z1, z2, fake_delta, fake_delta2, decoder_layers, decoder_layers2, real_delta_x1x2, reconstruction_loss_feature, renconstruction_loss_image = \
                self.generate(conditional_images=conditional_inputs,
                              support_input=support_input,
                              input_global_y_i=input_global_y_i,
                              input_global_x_j_selected=input_global_x_j_selected,
                              input_y_i=input_y_i,
                              input_y_j=input_y_j,
                              input_global_y_j_selected=input_global_y_j_selected,
                              selected_classes=self.selected_classes,
                              support_num=self.support_num,
                              classes=classes,
                              z_input11=self.z_inputs,
                              z_input_22=self.z_inputs_2,
                              is_z2=self.is_z2,
                              is_z2_vae=self.is_z2_vae)
            return self.input_x_i[0], self.input_x_j[
                0], generated, gan_recons_image1, gan_decoder_delta_trs, input_y_i, input_global_y_i
        else:
            generated, similarities, d_loss, preds_fake = self.generate(
                conditional_images=conditional_inputs,
                support_input=support_input,
                input_global_y_i=input_global_y_i,
                input_global_x_j_selected=input_global_x_j_selected,
                input_y_i=input_y_i,
                input_y_j=input_y_j,
                input_global_y_j_selected=input_global_y_j_selected,
                selected_classes=self.selected_classes,
                support_num=self.support_num,
                classes=classes,
                z_input11=self.z_inputs,
                z_input_22=self.z_inputs_2,
                is_z2=self.is_z2,
                is_z2_vae=self.is_z2_vae)

            # print('here',preds_fake) (16, 5)
            # few_shot_fake_category = tf.argmax(preds_fake, axis=1)
            # softmax = tf.nn.softmax(preds_fake)
            # few_shot_confidence_score = tf.reduce_max(softmax, axis=1)

            # print('11111',few_shot_fake_category) shape=(16,)
            # print('22222',few_shot_confidence_score) shape=(16,)
            return self.input_x_i[0], self.input_x_j[
                0], generated, input_y_i, input_global_y_i, similarities, similarities, similarities

