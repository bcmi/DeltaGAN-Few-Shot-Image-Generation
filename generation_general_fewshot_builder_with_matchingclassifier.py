import utils.interpolations as interpolations
import tqdm
from utils.storage import *
from tensorflow.contrib import slim

from dagan_networks_wgan_with_matchingclassifier import *
from utils.sampling_with_matchingclassifier import *
import numpy as np

from densenet_classifier import densenet_classifier

'''
forming episode setting, calculating the mean accuracy of each episode, in each episode training the training epoch of the classifier is setting as 5
'''
class ExperimentBuilder(object):
    def __init__(self, parser, data):
        tf.reset_default_graph()
        args = parser.parse_args()
        self.continue_from_epoch = args.continue_from_epoch
        self.experiment_name = args.experiment_title
        self.saved_models_filepath, self.log_path, self.save_image_path = build_experiment_folder(self.experiment_name)
        self.num_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        gen_depth_per_layer = args.generator_inner_layers
        discr_depth_per_layer = args.discriminator_inner_layers
        self.z_dim = args.z_dim
        self.num_generations = args.num_generations
        self.dropout_rate_value = args.dropout_rate_value
        self.data = data
        self.reverse_channels = False
        # self.support_number = args.support_number
        self.classification_total_epoch = args.classification_total_epoch
        image_channel = data.image_channel
        self.use_wide_connections = args.use_wide_connections

        generator_layers = [64, 64, 128, 128]
        self.discriminator_layers = [64, 64, 128, 128]

        gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer]
        self.discr_inner_layers = [discr_depth_per_layer, discr_depth_per_layer, discr_depth_per_layer,
                              discr_depth_per_layer]
        generator_layer_padding = ["SAME", "SAME", "SAME", "SAME"]

        image_height = data.image_height
        image_width = data.image_width
        image_channel = data.image_channel


        self.support_number = args.support_number
        self.selected_classes = args.selected_classes
        self.general_classification_samples = args.general_classification_samples


        self.classes = tf.placeholder(tf.int32)
        self.selected_class = tf.placeholder(tf.int32)
        self.number_support = tf.placeholder(tf.int32)
        
        self.input_x_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, image_height, image_width,
                                                     image_channel], 'batch')
        self.input_y_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.data.selected_classes], 'y_inputs_bacth')
        self.input_global_y_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.data.testing_classes], 'y_inputs_bacth_global')

        
        self.input_x_j = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.data.selected_classes*self.data.support_number ,image_height, image_width,
                                                     image_channel], 'support')
        self.input_y_j = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.data.selected_classes*self.data.support_number, self.data.selected_classes], 'y_inputs_support')
        self.input_global_y_j = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.data.selected_classes*self.data.support_number, self.data.testing_classes], 'y_inputs_support_global')


        self.input_x_j_selected = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, image_height, image_width,
                                                     image_channel], 'support_discriminator')
        self.input_global_y_j_selected = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.data.testing_classes], 'y_inputs_support_discriminator')

        #### setting placehoder for the matchingGAN, mainly for the support images 
        self.input_y_i_dagan = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.selected_classes], 'y_inputs_bacth_dagan')
        self.input_x_j_dagan = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.support_number ,image_height, image_width, image_channel], 'support_dagan')
        self.input_y_j_dagan = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.support_number, self.data.selected_classes], 'y_inputs_support_dagan')
        self.input_global_y_j_dagan = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.support_number, self.data.testing_classes], 'y_inputs_support_global_dagan')



        

        self.z_input = tf.placeholder(tf.float32, [self.batch_size, 6,6,self.z_dim], 'z-input')
        self.z_input_2 = tf.placeholder(tf.float32, [self.batch_size,6,6, self.z_dim], 'z-input_2')

        self.feed_augmented = tf.placeholder(tf.int32)
        self.feed_confidence = tf.placeholder(tf.int32)
        self.feed_loss_d = tf.placeholder(tf.int32)


        
        

        # self.selected_loss_d = tf.placeholder(tf.int32)
        # self.selected_confidence = tf.placeholder(tf.int32)
        # self.number_augmented = tf.placeholder(tf.int32)
        self.training_phase = tf.placeholder(tf.bool, name='training-flag')
        self.random_rotate = tf.placeholder(tf.bool, name='rotation-flag')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout-prob')
        self.z1z2_training = tf.placeholder(tf.bool, name='z1z2_training-flag')
        self.is_z2 = args.is_z2
        self.is_z2_vae = args.is_z2_vae

        self.is_z2 = args.is_z2
        self.is_z2_vae = args.is_z2_vae
        self.loss_G = args.loss_G
        self.loss_D = args.loss_D
        self.loss_CLA = args.loss_CLA
        self.loss_FSL = args.loss_FSL
        self.loss_KL = args.loss_KL
        self.loss_recons_B = args.loss_recons_B
        self.loss_matching_G = args.loss_matching_G
        self.loss_matching_D = args.loss_matching_D
        self.loss_sim = args.loss_sim
        self.strategy = args.strategy


        self.is_fewshot_setting = args.is_fewshot_setting
        # self.few_shot_episode_classes = args.few_shot_episode_classes
        self.few_shot_episode_classes = args.selected_classes
        self.confidence =args.confidence
        self.loss_d = args.loss_d
        self.augmented_number = args.augmented_number
        self.matching = args.matching
        self.fce = args.fce
        self.full_context_unroll_k = args.full_context_unroll_k
        self.average_per_class_embeddings = args.average_per_class_embeddings
        self.restore_path = args.restore_path
        self.restore_classifier_path = args.restore_classifier_path
        self.episodes = args.episodes_number




       
    
        if self.augmented_number>0:
            dagan = DAGAN(batch_size=self.batch_size, input_x_i=self.input_x_i, input_x_j=self.input_x_j_dagan,
                      input_y_i=self.input_y_i_dagan, input_y_j=self.input_y_j_dagan, input_global_y_i=self.input_global_y_i,
                      input_global_y_j=self.input_global_y_j_dagan,
                      input_x_j_selected=self.input_x_j_selected,
                      input_global_y_j_selected=self.input_global_y_j_selected, \
                      selected_classes=self.selected_classes, support_num=self.support_number,
                      classes=self.data.training_classes,
                      dropout_rate=self.dropout_rate, generator_layer_sizes=generator_layers,
                      generator_layer_padding=generator_layer_padding, num_channels=data.image_channel,
                      is_training=self.training_phase, augment=self.random_rotate,
                      discriminator_layer_sizes=self.discriminator_layers,
                      discr_inner_conv=self.discr_inner_layers, is_z2=self.is_z2, is_z2_vae=self.is_z2_vae,
                      gen_inner_conv=gen_inner_layers, num_gpus=self.num_gpus, z_dim=self.z_dim, z_inputs=self.z_input,
                      z_inputs_2=self.z_input_2,
                      use_wide_connections=args.use_wide_connections, fce=self.fce, matching=self.matching,
                      full_context_unroll_k=self.full_context_unroll_k,
                      average_per_class_embeddings=self.average_per_class_embeddings,
                      loss_G=self.loss_G, loss_D=self.loss_D, loss_KL=self.loss_KL, loss_recons_B=self.loss_recons_B,
                      loss_matching_G=self.loss_matching_G, loss_matching_D=self.loss_matching_D,
                      loss_CLA=self.loss_CLA, loss_FSL=self.loss_FSL, loss_sim=self.loss_sim,
                      z1z2_training=self.z1z2_training)
            self.same_images = dagan.sample_same_images()



        if self.is_fewshot_setting:
            print('fewshot classifier categories:',self.few_shot_episode_classes)
            classifier = densenet_classifier(input_x_i=self.input_x_i, input_y=self.input_y_i, classes=self.few_shot_episode_classes, 
                    batch_size=self.batch_size, layer_sizes=self.discriminator_layers, inner_layers=self.discr_inner_layers, num_gpus=self.num_gpus, 
                    use_wide_connections=args.use_wide_connections,
                    is_training=self.training_phase, augment=self.random_rotate,dropout_rate=self.dropout_rate)
        else:
            print('general classifier categories:',self.data.testing_classes)
            classifier = densenet_classifier(input_x_i=self.input_x_i, input_y=self.input_global_y_i, classes=self.data.testing_classes, 
                    batch_size=self.batch_size, layer_sizes=self.discriminator_layers, inner_layers=self.discr_inner_layers, num_gpus=self.num_gpus, 
                    use_wide_connections=args.use_wide_connections,
                    is_training=self.training_phase, augment=self.random_rotate,dropout_rate=self.dropout_rate)
        
        self.summary, self.losses, self.accuracy, self.graph_ops = classifier.init_train()


        

        self.total_train_batches = int(data.training_data_size / (self.batch_size * self.num_gpus))
        self.total_val_batches = int(data.validation_data_size / (self.batch_size * self.num_gpus))
        self.total_test_batches = int(data.testing_data_size / (self.batch_size * self.num_gpus))
        self.total_gen_batches = int(data.testing_data_size / (self.batch_size * self.num_gpus))
        self.init = tf.global_variables_initializer()
        self.spherical_interpolation = False

        self.tensorboard_update_interval = int(self.total_test_batches/10/self.num_gpus)

        self.z_vectors = np.random.normal(size=(10, self.z_dim))
        self.z_vectors_2 = np.random.normal(size=(10, self.z_dim))

        self.z_inputs = np.random.randn(self.batch_size,self.z_dim)
        self.z_inputs_2 = np.random.randn(self.batch_size,self.z_dim)

        self.total_test_items = int(self.general_classification_samples/self.data.support_number) + 1



    def augmented_images(self,sess,is_augmented):
        image_name = "z2vae{}_z2{}_Net_batchsize{}".format(
            self.is_z2_vae,self.is_z2,self.batch_size)
        x_total_images_list = []
        y_total_fewshot_list = []
        y_total_global_list = []

        x_support_images_list = []
        y_support_fewshot_list = []
        y_support_global_list = []

        x_test_images_list = []
        y_test_fewshot_list = []
        y_test_global_list = []



        with tqdm.tqdm(total=self.total_test_items) as pbar_samp:
            #####conduct the number of episode experiments
            for i in range(1):
                if self.general_classification_samples > 5:
                    x_test_i_selected_classes, x_test_j, y_test_i_selected_classes, y_test_j, y_global_test_i_selected_classes, y_global_test_j = self.data.get_train_batch()
                    global_classes = self.data.training_classes
                else:
                    x_test_i_selected_classes, x_test_j, y_test_i_selected_classes, y_test_j, y_global_test_i_selected_classes, y_global_test_j = self.data.get_test_batch()
                    global_classes = self.data.testing_classes
                x_test_i = x_test_i_selected_classes
                y_test_i = y_test_i_selected_classes
                y_global_test_i = y_global_test_i_selected_classes

                x_batch = x_test_i_selected_classes[:,:,0]
                y_batch = y_test_i_selected_classes[:,:,0]
                y_global_batch = y_global_test_i_selected_classes[:,:,0]


                if is_augmented:
                    x_total_images, y_total_fewshot, y_total_global = sample_generator_for_classifier(num_generations=self.num_generations, 
                                sess=sess,
                                same_images=self.same_images,
                                input_a=self.input_x_i, 
                                input_b= self.input_x_j_dagan,
                                input_y_i = self.input_y_i_dagan, 
                                input_y_j = self.input_y_j_dagan, 
                                input_global_y_i = self.input_global_y_i,
                                input_global_y_j = self.input_global_y_j_dagan,
                                classes=self.classes,
                                classes_selected = self.selected_class,
                                number_support = self.number_support,
                                z_input=self.z_input,
                                z_input_2 = self.z_input_2,
                                feed_augmented = self.feed_augmented,
                                feed_confidence = self.feed_confidence,
                                feed_loss_d = self.feed_loss_d,
                                selected_global_x_j = self.input_x_j_selected,
                                selected_global_y_j=self.input_global_y_j_selected,
                                # conditional_inputs=x_test_i,
                                # y_input_i = y_test_i,
                                # y_global_input_i = y_global_test_i,
                                conditional_inputs=x_batch,
                                y_input_i = y_batch,
                                y_global_input_i = y_global_batch,
                                support_input=x_test_j, 
                                y_input_j = y_test_j,
                                y_global_input_j = y_global_test_j,
                                classes_number=self.data.testing_classes,
                                selected_classes = self.selected_classes,
                                support_number = self.support_number,
                                z_vectors=self.z_vectors,
                                z_vectors_2 = self.z_vectors_2,
                                augmented_number=self.augmented_number,
                                confidence=self.confidence,
                                loss_d=self.loss_d,
                                input_global_x_j_selected = x_test_j[:,:,0,:,:,:],
                                input_global_y_j_selected = y_global_test_j[:,:,0,:],
                                data=self.data, 
                                batch_size=self.batch_size, 
                                file_name="{}/generation_{}_{}.png".format(self.save_image_path,
                                                                                      image_name,i),
                                dropout_rate=self.dropout_rate,
                                dropout_rate_value=self.dropout_rate_value,
                                training_phase=self.training_phase,
                                z1z2_training=self.z1z2_training,
                                is_training=False,
                                training_z1z2=False,
                                iteration=i,
                                z_dim = self.z_dim)
                    ##### total generated and original images
                    x_total_images_reshape = np.reshape(x_total_images,(self.batch_size*self.data.selected_classes*(self.data.support_number+self.augmented_number),self.data.image_width,self.data.image_height,self.data.image_channel))
                    y_total_fewshot_reshape = np.reshape(y_total_fewshot,(self.batch_size*self.data.selected_classes*(self.data.support_number+self.augmented_number),self.data.selected_classes))
                    y_total_global_reshape = np.reshape(y_total_global,(self.batch_size*self.data.selected_classes*(self.data.support_number+self.augmented_number),global_classes))
                    indices_1 = np.arange(self.batch_size*self.data.selected_classes*(self.data.support_number+self.augmented_number))
                    np.random.shuffle(indices_1)
                    x_total_images_reshape = x_total_images_reshape[indices_1]
                    y_total_fewshot_reshape = y_total_fewshot_reshape[indices_1]
                    y_total_global_reshape = y_total_global_reshape[indices_1]
                    x_total_images_list.append(x_total_images_reshape)
                    y_total_fewshot_list.append(y_total_fewshot_reshape)
                    y_total_global_list.append(y_total_global_reshape)


                ###### support images
                x_test_j_reshape = np.reshape(x_test_j,(self.batch_size*self.data.selected_classes*self.data.support_number,self.data.image_width,self.data.image_height,self.data.image_channel))
                y_test_j_reshape = np.reshape(y_test_j,(self.batch_size*self.data.selected_classes*self.data.support_number,self.data.selected_classes))
                y_global_test_j_reshape = np.reshape(y_global_test_j,(self.batch_size*self.data.selected_classes*self.data.support_number,global_classes))
                indices_2 = np.arange(self.batch_size*self.data.selected_classes*self.data.support_number)
                np.random.shuffle(indices_2)
                x_test_j_reshape = x_test_j_reshape[indices_2]
                y_test_j_reshape = y_test_j_reshape[indices_2]
                y_global_test_j_reshape = y_global_test_j_reshape[indices_2]
                x_support_images_list.append(x_test_j_reshape)
                y_support_fewshot_list.append(y_test_j_reshape)
                y_support_global_list.append(y_global_test_j_reshape)


                ##### test images
                x_test_i_reshape = np.reshape(x_test_i,(self.batch_size*self.data.selected_classes,self.data.image_width,self.data.image_height,self.data.image_channel))
                y_test_i_reshape = np.reshape(y_test_i,(self.batch_size*self.data.selected_classes,self.data.selected_classes))
                y_global_test_i_reshape = np.reshape(y_global_test_i,(self.batch_size*self.data.selected_classes,global_classes))
                indices_3 =  np.arange(self.batch_size*self.data.selected_classes)
                np.random.shuffle(indices_3)
                x_test_i_reshape = x_test_i_reshape[indices_3]
                y_test_i_reshape = y_test_i_reshape[indices_3]
                y_global_test_i_reshape = y_global_test_i_reshape[indices_3]
                x_test_images_list.append(x_test_i_reshape)
                y_test_fewshot_list.append(y_test_i_reshape)
                y_test_global_list.append(y_global_test_i_reshape)




            total_data = {}
            if is_augmented:
                x_total_images_list = np.array(x_total_images_list)
                y_total_fewshot_list = np.array(y_total_fewshot_list)
                y_total_global_list = np.array(y_total_global_list)
                total_images = np.reshape(x_total_images_list,[np.shape(x_total_images_list)[0]*np.shape(x_total_images_list)[1],self.data.image_width,self.data.image_height,self.data.image_channel])
                total_fewshot = np.reshape(y_total_fewshot_list,[np.shape(y_total_fewshot_list)[0]*np.shape(y_total_fewshot_list)[1],np.shape(y_total_fewshot_list)[2]])
                total_global = np.reshape(y_total_global_list,[np.shape(y_total_global_list)[0]*np.shape(y_total_global_list)[1],np.shape(y_total_global_list)[2]])
                total_data['augmented'] = total_images, total_fewshot, total_global

            x_support_images_list = np.array(x_support_images_list)
            y_support_fewshot_list = np.array(y_support_fewshot_list)
            y_support_global_list = np.array(y_support_global_list)
            support_images = np.reshape(x_support_images_list,[np.shape(x_support_images_list)[0]*np.shape(x_support_images_list)[1],self.data.image_width,self.data.image_height,self.data.image_channel])
            support_fewshot = np.reshape(y_support_fewshot_list,[np.shape(y_support_fewshot_list)[0]*np.shape(y_support_fewshot_list)[1],np.shape(y_support_fewshot_list)[2]])
            support_global = np.reshape(y_support_global_list,[np.shape(y_support_global_list)[0]*np.shape(y_support_global_list)[1],np.shape(y_support_global_list)[2]])
            total_data['original_support'] = support_images, support_fewshot, support_global

            x_test_images_list = np.array(x_test_images_list)
            y_test_fewshot_list = np.array(y_test_fewshot_list)
            y_test_global_list = np.array(y_test_global_list)
            test_images = np.reshape(x_test_images_list,[np.shape(x_test_images_list)[0]*np.shape(x_test_images_list)[1],self.data.image_width,self.data.image_height,self.data.image_channel])
            test_fewshot = np.reshape(y_test_fewshot_list,[np.shape(y_test_fewshot_list)[0]*np.shape(y_test_fewshot_list)[1],np.shape(y_test_fewshot_list)[2]])
            test_global = np.reshape(y_test_global_list,[np.shape(y_test_global_list)[0]*np.shape(y_test_global_list)[1],np.shape(y_test_global_list)[2]])
            total_data['testing'] = test_images, test_fewshot, test_global



            ### for network, no use
            total_data['original'] =  [x_test_j, y_test_j, y_global_test_j]
            # if is_augmented:
            #     print()
            #     print('training data shape',np.shape(total_images))
            #     print('testing data shape',np.shape(test_images))
            # else:
            #     print('training data shape',np.shape(support_images))
            #     print('testing data shape',np.shape(test_images))
            return total_data



    def run_experiment(self):
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            sess.run(self.init)
            # self.train_writer = tf.summary.FileWriter("{}/train_classification_logs/".format(self.log_path),
            #                                           graph=tf.get_default_graph())
            # self.valid_writer = tf.summary.FileWriter("{}/validation_classification_logs/".format(self.log_path),
            #                                                graph=tf.get_default_graph())
            

            self.saver = tf.train.Saver()
            if self.augmented_number!=0:
                if self.continue_from_epoch != -1:
                    print('loading trained MatchingGAN model')
                    checkpoint = self.restore_path
                    variables_to_restore = []
                    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                        variables_to_restore.append(var)
                    fine_tune = slim.assign_from_checkpoint_fn(
                        checkpoint,
                        variables_to_restore,
                        ignore_missing_vars=True)
                    fine_tune(sess)


                    print('loading pretrained classifier model')
                    classifier_checkpoint = self.restore_classifier_path

                    classifier_variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_classifier/')
                    fine_tune_classifier = slim.assign_from_checkpoint_fn(
                        classifier_checkpoint,
                        classifier_variables_to_restore,
                        ignore_missing_vars=True)
                    fine_tune_classifier(sess)


            def file_name(accuracy, i):
                file_name = "Fewshot{}_ModelKshot{}_ClassifierNway{}Kshot{}_Augmentednum{}_Epoch{}_Acc{}.ckpt".format(
                self.is_fewshot_setting,self.support_number,self.selected_classes,self.data.support_number,self.augmented_number,i, accuracy)
                return file_name


            self.iter_done = 0
            test_acc_total =[]
            test_loss_total = []
            print('the size of testing set of classifier in few-shot setting',len(self.data.x_test))
            if self.is_fewshot_setting:
                with tqdm.tqdm(total=self.episodes) as pbar_e:
                    for episode in range(self.episodes):
                        if self.augmented_number!=0:
                            total_data = self.augmented_images(sess=sess,is_augmented=True)
                            x_total_images_reshape, y_total_fewshot_reshape, y_total_global_reshape = total_data['augmented']
                            x_test_i_reshape, y_test_i_reshape,y_global_test_i_reshape = total_data['testing']  
                            x_test_j,_,_ = total_data['original']                      
                            if episode ==0:
                                print('fewshot setting, with augmentation',np.shape(y_total_fewshot_reshape),np.shape(y_test_i_reshape))
                        else:
                            total_data = self.augmented_images(sess,False) 
                            x_total_images_reshape, y_total_fewshot_reshape, y_total_global_reshape = total_data['original_support']
                            x_test_i_reshape, y_test_i_reshape,y_global_test_i_reshape = total_data['testing']
                            x_test_j,_,_ = total_data['original'] 
                            if episode ==0:
                                print('fewshot setting, without augmentation',np.shape(y_total_fewshot_reshape),np.shape(y_test_i_reshape))
                        
                        
                        #### each episode
                        best_d_val_loss = np.inf
                        lowest_d_val_accuracy = 0
                        with tqdm.tqdm(total=self.classification_total_epoch) as pbar_c:
                            for i in range(self.classification_total_epoch):
                                train_loss_episode = []
                                test_loss_episode = []
                                train_acc_episode = []
                                test_acc_episode =[]
                                for j in range(self.data.support_number+self.augmented_number):
                                    current_image = x_total_images_reshape[self.batch_size*j:self.batch_size*(j+1)]
                                    current_y = y_total_fewshot_reshape[self.batch_size*j:self.batch_size*(j+1)]
                                    
                                    ### adapt to the number of gpus
                                    current_image_batch = []
                                    current_y_batch = []
                                    for n_batch in range(self.num_gpus):
                                        current_image_batch.append(current_image)
                                        current_y_batch.append(current_y)
                                    current_image_batch = np.array(current_image_batch)
                                    current_y_batch = np.array(current_y_batch)

                                    _,train_loss_value, train_acc_value, train_summary = sess.run([self.graph_ops["loss_opt_op"],self.losses["losses"], self.accuracy,self.summary],
                                        feed_dict={self.input_x_i: current_image_batch,
                                                   self.input_y_i: current_y_batch,
                                                   self.dropout_rate: self.dropout_rate_value,
                                                   self.z_input: self.z_inputs,
                                                   self.z_input_2: self.z_inputs_2,
                                                   self.input_x_j_dagan: x_test_j[:,:,:self.support_number,:,:,:],
                                                   self.z1z2_training:False,
                                                   self.training_phase: True, self.random_rotate: True})

                                    train_loss_episode.append(train_loss_value)
                                    train_acc_episode.append(train_acc_value)

                                for j in range(self.data.selected_classes):
                                    current_image = x_test_i_reshape[self.batch_size*j:self.batch_size*(j+1)]
                                    current_y = y_test_i_reshape[self.batch_size*j:self.batch_size*(j+1)]
                                    ### adapt to the number of gpus
                                    current_image_batch = []
                                    current_y_batch = []
                                    for n_batch in range(self.num_gpus):
                                        current_image_batch.append(current_image)
                                        current_y_batch.append(current_y)
                                    current_image_batch = np.array(current_image_batch)
                                    current_y_batch = np.array(current_y_batch)

                                    test_loss_value, test_acc_value, test_summary = sess.run([self.losses["losses"], self.accuracy,self.summary],
                                        feed_dict={self.input_x_i: current_image_batch,
                                                   self.input_y_i: current_y_batch,
                                                   self.dropout_rate: self.dropout_rate_value,
                                                   self.z_input: self.z_inputs,
                                                   self.z_input_2: self.z_inputs_2,
                                                   self.input_x_j_dagan: x_test_j[:,:,:self.support_number,:,:,:],
                                                   self.z1z2_training:False,
                                                   self.training_phase:False, self.random_rotate:False})
                                    test_loss_episode.append(test_loss_value)
                                    test_acc_episode.append(test_acc_value)

                                total_test_accuracy_mean_classification = np.mean(test_acc_episode)
                                total_test_loss_mean_classification = np.mean(test_loss_episode)

                                if total_test_loss_mean_classification < best_d_val_loss:
                                    best_d_val_loss = total_test_loss_mean_classification
                                    model_name = file_name(total_test_accuracy_mean_classification,episode)
                                    if self.augmented_number!=0:
                                        val_save_path = self.saver.save(sess, "{}/{}".format(
                                            self.saved_models_filepath,model_name))
                                    else:
                                        val_save_path = self.saver.save(sess, "{}/val_no_augmented_{}".format(
                                            self.saved_models_filepath,model_name))
                                    print("valid loss decrease, model trained with augmented data", val_save_path)

                                if total_test_accuracy_mean_classification > lowest_d_val_accuracy:
                                    lowest_d_val_accuracy = total_test_accuracy_mean_classification
                                    if self.augmented_number !=0:
                                        val_save_path = self.saver.save(sess, "{}/val_augmented_{}".format(
                                            self.saved_models_filepath,model_name))
                                    else:
                                        val_save_path = self.saver.save(sess, "{}/val_no_augmented_{}".format(
                                            self.saved_models_filepath,model_name))
                                    print("valid accuracy increase, model trained with augmented data", val_save_path)


                                test_acc_total.append(lowest_d_val_accuracy)
                                test_loss_total.append(best_d_val_loss)
                                iter_out = " {}_epoch total_test_accuracy: {}, total_test_loss:{}".format(i,total_test_accuracy_mean_classification,total_test_loss_mean_classification)
                                pbar_c.set_description(iter_out)
                                pbar_c.update(1)

                        total_test_accuracy_mean_episode = np.mean(test_acc_total)
                        total_test_loss_mean_episode = np.mean(test_loss_total)
                        iter_out = " {}_episode total_test_accuracy: {}, total_test_loss:{}".format(episode,total_test_accuracy_mean_episode,total_test_loss_mean_episode)
                        pbar_e.set_description(iter_out)
                        pbar_e.update(1)


            else:
                best_d_val_loss = np.inf
                lowest_d_val_accuracy = 0
                with tqdm.tqdm(total=self.classification_total_epoch) as pbar_c:
                    for i in range(self.classification_total_epoch):
                        train_loss_classification = []
                        train_acc_classification = []
                        test_acc_classification =[]
                        test_loss_classification = []
                        ##### iteration
                        # with tqdm.tqdm(total=self.total_train_batches) as pbar_b_train:
                        # with tqdm.tqdm(total=1) as pbar_b_train:
                        #     for j in range(1):
                        if self.augmented_number!=0:
                            total_data = self.augmented_images(sess,True) 
                            x_total_images_reshape, y_total_fewshot_reshape, y_total_global_reshape = total_data['augmented']
                            x_test_i_reshape, y_test_i_reshape,y_global_test_i_reshape = total_data['testing'] 
                            x_test_j, y_test_j, y_global_test_j = total_data['original']
                            if i ==0 :
                                print('general setting, with augmentation',np.shape(y_total_global_reshape))
                        else:
                            total_data = self.augmented_images(sess,False) 
                            x_total_images_reshape, y_total_fewshot_reshape, y_total_global_reshape = total_data['original_support']
                            x_test_i_reshape, y_test_i_reshape,y_global_test_i_reshape = total_data['testing'] 
                            x_test_j, y_test_j, y_global_test_j = total_data['original']
                            if i ==0 :
                                print('general setting, without augmentation',np.shape(y_total_global_reshape))
                                
                        for k in range(self.data.support_number+self.augmented_number):
                            current_image = x_total_images_reshape[self.batch_size*k:self.batch_size*(k+1)]
                            current_y = y_total_global_reshape[self.batch_size*k:self.batch_size*(k+1)]
                            
                            ### adapt to the number of gpus
                            current_image_batch = []
                            current_y_batch = []
                            for n_batch in range(self.num_gpus):
                                current_image_batch.append(current_image)
                                current_y_batch.append(current_y)
                            current_image_batch = np.array(current_image_batch)
                            current_y_batch = np.array(current_y_batch)

                            _,train_loss_value, train_acc_value, train_summary = sess.run([self.graph_ops["loss_opt_op"],self.losses["losses"], self.accuracy,self.summary],
                                feed_dict={self.input_x_i: current_image_batch,
                                           self.input_global_y_i: current_y_batch,
                                           self.dropout_rate: self.dropout_rate_value,
                                           self.z_input: self.z_inputs,
                                           self.z_input_2: self.z_inputs_2,
                                           self.input_x_j_dagan: x_test_j[:,:,:self.support_number,:,:,:],
                                           self.z1z2_training:False,
                                           self.training_phase: True, 
                                           self.random_rotate: True})
                            
                            train_loss_classification.append(train_loss_value)
                            train_acc_classification.append(train_acc_value)
                        iter_out_train_batch = " {}_batch_train_accuracy: {}, batch_train_loss:{}".format(i,train_acc_value,train_loss_value)
                        pbar_c.set_description(iter_out_train_batch)
                        pbar_c.update(1)

                        for j in range(self.data.selected_classes):
                            current_image = x_test_i_reshape[self.batch_size*j:self.batch_size*(j+1)]
                            current_y = y_global_test_i_reshape[self.batch_size*j:self.batch_size*(j+1)]
                            ### adapt to the number of gpus
                            current_image_batch = []
                            current_y_batch = []
                            for n_batch in range(self.num_gpus):
                                current_image_batch.append(current_image)
                                current_y_batch.append(current_y)
                            current_image_batch = np.array(current_image_batch)
                            current_y_batch = np.array(current_y_batch)

                            test_loss_value, test_acc_value, test_summary = sess.run([self.losses["losses"], self.accuracy,self.summary],
                                feed_dict={self.input_x_i: current_image_batch,
                                           self.input_global_y_i: current_y_batch,
                                           self.dropout_rate: self.dropout_rate_value,
                                           self.z_input: self.z_inputs,
                                           self.z_input_2: self.z_inputs_2,
                                           self.input_x_j_dagan: x_test_j[:,:,:self.support_number,:,:,:],
                                           self.z1z2_training:False,
                                           self.training_phase:False, self.random_rotate:False})
                            test_loss_classification.append(test_loss_value)
                            test_acc_classification.append(test_acc_value)

                        total_test_accuracy_mean_classification = np.mean(test_acc_classification)
                        total_test_loss_mean_classification = np.mean(test_loss_classification)


                        if total_test_loss_mean_classification < best_d_val_loss:
                            best_d_val_loss = total_test_loss_mean_classification
                            model_name = file_name(total_test_accuracy_mean_classification,i)
                            if self.augmented_number!=0:
                                val_save_path = self.saver.save(sess, "{}/val_augmented_{}".format(
                                    self.saved_models_filepath,model_name))
                            else:
                                val_save_path = self.saver.save(sess, "{}/val_no_augmented_{}".format(
                                    self.saved_models_filepath,
                                    model_name))
                            print("valid loss decrease, model trained with augmented data", val_save_path)

                        if total_test_accuracy_mean_classification > lowest_d_val_accuracy:
                            lowest_d_val_accuracy = total_test_accuracy_mean_classification
                            if self.augmented_number !=0:
                                val_save_path = self.saver.save(sess, "{}/val_augmented_{}".format(
                                    self.saved_models_filepath,
                                    model_name))
                            else:
                                val_save_path = self.saver.save(sess, "{}/val_no_augmented_{}".format(
                                    self.saved_models_filepath,
                                    model_name))
                            print("valid accuracy increase, model trained with augmented data", val_save_path)

                        test_acc_total.append(lowest_d_val_accuracy)
                        test_loss_total.append(best_d_val_loss)
                        iter_out = " {}_epoch total_test_accuracy: {}, total_test_loss:{}".format(i,total_test_accuracy_mean_classification,total_test_loss_mean_classification)
                        pbar_c.set_description(iter_out)
                        pbar_c.update(1)
