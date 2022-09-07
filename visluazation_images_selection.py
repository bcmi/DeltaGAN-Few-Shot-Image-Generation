import scipy.misc as misc
import numpy as np
import time
import numpy as np
from glob import glob
import os
import cv2



def resize(image,image_width):
    # image = np.int(255*image)
    image = cv2.resize(image, (image_width, image_width), interpolation=cv2.INTER_LINEAR)
    return image


def images_selection(file_name, image_width, image_channel, batch_size, num_generations, support_number):
    filenames = glob(os.path.join(file_name, '*.*'))
    fake_categories = len(filenames) * batch_size
    fake_images = np.zeros([fake_categories * num_generations,  image_width,  image_width,  image_channel])
    for i,image_path in enumerate(filenames):
        store_name = file_name + '_split/'
        if not os.path.exists(store_name):
            os.mkdir(store_name)
        current_x = misc.imread(image_path)
        image_size = int(np.shape(current_x)[0]/ batch_size)
        for j in range(batch_size):
            for k in range(support_number+num_generations):
                current_iamge = current_x[image_size*j:image_size*(j+1),image_size*(k):image_size*(k+1)]
                current_iamge = resize(current_iamge,128)
                # if len(np.shape(current_iamge))<3:
                #     current_iamge = np.expand_dims(current_iamge,axis=-1)
                current_name =  store_name + image_path.split('/')[-1].split('png')[0] + 'batch{}_sample{}.png'.format(j,k) 
                misc.imsave(current_name, current_iamge)



file_name = '/media/user/05e85ab6-e43e-4f2a-bc7b-fad887cfe312/meta_gan/MatchingGAN-SelfAttention-XS/VISUALIZATION/vggface/1shot/visual_outputs/'
image_width = 128
image_channel =3
batch_size = 20
num_generations = 60
support_number = 3
images_selection(file_name, image_width, image_channel, batch_size, num_generations, support_number)