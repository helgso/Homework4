import numpy as np
import matplotlib.pyplot as plt
import pickle
import cleaning_method




#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#                                          IMPORT DATA                                              #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################


#Load train_images
images_train = np.load('input/train_images.npy', encoding='latin1')


#Load test images with numpy
images_test = np.load('/home/ncls/Documents/IFT6390/Devoirs/Devoir 4/input/test_images.npy', encoding='latin1')



# Dimension [0] gives a list where every example is sorted in regards to its category
#
# ----->   Ex: data[0][i] gives a vector that contains all the train images indexes for the i th category
#
# Dimension[1][i] gives the complete name of the i th category
sorted_train_set_idx = pickle.load( open( "pickles/sorted_train_set_idx.p", "rb" ) )



#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#                                         TRAINING SET                                              #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################

denoised_train_validation_images = [None] * len(images_train)


for i in range(len(images_train)):

    if(i%1000 == 0):
        print('train', i)

    train_image = (images_train[i][1]).reshape(100, 100)

    clean_im = cleaning_method.clean_pic(train_image, 5, 5)
    denoised_train_validation_images[i] = clean_im


pickle.dump( denoised_train_validation_images, open( 'denoised_train_validation_images.p', "wb" ) )


#OLD STUFF

# denoised_train_examples = [None] * 31
# for i in range(31):
#
#     print(i)
#
#     cat_num = i
#
#     clean_list = [None] * len(sorted_train_set_idx[0][cat_num][0])
#
#     for k in range(len(sorted_train_set_idx[0][cat_num][0])):
#
#
#         idx_vector = sorted_train_set_idx[0][cat_num]
#
#         idx_vector = np.array(idx_vector)
#
#         idx = idx_vector[0][k]
#
#
#         #Reshaping image to 100x100
#         train_image = (images_train[idx][1]).reshape(100, 100)
#
#         clean_im = cleaning_method.clean_pic(train_image, 5, 20)
#         clean_list[k] = clean_im
#
#
#     denoised_train_examples[i] = clean_list
#
# denoised_im_train_list = [denoised_train_examples,sorted_train_set_idx[1] ]
#
# pickle.dump( denoised_im_train_list, open( 'denoised_im_train_list.p', "wb" ) )
#
# print('DONE!')
#



#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#                                             TEST SET                                              #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################


denoised_test_images = [None] * len(images_test)

for i in range(len(images_test)):


    if(i%1000 == 0):
        print('test', i)


    test_image = (images_test[i][1]).reshape(100, 100)

    clean_im = cleaning_method.clean_pic(test_image, 5, 5)
    denoised_test_images[i] = clean_im

    plt.imshow(clean_im)
    plt.show()

pickle.dump(denoised_test_images, open('denoised_test_images.p', "wb"))



