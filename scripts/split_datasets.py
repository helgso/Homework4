import numpy as np
import pickle
import random


#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#                                           IMPORT DATAS                                            #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################


#Load images with numpy
images_train = np.load('/home/ncls/Documents/IFT6390/Devoirs/Devoir 4/input/train_images.npy', encoding='latin1')



#Train Set reformat
complete_Train_set = [None] * len(images_train)

for i in range(len(images_train)):
    # Reshaping image to 100x100
    image = (images_train[i][1]).reshape(100, 100)

    complete_Train_set[i] = image

complete_Train_set = np.array(complete_Train_set)


# #Load test images with numpy
# images_test = np.load('/home/ncls/Documents/IFT6390/Devoirs/Devoir 4/input/test_images.npy', encoding='latin1')
#
#
#
# #Test Set reformat
#
# complete_Test_set = [None] * len(images_test)
#
# for i in range(len(images_test)):
#     # Reshaping image to 100x100
#     image = (images_test[i][1]).reshape(100, 100)
#
#     complete_Test_set[i] = image
#
#
# complete_Test_set = np.array(complete_Test_set)
#




#Load denoised images
D_noise_complete_Train_set = np.array(pickle.load(open("../data/denoised_train_validation_images.p", "rb")))
D_noise_complete_test_set = np.array(pickle.load(open("../data/denoised_test_images.p", "rb")))


#Load labels
train_labels = np.array(pickle.load(open("../data/cat_examples_complete.p", "rb")))

train_labels = np.genfromtxt('/home/ncls/Documents/IFT6390/Devoirs/Devoir 4/Homework4-master/data/train_labels.csv',
                           names=True, delimiter=',', dtype=[('Id', 'i8'), ('Category', 'S5')])

train_labels = train_labels['Category']


#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#                                            MAKE SETS                                              #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################

# Nombre de classes
n_classes = 31
# Nombre de points d'entrainement
# n_train = pca_reduc_TRAIN_vectors.shape[0]
n_train = 8000


# Commenter pour avoir des resultats non-deterministes
random.seed(3395)
# Determiner au hasard des indices pour les exemples d'entrainement et de test
inds = list(range(complete_Train_set.shape[0]))

random.shuffle(inds)
train_inds = inds[:n_train]
test_inds = inds[n_train:]



Validation_train_labels_categories = train_labels[train_inds]
Validation_test_labels_categories = train_labels[test_inds]

pickle.dump( Validation_train_labels_categories, open( '../data/Validation_train_labels_categories.p', "wb" ) )
pickle.dump( Validation_test_labels_categories, open( '../data/Validation_test_labels_categories.p', "wb" ) )





# Separer les donnees dans les deux ensembles
Validation_train_set = complete_Train_set[train_inds,:]
Validation_test_set = complete_Train_set[test_inds,:]

pickle.dump( Validation_train_set, open( '../data/Validation_train_set.p', "wb" ) )
pickle.dump( Validation_test_set, open( '../data/Validation_test_set.p', "wb" ) )

#For memory reasons
Validation_train_set = 0
Validation_test_set = 0







D_noised_Validation_train_set = D_noise_complete_Train_set[train_inds,:]
D_noised_Validation_test_set = D_noise_complete_Train_set[test_inds,:]

pickle.dump( D_noised_Validation_train_set, open( '../data/D_noised_Validation_train_set.p', "wb" ) )
pickle.dump( D_noised_Validation_test_set, open( '../data/D_noised_Validation_test_set.p', "wb" ) )

#For memory reasons
D_noised_Validation_train_set = 0
D_noised_Validation_test_set = 0


