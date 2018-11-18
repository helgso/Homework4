import numpy as np # linear algebra
import matplotlib.pyplot as plt
#import Bonnes_fonctions
import pickle
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__)))

training_seg_im = pickle.load( open( parent_dir+"/segmentation_Pickles/train_set.p", "rb" ) )
test_seg_im = pickle.load( open( parent_dir+"/segmentation_Pickles/test_set.p", "rb" ) )



#Load labels
train_labels = np.genfromtxt('data/train_labels.csv', names=True, delimiter=',', dtype=[('Id', 'i8'), ('Category', 'S5')])


label_cat = train_labels['Category']

diff_cat = np.unique(label_cat)



#Visualize one image for every categories
for i in range(30):
    print(diff_cat[i])

    cat_number = i

    ex_idx = np.random.randint(1, 100)

    plt.imshow(training_seg_im[cat_number][ex_idx])
    plt.show()



#Visualize ramdom images of the test set
nb_images = 15
for i in range(nb_images):

    ex_idx = np.random.randint(0, 9999)

    plt.imshow(test_seg_im[ex_idx])
    plt.show()
