import numpy as np
import pickle


def conf_matrix(etiquettesTest, etiquettesPred):
    n_classes = int(max(etiquettesTest+1))
    matrix = np.zeros((n_classes, n_classes))

    for (test, pred) in zip(etiquettesTest, etiquettesPred):
        matrix[int(test), int(pred)] += 1

    return matrix


#Chose the noisy or denoised test set comment the other

#Original images
test_set = np.array(pickle.load(open("../data/Validation_test_set.p", "rb")))

#Denoised images
test_set = np.array(pickle.load(open("../data/D_noised_Validation_test_set.p", "rb")))



#Test labels Ground Truth
test_labels_gt = np.array(pickle.load(open("../data/Validation_test_labels_categories.p", "rb")))





#Test labels Ground Truth
test_labels_gt = np.array(pickle.load(open("../data/Validation_test_labels_categories.p", "rb")))




#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#  # Make the predictions on the test data an store in test_predictions_vector                      #                                                                            #
#                                                                                                   #
#####################################################################################################

test_predictions_vector = np.zeros(test_set.shape[0])





results = (1.0 - np.equal(test_labels_gt, test_predictions_vector)).mean() * 100.0

print("Taux d'erreur sur l'ensemble de test", (results))


conf_mat = (conf_matrix(test_labels_gt, test_predictions_vector))



