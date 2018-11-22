import numpy as np


## Fonction pour centrer la matrice de l'image dans une matrice de plus haut rang
def CenterImage(image, frame_size):


    # Initialize a matrix im1(MxM) with all values equal to zeros
    frame = np.zeros((frame_size, frame_size))

    # add the old image in the center of the frame
    x = np.floor(abs(frame.shape[0] - image.shape[0]) / 2)
    y = np.floor(abs(frame.shape[1] - image.shape[1]) / 2)

    frame[int(x): int(x + image.shape[0]), int(y): int(y + image.shape[1])] = image

    return frame


#Denoise the picture
def clean_pic(image, frame_offset, nb_try):

    framed_im = CenterImage(image, image.shape[0]+2*frame_offset)


    image_bin = framed_im > 150


    image_bin = image_bin.astype(int)

    for i in range(nb_try):

        inspect_win_size = [int(np.random.random_integers(5, 12)),
                            int(np.random.random_integers(5, 12))]


        for j in range(framed_im.shape[0]):
            for k in range(framed_im.shape[0]):

                win_x_start = j
                win_x_stop = inspect_win_size[0] + j
                win_y_start = k
                win_y_stop= inspect_win_size[1] + k



                if (win_x_stop > framed_im.shape[0]):
                    win_x_start = framed_im.shape[0]-inspect_win_size[0]
                    win_x_stop = framed_im.shape[0]


                if (win_y_stop > framed_im.shape[1]):
                    win_y_start = framed_im.shape[1] - inspect_win_size[1]
                    win_y_stop = framed_im.shape[1]

                cleaning_win = image_bin[win_x_start: win_x_stop, win_y_start: win_y_stop]

                ProjVecV_start = np.sum(cleaning_win[:, 0])
                ProjVecV_stop = np.sum(cleaning_win[:, -1])

                ProjVecH_start = np.sum(cleaning_win[0, :])
                ProjVecH_stop = np.sum(cleaning_win[-1, :])

                if (ProjVecV_start == 0 and ProjVecV_stop == 0 and ProjVecH_start == 0 and ProjVecH_stop == 0 ):
                    framed_im[win_x_start:win_x_stop, win_y_start:win_y_stop] = 0

    # look = framed_im[frame_offset:framed_im.shape[0]-frame_offset, frame_offset:framed_im.shape[1]+frame_offset]
    #
    # plt.imshow(look)
    # plt.show()


    return framed_im[frame_offset:framed_im.shape[0]-frame_offset, frame_offset:framed_im.shape[1]+frame_offset]
