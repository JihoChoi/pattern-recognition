import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.linalg as la
from PIL import Image

#
# Principal Component Analysis (PCA)
#
# Jiho Choi (jihochoi@snu.ac.kr)
#

def reduce_image_rgb(image_rgb, num_pc):

    # image_r, image_g, image_b = image_rgb[0], image_rgb[1], image_rgb[2]
    result = []

    for image_c in image_rgb:  # for each channel

        ## Covariance Matrix
        covariance_mat = np.cov(image_c - np.mean(image_c , axis=1))

        ## Eigenvalue & Eigenvector
        # eig_vals, eig_vecs = la.eig(covariance_mat)
        eig_val, eig_vec = np.linalg.eigh(covariance_mat)
        num_org_pc = np.size(eig_vec, axis =1)  # component

        index = np.argsort(eig_val)[::-1]
        eig_val = eig_val[index]
        eig_vec = eig_vec[:,index]

        if num_pc < num_org_pc:
            eig_vec = eig_vec[:, range(num_pc)]

        cost = np.dot(eig_vec.T, image_c - np.mean(image_c , axis=1))
        reduced_color = np.dot(eig_vec, cost) + np.mean(image_c, axis = 1).T
        reduced_color = np.uint8(np.absolute(reduced_color))
        result.append(reduced_color)

    return result

def reduce_image_gray(image_rgb, num_pc):
    R, G, B = image_rgb[0], image_rgb[1], image_rgb[2]
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.show()




def pca(image_file=None, num_pc=10):

    ## Prepare Image
    image = scipy.misc.imread(image_file)
    print(type(image), image.shape)
    # print(image)

    ## Extract RGB

    image_r, image_g, image_b = image[:,:,0], image[:,:,1], image[:,:,2]
    print(type(image_r), image_r.shape)  # 400x400 (x1)
    # print(image_r)


    # =======
    #   GRAY
    # =======
    # TODO
    # reduced_image = reduce_image_gray([image_r, image_g, image_b], num_pc)

    # =======
    #   RGB
    # =======
    image_r, image_g, image_b = reduce_image_rgb([image_r, image_g, image_b], num_pc)
    reduced_image = Image.fromarray(np.dstack((image_r, image_g, image_b)))

    # reconstruction.show()
    # reduced_image.save('./out/' + image_file + '_' + str(num_pc) + '.png')

    if not os.path.exists('./out'):
        os.makedirs('./out')

    reduced_image.save("./out/{0}_{1}.png".format(image_file, str(num_pc)))




def main():

    input_image_path = "01_snu.JPG"  # 400x400x3
    # input_image_path = "02_gen_circle_plot.png"  # 400x400x4

    for num_pc in range(0,150,10):
        pca(input_image_path, num_pc)

    for num_pc in [0, 1, 2, 5, 10, 20, 30, 40, 50]:
        pca(input_image_path, num_pc)


    pca(input_image_path, num_pc=5)





print("============================================")
print("     Principal Component Analysis (PCA)     ")
print("============================================")

print("\n==========="); print("   START"); print("===========\n")
if __name__ == '__main__':
    main()
print("\n==========="); print("   E N D"); print("===========\n")