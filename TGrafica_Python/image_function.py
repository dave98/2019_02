import numpy as np
import cv2
from matplotlib import pyplot as plt

main_state = [500, 500]
secondary_state = [500, 500]

class i_container:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.route = None
        self.image = None

    def i_open(self, is_read, route):
        if(is_read == True):
            self.route = route
            self.image = cv2.imread(self.route, 0)
            self.image = cv2.resize(self.image, (self.size[0], self.size[1]))
        else:
            self.image = np.zeros((self.size[0], self.size[1], 1), np.uint8)

    def i_show(self):
        if(self.image is not None):
            cv2.imshow(self.name, self.image)
            #cv2.waitKey(0)
            #cv2.destroyWindow(self.name)
        else:
            print('Not created image to show')

    def p_info(self):
        print('Title: ', self.name)
        print('Size of: ', self.size[0], 'x', self.size[1])
        print('Route: ', self.route)

    def p_image_info(self):
        if(self.image is not None):
            for x in self.image:
                for y in self.image[x]:
                    print(self.image[x, y])
        else:
            print('Not created image to show')





class i0_mask:
    #Permite crear mascaras de 3x3 o 5x5
    def __init__(self, size = 3):
        self.size = size
        self.acc_prom = 0
        if self.size == 5:
            self.matrix = np.zeros((self.size, self.size), dtype=int)
        else:
            self.matrix = np.zeros((3, 3), dtype=int)

    # m_type indica la clase de mascara seleccionada.
    def set_mask(self, m_type = 1):
        if self.size == 3:
            if m_type == 1:
                self.matrix = np.array([[1, 1, 1],
                                        [1, 2, 1],
                                        [1, 1, 1]], dtype = int)
                self.acc_prom = 10

    # Ambas imagenes son del mismo tamaño, usar varaibles globales
    def apply_matrix(self, i_img, o_img):
        for y in range(1, main_state[0] - 1):
            for x in range(1, main_state[1] - 1):
                #o_img.image[y, x] = self
                print('hola')

    def print_inner_matrix(self):
        print(self.matrix)



#Just works for garyscale images
class i0_function:
    def f_log(self, i_img, o_img):
        if(i_img.image is not None and o_img.image is not None):
            for x in range(main_state[0]):
                for y in range(main_state[1]):
                    o_img.image[x, y] = int(        (  np.log2(( (   float(i_img.image[x, y]) /  254 + 1))) - 1 )             * 255)
        else:
            print('Cannot open images')

    def f_exp(self, i_img, o_img):
        if(i_img.image is not None and o_img.image is not None):
            for x in range(main_state[0]):
                for y in range(main_state[1]):
                    o_img.image[x, y] = int(         (np.power(2, ((  float(i_img.image[x, y]) /  255 ))) - 1 )            * 255)
        else:
            print('Cannot open images')


    def f_negat(self, i_img, o_img):
        if(i_img.image is not None and o_img.image is not None):
            for x in range(main_state[0]):
                for y in range(main_state[1]):
                    o_img.image[x, y] = int(  (-1 * (i_img.image[x, y] /  255)) * 255)
        else:
            print('Cannot open images')

    def f_pow(self, i_img, o_img):
        if(i_img.image is not None and o_img.image is not None):
            for x in range(main_state[0]):
                for y in range(main_state[1]):
                    o_img.image[x, y] = int( np.power( ((  float(i_img.image[x, y]) /  255)) + 0.01, 2) * 255)
        else:
            print('Cannot open images')

    def f_i_pow(self, i_img, o_img):
        if(i_img.image is not None and o_img.image is not None):
            for x in range(main_state[0]):
                for y in range(main_state[1]):
                    o_img.image[x, y] = int( np.power( ((i_img.image[x, y] /  255)) + 0.01, 1/2) * 255)
        else:
            print('Cannot open images')

#Start here ---------------------------------------------------------
img1 = i_container( 'img_1', main_state)

#Cargar imagen, cualquiera.
img1.i_open(True, 'lena.jpg')

img2 = i_container('img_2', secondary_state)
img2.i_open(False, '')

#Creando clase función 
f_function = i0_function()
f_function.f_negat(img1, img2)
#f_function.f_log(img1, img2)
#f_function.f_exp(img1, img2)
#f_function.f_pow(img1, img2)
#f_function.f_i_pow(img1, img2)

img1.i_show()
img2.i_show()

cv2.waitKey(0)
cv2.destroyAllWindows()
