import numpy as np
import cv2

main_state = [400, 400]
secondary_state = [400, 400]

class i_container: #Administrador de imágenes, no es relevante en fft
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

    def i_show(self, inverted=False):
        if(self.image is not None):
            if inverted:
                cv2.imshow(self.name, cv2.bitwise_not(self.image))
            else:
                cv2.imshow(self.name, self.image)

        else:
            print('Not created image to show')


class filter:
    def __init__(self, standard_filter = True, n_filter = 1):
        self.center = np.array([0, 0])
        if standard_filter:
            self.size = 3
            self.filter = np.zeros((size, size), dtype='int')
            for i in range(0, self.filter.shape[0]): #shape[0] = shape[1]
                self.filter[i, int(self.filter.shape[1]/2)] = 1
                self.filter[int(self.filter.shape[0]/2), i] = 1
            self.filter[int(self.filter.shape[0]/2), int(self.filter.shape[1] /2)] = 1
            self.center = [int(self.filter.shape[0]/2), int(self.filter.shape[1] /2)]


        else: # El centro es el primer 2 que se encuentre y que este definido en la tupla
            if n_filter == 0:
                self.size = 3
                self.center = np.array([1, 1])
                self.filter = np1.array([[0, 1, 0],
                                         [0, 1, 0],
                                         [0, 1, 0]])
            elif n_filter == 1:
                self.size = 3
                self.center = np.array([1, 1])
                self.filter = np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]])
            elif n_filter == 2:
                self.size = 3
                self.center = np.array([1, 1])
                self.filter = np.array([[0, 0, 0],
                                        [1, 1, 1],
                                        [0, 0, 0]])
            elif n_filter == 3:
                self.size = 2
                self.center = np.array([1, 0])
                self.filter = np.array([[1, 1],
                                        [0, 1]])
            else:
                self.size = 5
                self.center = np.array([2, 2])
                self.filter = np.array([[0, 0, 1, 1, 1],
                                    [0, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 1, 0, 0]])
        print('Filtro usado: \n', self.filter)

    def change_filter(self, n_filter = 5):
        if n_filter == 0:
            self.size = 3
            self.center = np.array([1, 1])
            self.filter = np1.array([[0, 1, 0],
                                     [0, 1, 0],
                                     [0, 1, 0]])
        elif n_filter == 1:
            self.size = 3
            self.center = np.array([1, 1])
            self.filter = np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]])
        elif n_filter == 2:
            self.size = 3
            self.center = np.array([1, 1])
            self.filter = np.array([[0, 0, 0],
                                    [1, 1, 1],
                                    [0, 0, 0]])
        elif n_filter == 3:
            self.size = 2
            self.center = np.array([1, 0])
            self.filter = np.array([[1, 1],
                                    [0, 1]])
        else:
            self.size = 5
            self.center = np.array([2, 2])
            self.filter = np.array([[0, 0, 1, 1, 1],
                                    [0, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 1, 0, 0]])
        print('Filtro cambiado a: \n', self.filter)

    def print_filter(self):
        print(self.filter)
        print('Center ', self.center)

    def to_binary(self, i_image):
        result = i_image.image / 255
        result = (np.rint(result)).astype('uint8')
        #print(result)
        return result

    def apply_erosion(self, p_image):
        image_result = p_image.copy()
        for i in range(0,  p_image.shape[0] - int(self.filter.shape[0]) ):
            for j in range(0, p_image.shape[1] - int(self.filter.shape[1]) ):

                valid_q = True
                for l in range(0, self.filter.shape[0]):
                    for k in range(0, self.filter.shape[1]):
                            if (self.filter[l, k] == 1) and (p_image[i + l, j + k] != self.filter[l, k]):
                                valid_q = False

                if valid_q:
                    image_result[i + self.center[0], j + self.center[1]] =  1
                else:
                    image_result[i + self.center[0], j + self.center[1]] =  0
        return image_result


    def apply_dilation(self, p_image):
        image_result = p_image.copy()
        for i in range(0,  p_image.shape[0] - int(self.filter.shape[0]) ):
            for j in range(0, p_image.shape[1] - int(self.filter.shape[1]) ):

                if p_image[i, j] == 1:
                    for l in range(0, self.filter.shape[0]):
                        for k in range(0, self.filter.shape[1]):
                            if self.filter[l, k] == 1:
                                image_result[i + l, j + k] = self.filter[l, k]

        return image_result


#****************************START HERE**************************************
#---------------- Carga Imagen ------------------------- # Abrimos las imágenes con las que vamos a trabajar
img1 = i_container('Origen', main_state)
img1.i_open(True, 'b_1.jpg')
img2 = i_container('A binarios', secondary_state)
img2.i_open(False, '')
img3 = i_container('Inversa', secondary_state)
img3.i_open(False, '')
img4 = i_container('Filtro Seleccionado', secondary_state)
img4.i_open(False, '')


#---------------- Proceso -------------------------
filter_1 = filter(standard_filter=False, n_filter = 5)
temp_img = filter_1.to_binary(img1)
temp_composed = np.mod( (temp_img.copy()) + 1, 2) #INVERSA
temp_img2 = temp_img.copy()
#----------------- Erosion ------------------------
#temp_img2 = filter_1.apply_erosion(temp_img2)
#----------------- Dilation ------------------------
#temp_img2 = filter_1.apply_dilation(temp_img2)
#--------------------------- Contorno --------------------------------
#temp_img2 = filter_1.apply_erosion(temp_img2)
#temp_img2 = temp_img - temp_img2
#---------------------------- Hit or Miss -------------------------
temp_img2 = temp_img.copy()
temp_img2 = filter_1.apply_erosion(temp_img2)
filter_1.change_filter(n_filter=2)
temp_composed = filter_1.apply_erosion(temp_composed)

for i in range(0, temp_img2.shape[0]): # Interseccion operation
    for j in range(0, temp_img2.shape[1]):
        if temp_img2[i, j] == temp_composed[i, j]:
            temp_img2[i, j] = 1
        else:
            temp_img2[i, j] = 0

#---------------- Resultado (Don't touch) -------------------------
for i in range(0, secondary_state[0]): # Por alguna razón no podemos asignar los nuevos valores sino solo haciendo una iteración
    for j in range(0, secondary_state[1]):
        img2.image[i, j] = int(temp_img[i, j]*255) # Binarios
        img3.image[i, j] = int(temp_composed[i, j]*255) # Inversa
        img4.image[i, j] = int(temp_img2[i, j]*255) # Filtro aplicado

img1.i_show()
img2.i_show()
img3.i_show()
img4.i_show()
cv2.waitKey(0)
cv2.destroyAllWindows()



















#********************************END HERE************************************
