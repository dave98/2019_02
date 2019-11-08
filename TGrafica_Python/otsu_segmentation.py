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


class otsu:
    def __init__(self):
        self.inner_histogram = np.zeros(256, dtype='int')
        self.prob_ocurrences = np.zeros(256, dtype='float64')
        self.threshold = 0 # Default

    def get_histogram(self, in_image):
        for i in range(0, main_state[0]):
            for j in range(0, main_state[1]):
                self.inner_histogram[int(in_image[i, j])] += 1
        self.prob_ocurrences = self.inner_histogram / (main_state[0] * main_state[1])

    def method_1(self): #Intensive method
        min_T = np.arange(1, 256)
        for T in range(1, 256):
            q1 = np.sum(self.prob_ocurrences[:T]) # Last value in slicing does not include that value
            q2 = np.sum(self.prob_ocurrences[T:256])

            u1 = np.sum(np.multiply( np.arange(1, T+1), self.prob_ocurrences[:T]) / (q1 + 0.00001) )
            u2 = np.sum(np.multiply( np.arange(T+1, 257), self.prob_ocurrences[T:256]) / (q2 + 0.00001))

            v1 = np.sum( np.multiply(np.power(np.arange(1, T+1) - u1, 2) , ((self.prob_ocurrences[:T]) / (q1 + 0.00001)))  )
            v2 = np.sum( np.multiply(np.power(np.arange(T+1, 257) - u2, 2) , ((self.prob_ocurrences[T:256]) / (q1 + 0.00001)))  )
            wv = (q1 *v1) + (q2*v2)
            min_T[T-1] = wv

        result = np.argmin(min_T)+1
        self.threshold = result

    def apply_threshold(self, in_image):
        out_image = in_image.copy()
        for i in range(0, main_state[0]):
            for j in range(0, main_state[1]):
                if in_image[i, j] <= self.threshold:
                    out_image[i, j] = 0
                else:
                    out_image[i, j] = 255
        return out_image

    def describe(self):
        print('Histogram: \n', self.inner_histogram)
        print('ProbaOcurrence: \n', self.prob_ocurrences)



#****************************START HERE**************************************
#---------------- Carga Imagen ------------------------- # Abrimos las imágenes con las que vamos a trabajar
img1 = i_container('Origen', main_state)
#img1.i_open(True, 'cat_1.jpg')
#img1.i_open(True, 'cat_2.jpg')
img1.i_open(True, 'cat_3.png')
img2 = i_container('Binary', secondary_state)
img2.i_open(False, '')

otsu1 = otsu()
#---------------- Proceso -------------------------
otsu1.get_histogram(img1.image)
otsu1.method_1()
temp_img1 = otsu1.apply_threshold(img1.image)

print('Threshold de: ', otsu1.threshold)
#---------------- Resultado -------------------------
for i in range(0, secondary_state[0]): # Por alguna razón no podemos asignar los nuevos valores sino solo haciendo una iteración
    for j in range(0, secondary_state[1]):
        img2.image[i, j] = int(temp_img1[i, j])
        #img4.image[i, j] = int(temp_composed[i, j]*255) # Deletable
        #img3.image[i, j] = int(temp_img2[i, j]*255)

img1.i_show()
img2.i_show()
cv2.waitKey(0)
cv2.destroyAllWindows()
