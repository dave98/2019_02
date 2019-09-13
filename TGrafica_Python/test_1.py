import numpy as np
import cv2
from matplotlib import pyplot as plt

main_state = [300, 300]
secondary_state = [600, 600]

y_base_center = int(secondary_state[0]/2) - int(main_state[0]/2) #Almacena el punto de inicio para el dibujo
x_base_center = int(secondary_state[1]/2) - int(main_state[1]/2)

def is_pixel_in_area(n_position):
    if(n_position[0] < 0 or n_position[0] > secondary_state[0] - 1):
        return False
    elif(n_position[1] < 0 or n_position[1] > secondary_state[1] - 1):
        return False
    else:
        return True

#Func -> Centrar imagen
#Datos pasados por referencia
def f_img_centering(f_img_base, f_img_destiny):
    for y in range(0, main_state[0]):
        for x in range(0, main_state[1]):
            f_img_destiny[y + y_base_center, x + x_base_center] = f_img_base[y, x]

#Func -> Mover una imagen dentro del plano de la segunda imagen
#img, img, array
def f_img_mover(f_img_base, f_img_destiny, t_traslation):
    temp_pos = [0, 0]
    for y in range(0, main_state[0]):
        for x in range(0, main_state[1]):
            temp_pos = [y + y_base_center + t_traslation[0], x + x_base_center + t_traslation[1]]
            if(is_pixel_in_area(temp_pos)):
                f_img_destiny[temp_pos[0], temp_pos[1]] = f_img_base[y, x]

#Func -> Rotar una imagen a partir del plano x
def f_img_rotar(f_img_base, f_img_destiny, t_angle):
    t_angle = -1 * (t_angle * np.pi / 180)
    sin_angle = (np.sin(t_angle))
    cos_angle = (np.cos(t_angle))
    temp_pos = [0, 0]

    for y in range(0, main_state[0]):
        for x in range(0, main_state[1]):
            temp_pos = [ int(( (y - main_state[0] / 2)*cos_angle) + ( (x - main_state[1] / 2)*sin_angle) + y_base_center), int(( (x - main_state[1] / 2)*cos_angle) - ( (y - main_state[0] / 2)*sin_angle) +  x_base_center)]
            if(is_pixel_in_area(temp_pos)):
                f_img_destiny[temp_pos[0], temp_pos[1]] = f_img_base[y, x]

def f_img_scale(f_img_base, f_img_destiny, t_scala):
    temp_pos = [0, 0]
    for y in range(0, main_state[0]):
        for x in range(0, main_state[1]):
            temp_pos = [ int(((y - main_state[0] / 2)*t_scala[0]) + y_base_center), int(((x - main_state[1] / 2)*t_scala[1]) + x_base_center)]
            if(is_pixel_in_area(temp_pos)):
                f_img_destiny[temp_pos[0], temp_pos[1]] = f_img_base[y, x]


#Carga de imagen a grises
img = cv2.imread('lena.jpg', 1)
#Escalando imágenes:
img = cv2.resize(img, (main_state[0], main_state[1]))
#Imagen en blanco
result_image = np.zeros((secondary_state[0], secondary_state[1], 3), np.uint8)

#for y in range(0, secondary_state[0]):
    #for x in range(0, secondary_state[1]):
        #result_image[y, x] = [255, 0, 255]

#*****************************EJECUCION**************************

#f_img_mover(img, result_image, [50, 50]) # Mover en Y y X
#f_img_rotar(img, result_image, 180) # Rotar en grados
f_img_scale(img,result_image, [1.5, 1.5]) # Escala

cv2.imshow('image', img)
cv2.imshow('results', result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
