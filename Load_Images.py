import numpy as np
import cv2

# para reducir las dimensiones... (max_pooling)
def max_pool(image):
    total = []

    for i in range(3):
        mat = image[i]

        M, N = mat.shape
        K = 2
        L = 2

        MK = M // K
        NL = N // L

        alpha = mat[:MK * K, :NL * L].reshape(MK, K, NL, L).max(axis=(1, 3))

        total.append(alpha)
    return np.asarray(total)

# para normlizar imagenes...
def normalize(image):
    C, H, W = image.shape
    mascara = np.ones((C, H, W)) * 255
    imagen_normalizada = image / mascara

    return imagen_normalizada


## para rotar las imagenes
def rotate_images(image, angle):
    (h, w) = image.shape[:2]
    medio = (w/2, h/2)

    M = cv2.getRotationMatrix2D(medio, angle, 1.0)
    imagen_rotada = cv2.warpAffine(image, M, (w, h))
    return imagen_rotada

## para invertir las imagenes
def flip_images(image):
    h_img = image.copy()
    v_img = image.copy()
    h_img = cv2.flip(image, 0)
    v_img = cv2.flip(image, 1)
    return h_img, v_img

def open_images(n_imag):
    ## opening images...
    data = []

    path = "000"


    for i in range(n_imag):
        path1 = int(path)
        path1 += 1
        path2 = str(path1)

        image = cv2.imread(
            "C:/Users/axel_/Desktop/Varios/BCCD_Dataset/BCCD/JPEGImages/BloodImage_0000" + path2 + ".jpg")
        imarray = np.array(image)

        data_array = np.asarray(imarray)
        images = data_array.reshape(3, 480, 640)

        aa = images[:, :, 80:560]
        imagess = normalize(aa)
        image = max_pool(imagess)
        imagen = max_pool(image)

        img_rotada1 = rotate_images(imagen, angle=90)
        img_rotada2 = rotate_images(imagen, angle=180)
        img_rotada3 = rotate_images(imagen, angle=270)

        h_img1, v_img1 = flip_images(img_rotada1)
        h_img2, v_img2 = flip_images(img_rotada2)
        h_img3, v_img3 = flip_images(img_rotada3)

        data.append(imagen)

        data.append(img_rotada1)
        data.append(img_rotada2)
        data.append(img_rotada3)

        data.append(h_img1)
        data.append(v_img1)
        data.append(h_img2)
        data.append(v_img2)
        data.append(h_img3)
        data.append(v_img3)

        path2 = path

    total = np.asarray(data)
    return total

