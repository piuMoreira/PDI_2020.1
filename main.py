from loader import *
import util as imgutil
from pymlfunc import normxcorr2
from sys import argv


def rgb2yiq(img_in : np.array):
	mat_yiq = np.array(
        [[0.299, 0.587, 0.114], [0.59590059, -0.27455667, -0.32134392], [0.21153661, -0.52273617, 0.31119955]])
	img_out = np.copy(img_in).astype(np.float32)
	
	for i in range(len(img_in)):
		for j in range(len(img_in[0])):
			pixelrgb = img_in[i][j]
			pixelyiq = np.dot(mat_yiq, pixelrgb)
			pixelyiq /= 255
			img_out[i, j] = pixelyiq
			
	return img_out

def yiq2rgb(img_in : np.array):
	mat_rgb = np.array(
        [[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.106, 1.703]])
	img_out = np.copy(img_in)
	
	for i in range(len(img_in)):
		for j in range(len(img_in[0])):
			pixelyiq = img_in[i][j]
			pixelyiq *= 255
			pixelrgb = np.dot(mat_rgb, pixelyiq)
			img_out[i, j] = pixelrgb
	
	return img_out.astype(np.uint8)

def complete_negative(img : np.array):
# Aplica o negativo em todos os canais da imagem
    for channel in range(3):
        img[:,:,channel] = 255 - img[:,:,channel]
    return img

def channel_negative(img : np.array, channel : int):
# Aplica o negativo em um canal específico da imagem, esse  canal é especificado como um parâmetro de entrada
    img[:,:,channel] = 255 - img[:,:,channel]
    return img

def correlacao_m_por_n(img: np.array, mask: np.array):
    return imgutil.fix_truncate_image_colors(imgutil.apply_mask(img, mask))

def sobel_grad(img: np.array):
    return imgutil.round_image_colors((sobel_h(img) + sobel_v(img))/2)

def sobel_v(img: np.array):
    return imgutil.fix_truncate_image_colors(imgutil.apply_mask(img, mask_default_sobel()))

def sobel_h(img: np.array):
    return imgutil.fix_truncate_image_colors(imgutil.apply_mask(img, mask_default_sobel().T))

def mask_default_sobel():
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def cross_relation_template(fullimg : np.array, templateimg : np.array) -> np.array:
    imgzip = zip(fullimg.T.astype(np.float), templateimg.T.astype(np.float)) # Separa as bandas das duas imagens por transposição
    n = np.array([normxcorr2(t, i) for i,t in imgzip]).mean(0).T # Para cada banda, aplica normxcorr2, depois transpõe novamente
    pm, ts = np.unravel_index(n.argmax(), n.shape), templateimg.shape # obtém as coordenadas do ponto de máximo de n
    return draw_rectangle(fullimg, (pm[1] - ts[1], pm[0] - ts[0], pm[1], pm[0]), outline=(255, 0, 127), width=3) # desenha o retângulo
    # return imgutil.round_image_colors(n**2*255)

def main_interpret(args):
    if len(args) < 2:
        print('Uso: python3 examples.py IMAGEM operacao1 [operacao2] [operacao3...]')
        return

    omaps = {
        'rgb2yiq': rgb2yiq,
        'yiq2rgb': yiq2rgb,
        'complete_negative': complete_negative,
        'channel_negative': channel_negative,
        'correlacao_m_por_n': correlacao_m_por_n,
        'sobel': sobel_grad,
        'cross_relation_template': lambda i: cross_relation_template(i, open_image('images/babooneye.png'))
        }

    img = open_image(args[0])
    for o in args[1:]:
        img = omaps[o](img)
    
    show_image(img)

if __name__ == "__main__":
    main_interpret(argv[1:])