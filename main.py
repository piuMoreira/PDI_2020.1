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

def channel_negative(img : np.array, channel):
# Aplica o negativo em um canal específico da imagem, esse  canal é especificado como um parâmetro de entrada
    channel = int(channel)
    img[:,:,channel] = 255 - img[:,:,channel]
    return img

def correlacao_m_por_n(img: np.array, mask: np.array):
    return imgutil.fix_truncate_image_colors(imgutil.apply_mask(img, mask))

def mean_filter(img : np.array, m=9, n=9):
    m, n = int(m), int(n)
    mmask = lambda f, g: np.array([[1/f/g]*f]*g)
    return correlacao_m_por_n(img, mmask(m, n))

def median_filter(img : np.array, maskh=3, maskv=3, extzero=True):
    maskh, maskv = int(maskh), int(maskv)
    return imgutil.round_image_colors(imgutil.apply_mask_func_each_channel(np.median, img, maskh, maskv, ext_zero=extzero))

def sobel_grad(img: np.array, mode='normal'):
    return imgutil.round_image_colors((sobel_h(img, mode) + sobel_v(img, mode))/2)

def sobel_v(img: np.array, mode='normal'):
    a = imgutil.apply_mask(img, mask_default_sobel())
    return imgutil.fix_truncate_image_colors(np.abs(a) if mode.lower() == 'abs' else a)

def sobel_h(img: np.array, mode='normal'):
    a = imgutil.apply_mask(img, mask_default_sobel().T)
    return imgutil.fix_truncate_image_colors(np.abs(a) if mode.lower() == 'abs' else a)

def mask_default_sobel():
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def cross_relation_template(fullimg : np.array, templateimg='images/babooneye.png') -> np.array:
    if isinstance(templateimg, str):
        templateimg = open_image(templateimg)
    
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
        'median_filter': median_filter,
        'mean_filter': mean_filter,
        'sobel': sobel_grad,
        'sobel_v': sobel_v,
        'sobel_h': sobel_h,
        'cross_relation_template': cross_relation_template
        }

    img = open_image(args[0])
    operations = [] # Esteira de operações sobre a imagem

    tmpstack = None # Pilha de argumentos da operação
    ko, ke = '{', '}'
    for o in args[1:]: # Para cada argumento da linha de comando (o SO/Python separa por espaços)
        if not (tmpstack is None): # Se estamos preenchendo a pilha de argumentos
            if o.strip()[-1] == ke: # Último argumento da pilha, então adiciona a pilha a esteira
                operations.append([i.strip() for i in (tmpstack + [o[:-1]]) if i.strip()])
                tmpstack = None # Esvazia a pilha
            else: # Se haverá mais argumentos na função, então apenas empilha e deixa o for continuar
                tmpstack.append(o)
        # Se não estamos preenchendo a pilha, então esperamos o nome da função
        elif ko in o: # Se o nome da função vem acompanhado do "{", então esperamos os argumentos dela
            if o.strip()[-1] == ke: # Se temos "}" aqui, é porque tem apenas 1 argumento.
                operations.append([i.strip() for i in (o[:o.rfind(ko)], o[o.rfind(ko)+1:-1]) if i.strip()])
            else: # Do contrário, cria a pilha de argumentos com o primeiro argumento passado
                tmpstack = [o[:o.rfind(ko)], o[o.rfind(ko)+1:]]
        else:# Se o nome da função não vem acompanhado de "{", então não tem argumentos
            operations.append([o.strip()])

    for o in operations: # Para cada operação, aplica-a em img e retorna em img
        img = omaps[o[0]](img, *o[1:])

    show_image(img)

if __name__ == "__main__":
    main_interpret(argv[1:])
