import numpy as np
from PIL import Image, ImageDraw

def _to_img(img : np.array):
    return Image.fromarray(img)

def open_image(path: str) -> np.array:
    ''' Carrega uma imagem para um array do numpy.
    '''
    return np.array(Image.open(path))

def save_image(img : np.array, path : str):
    ''' Salva uma imagem expressa em array do numpy em um arquivo.
    '''
    return _to_img(img).save(path)

def show_image(img : np.array):
    ''' Método para visualizar uma imagem a partir do terminal.
	Obs. até então, só funciona no linux.
    '''
    _to_img(img).show()

def draw_rectangle(img : np.array, *args, **kwargs):
    ''' Consulte PIL.ImageDraw.rectangle
    '''
    im = _to_img(img)
    draw = ImageDraw.Draw(im)
    draw.rectangle(*args, **kwargs)
    return np.array(im)
