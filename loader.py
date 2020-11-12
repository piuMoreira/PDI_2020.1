import numpy as np
from PIL import Image

def open_image(path: str) -> np.array:
    ''' Carrega uma imagem para um array do numpy.
    '''
    return np.array(Image.open(path))

def save_image(img : np.array, path : str):
    ''' Salva uma imagem expressa em array do numpy em um arquivo.
    '''
    return Image.fromarray(img).save(path)

def show_image(img : np.array):
    ''' Método para visualizar uma imagem a partir do terminal.
	Obs. até então, só funciona no linux.
    '''
    Image.fromarray(img).show()
