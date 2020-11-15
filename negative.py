import numpy as np

def complete_negative(img : np.array):
# Aplica o negativo em todos os canais da imagem
    for channel in range(3):
        img[:,:,channel] = 255 - img[:,:,channel]
    return img

def channel_negative(img : np.array, channel : int):
# Aplica o negativo em um canal específico da imagem, esse  canal é especificado como um parâmetro de entrada
    img[:,:,channel] = 255 - img[:,:,channel]
    return img