from loader import *
from sys import argv

def avg_grayscale(img : np.array):
    avg = np.average(img, 2)
    for i in range(img.shape[2]):
        img[:,:,i] = avg
    return img

def flip_v(img : np.array):
    return np.flipud(img)

def flip_h(img : np.array):
    return np.fliplr(img)

def erase_channel(img : np.array, channel : int):
    img[:,:,channel] = np.zeros(img[:,:,channel].shape)
    return img

erase_red   = lambda img: erase_channel(img, 0)
erase_green = lambda img: erase_channel(img, 1)
erase_blue  = lambda img: erase_channel(img, 2)



def main_interpret(args):
    if len(args) < 2:
        print('Uso: python3 examples.py IMAGEM operacao1 [operacao2] [operacao3...]')
        return

    omaps = {'avg_grayscale': avg_grayscale, 'flip_v': flip_v, 'flip_h': flip_h,
             'erase_red': erase_red, 'erase_green': erase_green, 'erase_blue': erase_blue}

    img = open_image(args[0])
    for o in args[1:]:
        img = omaps[o](img)
    
    show_image(img)

if __name__ == "__main__":
    main_interpret(argv[1:])
