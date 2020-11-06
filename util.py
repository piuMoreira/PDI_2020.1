from loader import *

def extend_with_zeros(img : np.array, horiz : int, vert : int, horizr=None, vertb=None):
    ''' Extende uma imagem com zeros nas bordas de acordo com os valores de horiz e vert.
    Exemplo: extend_with_zeros(np.array([[1, 2], [3, 4]]), 2, 1)
             >>> array([[0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 2, 0, 0],
                        [0, 0, 3, 4, 0, 0],
                        [0, 0, 0, 0, 0, 0]])
    '''
    opt = lambda x, y: y if x is None else x
    horizr, vertb = opt(horizr, horiz), opt(vertb, vert)
    if len(img.shape) == 3:
        return np.array([np.pad(i, ((vert, vertb), (horiz, horizr))) for i in img.T]).T
    return np.pad(img, ((vert, vertb), (horiz, horizr)))


def extend_with_zeros_mask(img : np.array, horiz : int, vert : int, noncenter=True, rrr={}):
    ''' Extende uma imagem com zeros nas bordas que possibilite a aplicação de uma máscara horiz x vert.
    noncenter extenderá a janela direita e/ou inferior em +1 caso a máscara seja par
    Exemplo: extend_with_zeros_mask(np.array([[1, 2], [3, 4]]), 3, 3)
             >>> array([[0, 0, 0, 0],
                        [0, 1, 2, 0],
                        [0, 3, 4, 0],
                        [0, 0, 0, 0]])
    Exemplo: extend_with_zeros_mask(np.array([[1, 2], [3, 4]]), 2, 2)
             >>> array([[1, 2, 0],
                        [3, 4, 0],
                        [0, 0, 0]])
    '''
    cc = lambda x: x//2 - 1 if noncenter and x % 2 == 0 else x//2
    rrr['xi'], rrr['yi'] = cc(horiz), cc(vert)
    return extend_with_zeros(img, cc(horiz), cc(vert), horiz//2, vert//2)

def apply_mask_func(func, img : np.array, maskh : int, maskv : int, ext_zero=True, pass_coordinates=False):
    ''' Aplica uma máscara de tamanho maskh x maskv sobre img usando uma função fornecida func.
    Se ext_zero = True, então a extensão por zeros é usada.

    func é uma função que pode possuir duas assinaturas:
    Se pass_coordinates = False, func recebe 1 parâmetro: a janela J de filtro de tamanho (usualmente) maskh x maskv.
    Se pass_coordinates = True, então func recebe três parâmetros: a janela J, i e j (indicando as coordenadas do pixel
        em relação a imagem). Por exemplo, seja uma imagem 512x512, 1 canal, uma janela 3x3, no pixel 201x201,
        func receberá três parâmetros:
            1: A janela que compreende 200x200 até 202x202 (inclusos)
            2: 201
            3: 201
    
    func deve retornar um escalar para o ponto pixel se estiver trabalhando em um único canal. Isto é, img.shape = (W, H).
    Se img for tridimensional/multicanal, img.shape = (W, H, C), então func receberá como parâmetro um array de tamanho
        C (usualmente C=3) de janelas de tamanho maskh x maskv, e deve retornar um array de escalares de tamanho C.
    Consulte a função apply_mask_func_each_channel para separar os canais automaticamente.

    Se a extensão por zeros é usada, então o parâmetro da func, a janela de filtro, tem SEMPRE tamanho maskh x maskv.
    Senão, a janela J será cortada nas bordas. No mesmo exemplo da imagem 512x512 e janela 3x3, no pixel 0x0 a janela J
        será de tamanho 2x2, como apresentado no esquemático a seguir, onde x é cortado, 0 corresponde ao pixel 0x0:
        [[x, x, x
          x, 0, 1
          x, 2, 3]]
    Nesse caso, na função func(J), J.shape = (2, 2), e não J.shape = (3, 3) como no caso do pixel 201x201.
    '''
    # def funcassert(x, *args):
    #     assert x.shape == (maskh, maskv)
    #     return func(x, *args)
    
    w, h = img.shape[0], img.shape[1] # salvamos a largura e altura
    imgz = img if not ext_zero else extend_with_zeros_mask(img, maskh, maskv) # extensão por zeros
    cc = lambda x: x//2 - 1 if x % 2 == 0 else x//2 # SEM extensão por zero, corte das janelas
    nhl, nvu, nhr, nvd = cc(maskh), cc(maskv), maskh//2 + 1, maskv//2 + 1 # pré-calculando cortes
    
    def sel_extz(i, j, *args): # Para extensão por zero
        return func(imgz[i:i+maskh, j:j+maskv], *args) # Aplica func em i,j + MASK
    def sel_nextz(i, j, *args): # Para SEM extensão por zero
        return func(imgz[max(0, i-nhl):min(w, i+nhr), max(0, j-nvu):min(h, j+nvd)], *args) # Aplica func com limites
    sel_f = sel_extz if ext_zero else sel_nextz # Escolhe se extensão por zero

    if pass_coordinates: # Se fornece i,j para a função
        return np.array([[sel_f(i, j, i, j) for j in range(h)] for i in range(w)])
    else: # Senão...
        return np.array([[sel_f(i, j) for j in range(h)] for i in range(w)])

def apply_mask_func_each_channel(func, img : np.array, maskh : int, maskv : int, ext_zero=True):
    ''' Aplica uma máscara de tamanho maskh x maskv sobre img usando uma função fornecida func.
    Se ext_zero = True, então a extensão por zeros é usada.

    func é uma função que pode possuir duas assinaturas:
    Se pass_coordinates = False, func recebe 1 parâmetro: a janela J de filtro de tamanho (usualmente) maskh x maskv.
    Se pass_coordinates = True, então func recebe três parâmetros: a janela J, i e j (indicando as coordenadas do pixel
        em relação a imagem). Por exemplo, seja uma imagem 512x512, 1 canal, uma janela 3x3, no pixel 201x201,
        func receberá três parâmetros:
            1: A janela que compreende 200x200 até 202x202 (inclusos)
            2: 201
            3: 201
    
    Essa função divide cada canal separadamente/independentemente, e então aplica a função apply_mask_func.

    Se a extensão por zeros é usada, então o parâmetro da func, a janela de filtro, tem SEMPRE tamanho maskh x maskv.
    Senão, a janela J será cortada nas bordas. No mesmo exemplo da imagem 512x512 e janela 3x3, no pixel 0x0 a janela J
        será de tamanho 2x2, como apresentado no esquemático a seguir, onde x é cortado, 0 corresponde ao pixel 0x0:
        [[x, x, x
          x, 0, 1
          x, 2, 3]]
    Nesse caso, na função func(J), J.shape = (2, 2), e não J.shape = (3, 3) como no caso do pixel 201x201.
    '''
    return np.array([apply_mask_func(func, i, maskh, maskv, ext_zero) for i in img.T]).T
