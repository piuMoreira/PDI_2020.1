import numpy as np

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


def round_image_colors(img : np.array, doAssert=True):
    ''' Arredonda valores quebrados e converte o tipo de array para imagem.
    Valores de cor excedentes NÃO são tratados.
    '''
    if doAssert:
        assert img.min() >= -0.0 or img.min() >= 0.0
        assert img.max() <= 255.001
    return img.round().astype(np.uint8)

def fix_scale_image_colors(img : np.array, auto_round=True, forceUpperbound=False, forceLowerbond=False):
    ''' Escala todas as cores em caso de valores excedentes na imagem.
    Se houver valores negativos em img ou forceLowerbond=True, então subtrai-se o módulo do menor valor de img tal que
        o novo menor valor da imagem resultado é zero (todos os outros valores de img também serão atingidos pela soma)
    Se houver valores acima de 255 ou forceUpperbound=True, então img é dividido tal que o maior valor passará a ser 255.
    Se auto_round=True, então a função round_image_colors é chamada ao fim do processo.
    '''
    if img.min() < -0.0 or img.min() < 0.0 or forceLowerbond:
        img = img-img.min()
    if img.max() > 255 or forceUpperbound:
        img = img/img.max()
    return round_image_colors(img) if auto_round else img

def fix_truncate_image_colors(img : np.array, auto_round=True):
    ''' Trunca valores excedentes de cor da imagem.
    '''
    img = img.copy()
    img[img > 255] = 255
    img[img < 0] = 0
    return round_image_colors(img) if auto_round else img


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
    imgz = img if not ext_zero else extend_with_zeros_mask(img, maskv, maskh) # extensão por zeros

    # se não há extensão por zero, então é feito alguns cálculos:
    cc = lambda x: x//2 - 1 if x % 2 == 0 else x//2 # para máscaras de tamanho irregular/par
    nhl, nvu, nhr, nvd = cc(maskh), cc(maskv), maskh//2 + 1, maskv//2 + 1 # pré-calculando cortes
    
    rc = np.zeros(img.shape) # cria uma imagem com as mesmas dimensões de img
    for i in range(w): # percorre largura e altura (essa função não percorre bandas individuais)
        for j in range(h):
            args = [i, j] if pass_coordinates else [] # opcional, passa i e j para func
            if ext_zero: # Se há extensão por zero: não precisa cortar a máscara, os zeros garantem
                assert i+maskh <= imgz.shape[0]    # que não há acesso de indice fora dos limites
                assert j+maskv <= imgz.shape[1]
                rc[i,j] = func(imgz[i:i+maskh, j:j+maskv], *args) # Aplica func em i,j c/ ext. zero
            else: # Se NÃO há extensão por zero: corta a máscara nas bordas
                rc[i,j] = func(imgz[max(0, i-nhl):min(w, i+nhr), max(0, j-nvu):min(h, j+nvd)], *args)

    return rc


def apply_mask_func_each_channel(func, img : np.array, maskh : int, maskv : int, ext_zero=True, pass_coordinates=False):
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
    return np.array([apply_mask_func(func, i, maskh, maskv, ext_zero, pass_coordinates) for i in img.T]).T

def apply_mask(img : np.array, mask : np.array, auto_round=False) -> np.array:
    ''' Aplica uma máscara mask sobre a imagem img. Se auto_round=True, então round_image_colors é invocado com a saída.
    Utiliza o método apply_mask_func_each_channel, porém simplificando a aplicação de máscaras. Consulte examples.py
    '''
    r = apply_mask_func_each_channel(lambda x, *args: (mask*x).sum(), img, mask.shape[0], mask.shape[1])
    return round_image_colors(r) if auto_round else r
