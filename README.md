# PDI_2020.1
Repositório para a disciplina de Introdução ao Processamento Digital de Imagens da UFPB 2020.1

## Dependências
- [Python3](https://www.python.org/)
- [Numpy](https://numpy.org/)
- [Pillow](https://pypi.org/project/Pillow/)
- [pymlfunc](https://pypi.org/project/pymlfunc/)

Opcionalmente, pode-se utilizar o gerenciador de pacotes *pip* para instalar automaticamente as dependências:

`pip install -r requirements.txt`

ou ainda,

`python3 -m pip install -r requirements.txt`

## Uso

`python3 examples.py IMAGEM operacao1 [operacao2] [operacao3...]`

Exemplo:
`python3 examples.py images/baboon.png avg_grayscale flip_v`
