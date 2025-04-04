## Sistema de Identificação de Placas

# Descrição do Projeto

Este projeto tem como objetivo a identificação de placas de veículos a partir de imagens e vídeos utilizando o modelo YOLOv8 para detecção dos veículos e o EasyOCR para reconhecimento de texto nas placas. Ele permite a extração e validação de números de placas, que são comparados com uma base de dados para identificar proprietários e características do veículo.

# Funcionalidades

Detecção de veículos e placas em imagens e vídeos.

Reconhecimento de caracteres nas placas.

Validação de padrões de placas convencionais e do Mercosul.

Conversão de caracteres ambíguos para melhorar a precisão.

Armazenamento e verificação de placas identificadas em um banco de dados.

Exportação dos resultados em formato CSV.

## Tecnologias Utilizadas

Python - Linguagem principal do projeto.

YOLOv8 - Modelo de detecção para identificar veículos e placas.

EasyOCR - Biblioteca para reconhecimento de caracteres.

OpenCV - Manipulação e processamento de imagens.

Pandas - Tratamento e exportação de dados.

Django - Backend para armazenamento e gerenciamento de dados.
