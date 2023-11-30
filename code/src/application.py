import os
from src.model.model import DDModel
from src.lib.data_helper import DataHelper
from skimage import io,transform,feature,color,img_as_float
import numpy as np
import math
import time

class Application():
    
    # O método de inicialização
    def __init__(self,config):
        self.config = config
        self.model = DDModel(config)
        if(config.application.deblurring_result_dir is None):
            config.application.deblurring_result_dir = config.resource.output_dir
        if not os.path.exists(config.application.deblurring_result_dir):
            os.makedirs(config.application.deblurring_result_dir)
        self.__fileBlurList=[]

    def start(self):
        self.application()

    def __tuneSize(self,shape):
        pad = []
        for i in range(2):
            size = shape[i]
            if(size % 256 == 0):
                pad.append(0)
            else:
                n = size // 256 + 1
                pad.append((n*256 - size) // 2)
        return pad
        
    # Método interno para carregar e ajustar uma imagem para ser processada pelo modelo.
    def __getImage(self, fileFullPath):  # Método para carregar e ajustar uma imagem para processamento
    # Carrega a imagem do caminho fornecido e converte para float no intervalo [0, 1]
    imageBlur = img_as_float(io.imread(fileFullPath))
    
    # Garante que o número de linhas (row) e colunas (col) da imagem é par
    row = imageBlur.shape[0]
    col = imageBlur.shape[1]
    row = row - 1 if row % 2 == 1 else row
    col = col - 1 if col % 2 == 1 else col
    
    # Corta a imagem para ter número par de linhas e colunas
    imageBlur = imageBlur[0:row, 0:col]
    
    # Cria uma cópia da imagem original para a imagem de origem
    imageOrigin = imageBlur
    
    # Ajusta o tamanho da imagem para garantir que as dimensões sejam pares
    pad = self.__tuneSize(imageBlur.shape)
    
    # Adiciona reflexões nos limites da imagem para preencher as regiões adicionadas
    imageBlur = np.pad(imageBlur, ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), 'reflect')
    
    # Retorna a imagem ajustada e a imagem de origem
    return imageBlur, imageOrigin
        
    # Método interno para obter uma lista de caminhos de arquivos de imagem a serem processados.
    def __getData(self,root):
        for parent,dirnames,filenames in os.walk(root):
            for filename in filenames:
                self.__fileBlurList.append(os.path.join(parent,filename))
        self.data_length = len(self.__fileBlurList)
        print(f'total data:{self.data_length}!')
        
    # Método interno para aplicar o modelo SIUN à imagem borrosa e obter a imagem desbloqueada.
   def __deblur(self, imageBlur, imageOrigin):
    # Cria uma pirâmide gaussiana a partir da imagem borrada
    # Uma pirâmide gaussiana é uma técnica utilizada em processamento de imagem para representar uma imagem em várias escalas de resolução
    # Cada camada da pirâmide representa uma versão da imagem em uma resolução diferente, sendo a camada superior a mais suavizada e a de menor resolução
       
    pyramid = tuple(transform.pyramid_gaussian(imageBlur, downscale=2, max_layer=self.max_iter, multichannel=True))
    deblurs = []  # Lista para armazenar os resultados de desblurring em diferentes escalas

    # Itera sobre o número de iterações especificadas (self.iters)
    for iter in self.iters:
        batch_blur2x = []
        batch_blur1x = []
        runtime = 0

        # Itera sobre as camadas da pirâmide
        for i in range(iter, 0, -1):
            if i == iter:
                imageBlur2x = pyramid[i]
                batch_blur2x.append(imageBlur2x)
                batch_gen = batch_blur2x
            else:
                batch_blur2x = batch_blur1x
                batch_blur1x = []

            imageBlur1x = pyramid[i - 1]
            batch_blur1x.append(imageBlur1x)

            # Constrói os dados de entrada para o modelo SIUN
            data_X1 = np.concatenate((batch_blur2x, batch_gen), axis=3)  # 6 canais
            data_X = {'imageSmall': data_X1, 'imageUp': np.array(batch_blur1x)}

            # Mede o tempo de execução da predição do modelo
            start = time.time()
            batch_gen = self.model.generator.predict(data_X)
            print(f'Runtime @scale {i}: {time.time() - start:4.3f}')
            runtime += time.time() - start

        print(f'Runtime total @iter {iter}: {runtime:4.3f}')

        # Ajusta a escala da saída do modelo e converte para 'uint8'
        deblur = self.__clipOutput(batch_gen[0], imageOrigin.shape)
        deblur = (deblur * 255).astype('uint8')  # Ajusta a escala e converte para 'uint8'
        deblurs.append(deblur)

    return deblurs
    # Método principal que coordena a aplicação do modelo para desblurring. Lê as opções da configuração, carrega imagens e chama o método
    def application(self):
    # Verifica se o número de iterações é fornecido; se não, usa [1, 2, 3, 4] como padrão
    if self.config.application.iter == 0:
        self.iters = [1, 2, 3, 4]
    else:
        self.iters = [self.config.application.iter]
    
    # Determina o número máximo de iterações
    self.max_iter = max(self.iters)
    
    # Obtém os caminhos de entrada fornecidos na configuração
    deblurring_file_path = self.config.application.deblurring_file_path
    deblurring_dir_path = self.config.application.deblurring_dir_path
    
    # Se um caminho de arquivo individual é fornecido e existe, processa essa imagem
    if deblurring_file_path and os.path.exists(deblurring_file_path):
        imageBlur, imageOrigin = self.__getImage(deblurring_file_path)
        deblurs = self.__deblur(imageBlur, imageOrigin)
        infos = deblurring_file_path.rsplit('/', 1)
        iter_times = len(deblurs)
        
        # Salva cada resultado de desblurring em um arquivo
        for i in range(iter_times):
            deblur = deblurs[i]
            deblur = (deblur * 255).astype('uint8')
            iter = self.iters[i]
            io.imsave(os.path.join(self.config.application.deblurring_result_dir, f'deblur{iter}_{infos[1]}'), deblur)
        print(f'Arquivo salvo')
    
    # Se um diretório de arquivos é fornecido e existe, processa todas as imagens no diretório
    elif deblurring_dir_path and os.path.exists(deblurring_dir_path):
        self.__getData(deblurring_dir_path)
        index = 0
        
        # Itera sobre todos os arquivos no diretório
        for fileFullPath in self.__fileBlurList:
            imageBlur, imageOrigin = self.__getImage(fileFullPath)
            deblurs = self.__deblur(imageBlur, imageOrigin)
            infos = os.path.basename(fileFullPath)
            iter_times = len(deblurs)
            
            # Salva cada resultado de desblurring em um arquivo
            for j in range(iter_times):
                deblur = deblurs[j]
                deblur = (deblur * 255).astype('uint8')
                iter = self.iters[j]
                io.imsave(os.path.join(self.config.application.deblurring_result_dir, f'deblur{iter}_{infos}'), deblur)
            
            index += 1
            print(f'{index}/{self.data_length} concluído!')
        print(f'Todos os arquivos salvos')
    
    else:
        print(f"Nenhum arquivo de desblurring encontrado")
    
    # Método interno para ajustar a saída da imagem do modelo para coincidir com o tamanho da imagem de entrada.  
    def __clipOutput(self,image,outSize):
        inSize = image.shape
        start = []
        for i in range(2):
            start.append((inSize[i] - outSize[i]) // 2)
        return image[start[0]:start[0]+outSize[0],start[1]:start[1]+outSize[1]]
