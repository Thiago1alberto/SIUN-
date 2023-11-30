from src.model.model import DDModel
from src.lib.data_producer import DataProducer
import tensorflow as tf
import keras.backend as K
from keras.optimizers import RMSprop, Adam
from skimage import io, transform, feature, color
import numpy as np
import sys
from keras.utils.training_utils import multi_gpu_model
import queue
import threading

class Trainer():
    def __init__(self, config):
        # Inicializa o treinador com as configurações fornecidas
        self.config = config
        self.model = DDModel(config)
        self.batch_size = config.trainer.batch_size
        self.learningSteps = [1e-4, 3e-5, 5e-6, 1e-6]
        self.currentStep = 0
        self.bestLoss = 2
        self.bestEpoch = 0
        self.current_size = 0
        self.iters = [3]
        self.iter_length = len(self.iters)
        self.pyramid_blurs = []
        self.pyramid_sharps = []

    def start(self):
        # Inicia o treinamento com o número máximo de épocas especificado
        self.train(self.config.trainer.maxEpoch)

    def __trainBatch(self):
        # Realiza o treinamento em lote
        batch_blurs2x = []
        batch_blurs1x = []
        batch_sharps1x = []
        n = len(self.pyramid_blurs)
        for i in range(self.max_iter, 0, -1):
            if i == self.max_iter:  # Primeira iteração
                # Gera batch_blurs2x
                for j in range(n):
                    pyramid_blur = self.pyramid_blurs[j]
                    imageBlur2x = pyramid_blur[i]
                    batch_blurs2x.append(imageBlur2x)
                batch_gen = batch_blurs2x
            else:
                # Gera batch_blurs2x
                batch_blurs2x = batch_blurs1x
                batch_blurs1x = []
                batch_sharps1x = []
            # Gera batch_blurs1x
            for j in range(n):
                pyramid_blur = self.pyramid_blurs[j]
                imageBlur1x = pyramid_blur[i-1]
                batch_blurs1x.append(imageBlur1x)
            # Gera batch_sharps1x
            for j in range(n):
                pyramid_sharp = self.pyramid_sharps[j]
                imageSharp1x = pyramid_sharp[i-1]
                batch_sharps1x.append(imageSharp1x)
            # Geração de dados concluída
            
            # Treina o Gerador 2x
            train_X1 = np.concatenate((batch_blurs2x, batch_gen), axis=3)  # 6 canais
            train_X = {'imageSmall': train_X1, 'imageUp': np.array(batch_blurs1x)}
            g_loss = self.generator.train_on_batch(train_X, np.array(batch_sharps1x))
            if i == 1:  # Última iteração
                self.g_loss += g_loss * n
            else:
                batch_gen = self.generator.predict(train_X)
        # Treino concluído, reset
        self.current_size = 0
        self.pyramid_blurs = []
        self.pyramid_sharps = []

    def __doInteration(self, blur, sharp, epoch):
        # Realiza uma iteração
        iter_index = epoch % self.iter_length
        self.max_iter = self.iters[iter_index]
        if self.current_size < self.batch_size:
            self.pyramid_blurs.append(tuple(transform.pyramid_gaussian(blur, downscale=2, max_layer=self.max_iter, multichannel=True)))
            self.pyramid_sharps.append(tuple(transform.pyramid_gaussian(sharp, downscale=2, max_layer=self.max_iter, multichannel=True)))
            self.current_size += 1
        if self.current_size == self.batch_size:  # Treina um lote
            self.__trainBatch()

    def __nextStep(self):
        # Passa para o próximo passo de aprendizado
        self.currentStep += 1
        if self.currentStep < len(self.learningSteps):
            lr = self.learningSteps[self.currentStep]
            K.set_value(self.generator.optimizer.lr, lr)
            self.model.save(self.model.generator, self.config.resource.generator_json_path, self.config.resource.generator_weights_path)
            f_lr = "{:.2e}".format(lr)
            print(f'learning rate: {f_lr}')
            return False
        else:  # Fim prematuro
            return True

    def __learningScheduler(self, epoch):
        # Agenda de aprendizado
        if epoch == 0:
            lr = K.get_value(self.generator.optimizer.lr)
            f_lr = "{:.2e}".format(lr)
            print(f'learning rate: {f_lr}')
            return False
        if self.bestLoss > self.g_loss:
            self.bestLoss = self.g_loss
            self.bestEpoch = epoch
            if self.currentStep == len(self.learningSteps)-1:  # Último passo
                self.model.save(self.model.generator, self.config.resource.generator_json_path, self.config.resource.generator_weights_path)
            return False
        # self.bestLoss <= self.g_loss, modelo não melhorado
        if self.currentStep == len(self.learningSteps)-1:  # Último passo
            patience = 50
        else:
            patience = 30
        if epoch - self.bestEpoch >= patience:
            return self.__nextStep()

    def train(self, maxEpoch):
        # Configura o otimizador e o modelo
        optimizer = Adam(self.learningSteps[self.currentStep])
        if self.config.trainer.gpu_num > 1:
            self.generator = multi_gpu_model(self.model.generator, self.config.trainer.gpu_num)
        else:
            self.generator = self.model.generator
        self.generator.compile(loss='mean_absolute_error', optimizer=optimizer)
        print(f'generator: {self.generator.metrics_names}')
        print(f'training strategy: {self.iters}')
        
        image_queue = queue.Queue(maxsize=self.config.trainer.batch_size*4)
        dataProducer = DataProducer('Producer', image_queue, self.config)
        n = dataProducer.loadDataList(self.config.resource.train_directory_path)
        dataProducer.start()
        for epoch in range(maxEpoch):
            # Ajusta a taxa de aprendizado
            if self.__learningScheduler(epoch):  # Fim prematuro
                print('early end')
                sys.exit()
            
            self.g_loss = 0
            for i in range(n):
                imageBlur, imageSharp = image_queue.get(1)  # Bloqueia
                self.__doInteration(imageBlur, imageSharp, epoch)
            if self.pyramid_blurs:
                # Último lote, pode ser menor que batch_size
                self.__trainBatch()
            self.g_loss = self.g_loss/n
            f_g_loss = "{:.3e}".format(self.g_loss)
            print(f'epoch: {epoch+1}/{maxEpoch}, [G loss: {f_g_loss}]')
        
        # Salva o modelo ao final do treinamento
        self.model.save(self.model.generator, self.config.resource.generator_json_path, self.config.resource.generator_weights_path)
