import os
from src.model.model import DDModel
from src.lib.data_helper import DataHelper
from src.lib.MLVSharpnessMeasure import MLVMeasurement
from skimage import io,transform #reize image
import numpy as np
import pickle
import math

class Tester():
    def __init__(self, config):
        # Inicializa o testador com as configurações fornecidas
        self.config = config
        self.model = DDModel(config)
        self.batch_size = 8
        self.current_size = 0
        self.pyramid_blurs = []
        self.batch_sharps = []
        # Métricas
        self.all_psnrs = {}

    def start(self):
        # Inicia o teste com a estratégia de iteração especificada
        if self.config.tester.iter == 0:
            self.iters = [1, 2, 3, 4]
        else:
            self.iters = [self.config.application.iter]
        self.max_iter = max(self.iters)
        for iter in self.iters:
            self.all_psnrs[iter] = []
        print(f'Test strategy: {self.iters}')
        self.test()

    def __compute_psnr(self, x, label, max_diff):
        # Calcula a métrica PSNR (Peak Signal-to-Noise Ratio)
        mse = np.mean((x - label) ** 2)
        return 10 * math.log10(max_diff**2 / mse)

    def __doBatchTest(self):
        # Realiza o teste em lote
        n = len(self.pyramid_blurs)
        for iter in self.iters:
            batch_blurs2x = []
            batch_blurs1x = []
            for i in range(iter, 0, -1):
                if i == iter:  # Primeira iteração
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
                # Gera batch_blurs1x
                for j in range(n):
                    pyramid_blur = self.pyramid_blurs[j]
                    imageBlur1x = pyramid_blur[i-1]
                    batch_blurs1x.append(imageBlur1x)
                # Preparação de dados concluída
                
                # Prediz 2x
                data_X1 = np.concatenate((batch_blurs2x, batch_gen), axis=3)  # 6 canais
                data_X = {'imageSmall': data_X1, 'imageUp': np.array(batch_blurs1x)}
                batch_gen = self.model.generator.predict(data_X)
            
            # Calcula métricas
            for i in range(n):
                pImage = batch_gen[i]
                pImage = pImage[24:744]
                psnr = self.__compute_psnr(pImage, self.batch_sharps[i], 1)
                self.all_psnrs[iter].append(psnr)
        
        # Reinicia variáveis
        self.current_size = 0
        self.pyramid_blurs = []
        self.batch_sharps = []

    def __doInteration(self, blur, sharp):
        # Realiza uma iteração
        if self.current_size < self.batch_size:
            blur = np.pad(blur, ((24, 24), (0, 0), (0, 0)), 'reflect')  # Será dividido por 256
            self.pyramid_blurs.append(tuple(transform.pyramid_gaussian(blur, downscale=2, max_layer=self.max_iter, multichannel=True)))
            self.batch_sharps.append(sharp)
            self.current_size += 1
        if self.current_size == self.batch_size:  # Treina um lote
            self.__doBatchTest()

    def test(self):
        # Inicia o teste
        dataHelper = DataHelper()
        dataHelper.load_data(self.config.resource.test_directory_path, 0)
        
        blurSharpParis = dataHelper.getLoadedPairs()
        for imageBlur, imageSharp in blurSharpParis:
            self.__doInteration(imageBlur, imageSharp)
        
        if self.pyramid_blurs:
            self.__doBatchTest()
        
        # Analisa resultados
        psnrs = []
        for iter in self.iters:
            psnrs.append(self.all_psnrs[iter])
        psnrs = np.array(psnrs)
        psnrs_by_iter = np.mean(psnrs, axis=1)
        for i in range(len(psnrs_by_iter)):
            print(f'PSNR: {psnrs_by_iter[i]} @ {self.iters[i]}')
        best_psnrs = np.amax(psnrs, axis=0)
        path = os.path.join(self.config.resource.output_dir, "psnrs.pkl")
        with open(path, 'wb') as pfile:
            pickle.dump(best_psnrs, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        best_iters_index = np.argmax(psnrs, axis=0)
        iters = np.array(self.iters)
        best_iters = iters[best_iters_index]
        path = os.path.join(self.config.resource.output_dir, "iters.pkl")
        with open(path, 'wb') as pfile:
            pickle.dump(best_iters, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        calculate_data_n = len(best_psnrs)
        print(f'{calculate_data_n}/{len(blurSharpParis)} done! Average PSNRs(Best): {np.mean(best_psnrs)}')
