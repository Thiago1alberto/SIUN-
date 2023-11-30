import os
import getpass

def _project_dir():
    # Função auxiliar para obter o diretório raiz do projeto
    d = os.path.dirname
    return d(d(os.path.abspath(__file__)))

class Config:
    def __init__(self):
        # Instâncias das classes de configuração
        self.resource = ResourceConfig()
        self.trainer = TrainConfig()
        self.tester = TestConfig()
        self.application = Application()

class ResourceConfig:
    def __init__(self):
        # Configurações relacionadas a recursos e diretórios
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", os.path.join(_project_dir(), "data"))
        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.project_dir, "model"))
        self.debug_dir = os.environ.get("DEBUG_DIR", os.path.join(self.project_dir, "debug"))
        self.output_dir = os.environ.get("OUTPUT_DIR", os.path.join(self.project_dir, "output"))
        
        # Caminhos específicos do projeto
        self.generator_json_path = os.path.join(self.model_dir, "generator.json")
        self.generator_weights_path = os.path.join(self.model_dir, "generator.h5")
        self.train_directory_path = "/mnt/SD_1/myye/Deblur/GoPro/train"
        self.test_directory_path = "/mnt/SD_1/myye/Deblur/GoPro/test"

    def create_directories(self):
        # Criação de diretórios se não existirem
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.debug_dir, self.output_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)

class TrainConfig:
    def __init__(self):
        # Configurações específicas para o treinamento do modelo
        self.generatorImageSize = 256
        self.generatorImageChannels = 3
        self.batch_size = 8
        self.maxEpoch = 2000
        self.gpu_num = 1

class TestConfig:
    def __init__(self):
        # Configurações específicas para os testes
        self.iter = 0

class Application:
    def __init__(self):
        # Configurações específicas para a aplicação do modelo
        self.iter = 4  # Tentará todas as iterações (1,2,3,4) se for definido como 0
        self.deblurring_file_path = None
        self.deblurring_dir_path = None
        self.deblurring_result_dir = None
