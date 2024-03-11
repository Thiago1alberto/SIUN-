SIUN - Scale Iterative Upscaling Network
O SIUN é uma rede neural projetada para aumentar a resolução de imagens, especialmente útil em tarefas de super-resolução de imagem. Ele opera de forma iterativa, aplicando etapas de aumento de escala para produzir imagens de alta resolução a partir de imagens de baixa resolução.

# Instruções de Uso do Git Clone

1. Clonar o Repositório:
   git clone 

2. Acessar o Diretório do Código:
   cd code

Uso Básico
- Sempre é possível adicionar '--gpu=<gpu_id>' para especificar o ID da GPU, o ID padrão é 0.

Para desfocar uma imagem:
   python deblur.py --apply --file-path='<caminho_do_arquivo/teste.png>'

Para desfocar todas as imagens em uma pasta:
   python deblur.py --apply --dir-path='<caminho_da_pasta/testeDir>'
   Adicione '--result-dir=<caminho_de_saída>' para especificar o caminho de saída. Se não for especificado, o caminho padrão é './output'.

Para testar o modelo:
   python deblur.py --test
   Note que este comando só pode ser usado para testar o conjunto de dados GOPRO. E ele carregará todas as imagens na memória primeiro. Recomendamos usar '--apply' como uma alternativa (Item 2).
   Por favor, defina o valor de 'test_directory_path' para especificar o caminho do conjunto de dados GOPRO no arquivo 'config.py'.

Para treinar um novo modelo:
   python deblur.py --train
   Por favor, remova o arquivo de modelo em 'model' primeiro e defina o valor de 'train_directory_path' para especificar o caminho do conjunto de dados GOPRO no arquivo 'config.py'.
   Quando terminar, execute:
   python deblur.py --verify
