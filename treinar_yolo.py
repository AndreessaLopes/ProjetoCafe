import os
from ultralytics import YOLO

# Configuração do Dataset
caminho_abs = os.path.abspath('datasets').replace(os.sep, '/')
yaml_content = f"""
path: {caminho_abs}
train: 
  - images/amarelas
  - images/vermelhas
val: 
  - images/amarelas
  - images/vermelhas
nc: 1
names: ['grao_cafe']
"""
with open('data.yaml', 'w') as f: f.write(yaml_content)

def main():
    model = YOLO('yolov8n.pt')
    
    print("--- INICIANDO TREINAMENTO TURBO (GPU: RTX 3060 Ti) ---")
    model.train(
        data='data.yaml', 
        epochs=50,      # Agora temos tempo! 50 épocas para ficar inteligente
        imgsz=640,      # Resolução alta para ver grãos pequenos
        batch=16,       # A placa tem memória, vamos encher
        device=0,       # Força o uso da NVIDIA
        workers=4,
        name='treino_gpu_final'
    )

if __name__ == '__main__':
    main()