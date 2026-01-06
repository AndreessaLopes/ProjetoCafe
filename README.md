# 🧠 ProjetoCafe

## ☕ Visão Geral

**ProjetoCafe** é um projeto de *Machine Learning* voltado para detecção e classificação de cafés em imagens, utilizando **YOLO** para detecção de objetos e **CNNs** para classificação.  
O repositório reúne scripts para preparação de dados, treinamento, teste de modelos e conversão de anotações entre formatos populares em visão computacional.

---

## 📂 Estrutura do Repositório

ProjetoCafe/  
├── converter_coco_yolo.py      # Conversão de anotações COCO ↔ YOLO  
├── gerar_dataset_cnn.py        # Geração e organização do dataset para CNN  
├── minha_arquitetura.py        # Definição da arquitetura da CNN  
├── modelo_final_cafe.keras     # Modelo final treinado (classificação)  
├── testar_modelo.py            # Testes e inferência do modelo treinado  
├── teste_gpu.py                # Verificação de disponibilidade de GPU  
├── treinar_yolo.py             # Treinamento do modelo YOLO  
├── yolov8n.pt                  # Pesos YOLO pré-treinados  
├── yolo11n.pt                  # Pesos YOLO adicionais  
└── README.md                   # Documentação do projeto  

---

## 🚀 Tecnologias Utilizadas

- Python 3.x  
- TensorFlow / Keras – Classificação com CNN  
- PyTorch / Ultralytics YOLO – Detecção de objetos  
- OpenCV  
- NumPy  
- GPU (opcional, via CUDA)

---

## 📌 Pré-requisitos

- Python 3.8 ou superior  
- Pip atualizado  
- (Opcional) GPU com CUDA configurada  

Instalação básica das dependências:

pip install tensorflow torch torchvision opencv-python numpy matplotlib ultralytics

---

## 🧠 Como Executar o Projeto

### 1️⃣ Preparação do Dataset

Caso as anotações estejam no formato COCO e seja necessário convertê-las para YOLO:

python converter_coco_yolo.py

Para organizar o dataset para treinamento da CNN:

python gerar_dataset_cnn.py

---

### 2️⃣ Treinamento do Detector (YOLO)

python treinar_yolo.py

---

3️⃣ Geração do Dataset (Mineração)

Usa o YOLO treinado acima para recortar grãos das imagens originais e separá-los em pastas (Verde/Maduro) automaticamente.

python gerar_dataset_cnn.py

---

4️⃣ Treinamento da CNN (Classificação)

Define a arquitetura, treina a rede neural nos recortes gerados e salva o arquivo .keras.

python minha_arquitetura.py

---

5️⃣ Testes e Inferência

Testar o modelo de classificação:

python testar_modelo.py

---

## 📊 Resultados

Após o treinamento, o projeto permite:

-Detectar múltiplos grãos de café em imagens de campo (YOLO).

-Classificar cada grão detectado como "Verde" ou "Maduro" (CNN).

-Visualizar a confiança da predição em tempo de execução.

---

## 🤝 Contribuições

Contribuições são bem-vindas.  
Sinta-se à vontade para abrir *issues* ou enviar *pull requests* com melhorias, correções ou novas funcionalidades.

---
