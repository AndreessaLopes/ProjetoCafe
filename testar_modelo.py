import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os

# --- CONFIGURAÇÕES ---
CAMINHO_YOLO = 'yolo_detector.pt'
CAMINHO_CNN = 'modelo_final_cafe.keras'
PASTA_IMAGENS = 'dataset/images/'

# Pega a primeira imagem da pasta
arquivos = [f for f in os.listdir(PASTA_IMAGENS) if f.endswith('.jpg')]
if not arquivos: 
    print("Nenhuma imagem encontrada!")
    exit()
    
# Se quiser testar uma imagem específica, descomente a linha abaixo e ponha o nome:
NOME_IMAGEM = 'cafe3.jpg' 

img_path = os.path.join(PASTA_IMAGENS, NOME_IMAGEM)
print(f"--- PROCESSANDO: {NOME_IMAGEM} ---")

# 1. Carrega Modelos
yolo = YOLO(CAMINHO_YOLO)
cnn = tf.keras.models.load_model(CAMINHO_CNN)

# IMPORTANTE: A ordem alfabética das pastas foi 'maduro', 'verde'.
# Então 0 = maduro, 1 = verde.
CLASSES = ['maduro', 'verde']

# 2. Carrega Imagem
img = cv2.imread(img_path)
img_final = img.copy()

# 3. DETECÇÃO (Ajuste de Sensibilidade)
# conf=0.01: Qualquer coisa que pareça 1% com um grão será detectada.
# iou=0.5: Evita quadrados duplicados no mesmo grão.
results = yolo(img, imgsz=640, conf=0.01, iou=0.5, verbose=False)

count_maduro = 0
count_verde = 0

print(f"Grãos detectados pelo YOLO: {len(results[0].boxes)}")

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # Recorte com margem de segurança
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        recorte = img[y1:y2, x1:x2]
        if recorte.size == 0 or recorte.shape[0] < 5 or recorte.shape[1] < 5: 
            continue
        
        # --- A CORREÇÃO DE COR CRUCIAL ---
        # O OpenCV abre a imagem em Azul-Verde-Vermelho (BGR).
        # A sua CNN aprendeu em Vermelho-Verde-Azul (RGB).
        # Se não converter, o Amarelo vira Azul Ciano e a IA erra.
        recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
        
        # Redimensiona para 64x64 (tamanho que a CNN aprendeu)
        recorte_cnn = cv2.resize(recorte_rgb, (64, 64))
        
        # Normaliza (0 a 1) se sua rede esperar float, mas o layer Rescaling já faz isso.
        # Adiciona dimensão do lote: (1, 64, 64, 3)
        img_arr = tf.expand_dims(recorte_cnn, 0)
        
        # Predição
        preds = cnn.predict(img_arr, verbose=0)
        score = tf.nn.softmax(preds[0])
        idx = np.argmax(score) # 0 ou 1
        classe = CLASSES[idx]
        confianca = 100 * np.max(score)
        
        # Desenha
        if classe == 'maduro':
            cor = (0, 255, 255) # Amarelo no OpenCV
            lbl = "M" # Abreviação pra não poluir
            count_maduro += 1
        else:
            cor = (0, 255, 0)   # Verde no OpenCV
            lbl = "V"
            count_verde += 1
            
        cv2.rectangle(img_final, (x1, y1), (x2, y2), cor, 2)
        # Texto menor para não tampar o grão
        cv2.putText(img_final, f"{lbl} {int(confianca)}%", (x1, y1-3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)

# Salva
cv2.imwrite("resultado_final_corrigido.jpg", img_final)

print("-" * 30)
print(f"ANÁLISE FINAL:")
print(f"Maduros (Amarelo): {count_maduro}")
print(f"Verdes  (Verde):   {count_verde}")
print(f"Imagem salva como: resultado_final_corrigido.jpg")
print("-" * 30)