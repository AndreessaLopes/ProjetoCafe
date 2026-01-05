import os
import cv2
import numpy as np
from ultralytics import YOLO
import shutil

# --- CONFIGURAÇÕES ---
# Atualizei para a pasta correta que apareceu no seu log (final2)
MODELO_PATH = 'runs/detect/treino_gpu_final2/weights/best.pt' 

# Vamos usar as duas pastas para garantir que teremos bastante grão VERDE
PASTAS_ORIGEM = ['datasets/images/amarelas', 'datasets/images/vermelhas']
PASTA_DESTINO = 'meu_dataset_cnn'

# Limpeza e preparação
if os.path.exists(PASTA_DESTINO): shutil.rmtree(PASTA_DESTINO)
os.makedirs(f"{PASTA_DESTINO}/maduro")
os.makedirs(f"{PASTA_DESTINO}/verde")

print(f"Carregando modelo turbinado: {MODELO_PATH}")
try:
    model = YOLO(MODELO_PATH)
except:
    print(f"ERRO: Não achei o arquivo em {MODELO_PATH}")
    print("Verifique se o nome da pasta 'runs' está correto.")
    exit()

def classificar_cor(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Filtra fundo preto/sombra
    mask = v > 40 
    if np.sum(mask) == 0: return None
    hue = np.mean(h[mask])
    
    # LÓGICA DE COR OTIMIZADA
    # Hue < 35: Laranjas e Amarelos (Maduro)
    # Hue >= 35: Amarelo-Limão e Verdes (Verde)
    return "maduro" if hue < 35 else "verde"

count = 0
for pasta in PASTAS_ORIGEM:
    print(f"--> Processando pasta: {pasta}")
    if not os.path.exists(pasta):
        print(f"Aviso: Pasta {pasta} não encontrada, pulando...")
        continue
        
    arquivos = [f for f in os.listdir(pasta) if f.lower().endswith('.jpg')]
    
    for arq in arquivos:
        img = cv2.imread(os.path.join(pasta, arq))
        if img is None: continue

        # Agora usamos imgsz=640 igual ao treino, para ele ver tudo
        results = model(img, imgsz=640, conf=0.15, verbose=False)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Garante que o recorte está dentro da imagem
                h_img, w_img = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                recorte = img[y1:y2, x1:x2]
                # Ignora recortes minúsculos (sujeira)
                if recorte.size == 0 or recorte.shape[0] < 10 or recorte.shape[1] < 10: 
                    continue
                
                classe = classificar_cor(recorte)
                if classe:
                    # Padroniza tamanho para a CNN
                    recorte = cv2.resize(recorte, (64, 64))
                    cv2.imwrite(f"{PASTA_DESTINO}/{classe}/grao_{count}.jpg", recorte)
                    count += 1
    print(f"   Parcial: {count} grãos recortados...")

print("-" * 30)
print(f"CONCLUÍDO! Total final: {count} grãos.")
print(f"Verifique a pasta '{PASTA_DESTINO}' se tem arquivos em 'verde' e 'maduro'.")