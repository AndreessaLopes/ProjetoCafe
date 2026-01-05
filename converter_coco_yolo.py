import json
import os
from pathlib import Path

def convert_coco_to_yolo(json_file, output_dir):
    # Cria a pasta de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Carrega o JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Cria um dicionário para buscar informações da imagem pelo ID rapidamente
    # Formato: {image_id: {'file_name': 'nome.jpg', 'width': 640, 'height': 640}}
    images_info = {img['id']: img for img in data['images']}
    
    # Itera sobre todas as anotações
    for ann in data['annotations']:
        image_id = ann['image_id']
        img_info = images_info.get(image_id)
        
        if not img_info:
            continue
            
        # Dimensões da imagem original
        img_w = img_info['width']
        img_h = img_info['height']
        
        # Bbox no COCO é [top-left-x, top-left-y, width, height]
        x, y, w, h = ann['bbox']
        
        # YOLO precisa de [x_center, y_center, width, height] NORMALIZADO (0 a 1)
        x_center = (x + (w / 2)) / img_w
        y_center = (y + (h / 2)) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        # Formata a linha para o arquivo TXT
        # Classe 0 (assumindo apenas uma classe 'grao')
        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
        
        # Nome do arquivo de saída (mesmo nome da imagem, mas .txt)
        file_name = img_info['file_name']
        txt_name = Path(file_name).stem + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        
        # Escreve no arquivo (append, pois pode haver vários grãos na mesma imagem)
        with open(txt_path, 'a') as f_out:
            f_out.write(yolo_line)

    print(f"Conversão concluída! Arquivos salvos em: {output_dir}")

# --- CONFIGURAÇÃO ---
# Mude os caminhos abaixo conforme sua estrutura
arquivo_json = 'datasets/vermelhas.json'  # O arquivo que você me enviou
pasta_saida = 'datasets/labels/vermelhas' # Onde os .txt serão salvos

convert_coco_to_yolo(arquivo_json, pasta_saida)