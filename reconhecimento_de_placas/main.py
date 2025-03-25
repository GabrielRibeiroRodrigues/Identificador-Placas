from ultralytics import YOLO
import cv2
import numpy as np 
from sort.sort import Sort
from util import ler_carro, ler_placas, verificar_camera, salvar_no_postgres, salvar_registro_frequencia, verificar_placa_registrada
import psycopg2
from datetime import datetime
import os

results = {}
mot_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)


conexao = psycopg2.connect(
    dbname="pci_transito",
    user="postgres",
    password="123456",
    host="localhost",
    port="5432"
)

cursor = conexao.cursor()

verde = "\033[92m"
reset = "\033[0m"
tamanho = 50
barra = verde + "━" * tamanho + reset

vermelho = "\033[91m"
barra_verm = vermelho + "━" * tamanho + reset


detector_carro = YOLO('yolov8n.pt')
detector_placa = YOLO("C:\\Users\\Pichau\\Desktop\\best (4).pt")
cap = cv2.VideoCapture("C:\\Users\\Pichau\\Desktop\\Projeto\\ffff.mp4")
# cap = cv2.VideoCapture("rtsp://admin:123456789abc@192.168.0.2:554/cam/realmonitor?channel=1&subtype=0")
porta = 3
veiculos = [2, 3, 5, 7]  
confianca_detectar_carro = 0.0  
confianca_gravar_texto = 0.0
frame_nmr = -1
ret = True
intervalo_frames = 1
frame_anterior = -8


while ret:
    data_e_hora_atuais = datetime.now()
    data_e_hora_em_texto = data_e_hora_atuais.strftime('%Y-%m-%d %H:%M:%S')

    for i in range(intervalo_frames):
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Não foi possível ler o frame {frame_nmr}.")
            break
    if frame is None:
        continue

    # Detecção de veículos usando o modelo de veículos
    detections_veiculos = detector_carro(frame)[0]
    veiculos_detectados = []
    for detection in detections_veiculos.boxes.data.tolist():
        x1, y1, x2, y2, confianca_atual, class_id = detection
        if confianca_atual >= confianca_detectar_carro and int(class_id) in veiculos:
            veiculos_detectados.append([x1, y1, x2, y2, confianca_atual])

    # Rastrear veículos
    if veiculos_detectados:
        track_ids = mot_tracker.update(np.asarray(veiculos_detectados))
    else:
        track_ids = []

    # Detecção de placas usando o modelo de placas
    detections_placas = detector_placa(frame)[0]
    placas_detectadas = []
    for detection in detections_placas.boxes.data.tolist():
        x1, y1, x2, y2, confianca_atual, class_id = detection
        if confianca_atual >= confianca_detectar_carro:
            placas_detectadas.append([x1, y1, x2, y2, confianca_atual])

    # Atribuir as placas aos veículos detectados
    for placa in placas_detectadas:
        x1, y1, x2, y2, confianca_atual = placa
        # Verificar qual veículo corresponde à placa
        xcar1, ycar1, xcar2, ycar2, car_id = ler_carro(placa, track_ids)

        if car_id != -1:
            # Verificação de limites
            if (0 <= x1 < frame.shape[1] and 0 <= x2 < frame.shape[1] and 0 <= y1 < frame.shape[0] and 0 <= y2 < frame.shape[0]):
                # Recortar a placa para processamento
                placa_carro_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                placa_carro_crop_gray = cv2.cvtColor(placa_carro_crop, cv2.COLOR_BGR2GRAY)
                _, placa_carro_crop_thresh = cv2.threshold(placa_carro_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Ler o texto da placa
                texto_detectado, confianca_texto_detectado = ler_placas(placa_carro_crop_thresh)

               

                #Salvando os registros no banco de dados
                if texto_detectado is not None and confianca_texto_detectado > confianca_gravar_texto:
                    salvar_no_postgres(frame_nmr, car_id, texto_detectado, confianca_texto_detectado)
                    
                    # Verificar se a placa já está registrada
                    info = verificar_placa_registrada(texto_detectado, cursor)

                    #Verifica qual camera esta sendo utilizada
                    tipo = verificar_camera(porta,cursor)
                    if info:
                        if frame_nmr not in range(frame_anterior + 1, frame_anterior + 10):
                            salvar_registro_frequencia(info['id_veiculo'],data_e_hora_em_texto,tipo)
                            print(f"Proprietário: {info['proprietario']}, Placa: {info['placa']}, Cor do Veículo: {info['cor']}, Marca: {info['marca_id']}, ID_veiculo {info['id_veiculo']}") 
                            print(barra)    
                        else:
                            print(barra_verm)
                        frame_anterior = frame_nmr
            else:
                print(f"Coordenadas de recorte fora dos limites: ({x1}, {y1}), ({x2}, {y2})")
        else:
            print("Nenhum veículo correspondente à placa foi detectado.")

cap.release()