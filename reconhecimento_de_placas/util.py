import string
import easyocr
import pandas as pd
import psycopg2
import cv2  # adicionado para exibir imagens

conexao = psycopg2.connect(
    dbname="pci_transito",
    user="postgres",
    password="123456",
    host="localhost",
    port="5432"
)
cursor = conexao.cursor()

reader = easyocr.Reader(['en'], gpu=False)

# Dicionario de conversões de caracteres entre char e int
char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    }

int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    }

#Função para definir os formatos de placas que irá ler
def license_complies_format(text):
      
    if len(text) != 7:
        return False
    
    
    

    # Teste
    if (text[0] in string.ascii_uppercase or text[0] in int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in int_to_char.keys()):
        return True
     

    #Formato padrão
    if (text[0] in string.ascii_uppercase or text[0] in int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in int_to_char.keys()) and \
       (text[2] in string.ascii_uppercase or text[2] in int_to_char.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in char_to_int.keys()) and \
       (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in char_to_int.keys()) and \
       (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in char_to_int.keys()) and \
       (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in char_to_int.keys()):
        return True
    #Formato Mercosul
    if (text[0] in string.ascii_uppercase or text[0] in int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in int_to_char.keys()) and \
       (text[2] in string.ascii_uppercase or text[2] in int_to_char.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in int_to_char.keys()) and \
       (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in char_to_int.keys()) and \
       (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in char_to_int.keys()):
        return True
    

    return False

#Função para verificar o formato da placa

def formato_placa(text):
    license_plate_ = ''
    for j in range(7):
        if j in [0, 1, 2]:
            if text[j] in int_to_char:
                license_plate_ += int_to_char[text[j]]
            else:
                license_plate_ += text[j]
        # Se estivermos nas posições que podem ser números (para ambos os formatos)
        elif j == 3:
            # Posição 3 pode ser um número em ambos os formatos
            if text[j] in char_to_int:
                license_plate_ += char_to_int[text[j]]
            else:
                license_plate_ += text[j]

        # Posição 4 pode ser uma letra (no formato `AAA1A23`) ou um número (no formato `AAA1234`)
        elif j == 4:
            if text[j] in string.ascii_uppercase or text[j] in int_to_char:
                # Mapear letra para número, se aplicável
                if text[j] in int_to_char:
                    license_plate_ += int_to_char[text[j]]
                else:
                    license_plate_ += text[j]
            elif text[j] in char_to_int:
                # Mapear número para letra, se aplicável
                license_plate_ += char_to_int[text[j]]
            else:
                license_plate_ += text[j]

        # Posições 5 e 6 sempre podem ser números (em ambos os formatos)
        elif j in [5, 6]:
            if text[j] in char_to_int:
                license_plate_ += char_to_int[text[j]]
            else:
                license_plate_ += text[j]

    return license_plate_

#Funções Banco de dados

def ler_placas(placa_carro_crop):
    cv2.imshow("Imagem da placa", placa_carro_crop)  # exibe a imagem da placa
    cv2.waitKey(1)  # delay curto para atualização em "vídeo"
    detections = reader.readtext(placa_carro_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return formato_placa(text), score

    return None, None
def ler_carro(placa, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = (*placa, None)

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
def verificar_camera(porta, cursor):
    try:
        comando_sql = """
        SELECT entradas FROM transito_camera
        WHERE porta = %s;
        """
        cursor.execute(comando_sql, (porta,))
        resultado = cursor.fetchone()
        if resultado:
            return  resultado[0],   
        else:
            return None
    except Exception as e:
        print(f"Erro ao verificar camera no banco de dados: {e}")
        return None
def salvar_no_postgres(frame_nmr, car_id, license_number, license_number_score):
    try:
        comando_sql = """
        INSERT INTO transito_leitura_placa (frame_nmr,car_id,license_number,license_number_score)
        VALUES (%s, %s, %s,%s);
        """
        valores = (int(frame_nmr), int(car_id), license_number, float(license_number_score))
        cursor.execute(comando_sql, valores)
        conexao.commit()
    except Exception as e:
        print(f"Erro ao inserir dados: {e}")    
        conexao.rollback()
def salvar_registro_frequencia(id_veiculo,data,tipo):
    try:
        comando_sql = """
        INSERT INTO transito_registro_entrada_saida (id_veiculo,data,tipo)
        VALUES (%s,%s,%s);
        """
        valores = (id_veiculo,data,tipo)
        cursor.execute(comando_sql, valores)
        conexao.commit()
    except Exception as e:
        print(f"Erro ao inserir dados: {e}")    
        conexao.rollback()
def verificar_placa_registrada(placa, cursor):  


    try:
        comando_sql = """
        SELECT proprietario,placa,cor_id,marca_id,id FROM transito_veiculo
        WHERE placa = %s;
        """
        cursor.execute(comando_sql, (placa,))
        resultado = cursor.fetchone()
        if resultado:
            return {
                "proprietario": resultado[0],
                "placa": resultado[1],
                "cor": resultado[2],
                "marca_id" : resultado[3],
                "id_veiculo": resultado[4],
            }
        else:
            return None
    except Exception as e:  
        print(f"Erro ao verificar placa no banco de dados: {e}")
        return None
def teste(a):
    print('a')