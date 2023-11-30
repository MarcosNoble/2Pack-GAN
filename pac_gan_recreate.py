import numpy as np
import os
import random
import pyshark
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    index = 0
    dataset = {"x_train": [], "y_train": [], "x_test": [], "y_test": []}
    total_packets = 80000  # Defina o número total de pacotes
    n = 28
    d = 2

    # Caminho para o arquivo PCAPNG
    file_path = os.path.join(current_dir, 'segundo_ip.pcap')
    # file_path = os.path.join(current_dir, 'primeiro_dns.pcap')

    pcap = pyshark.FileCapture(file_path, use_json=True, include_raw=True)

    for pkt in pcap:
        if index > 0:
            pacote_hexadecimal = pkt.frame_raw.value
            pacote_hexadecimal = pacote_hexadecimal[0:168]
            
            lista_hexadecimal = []
            lista_hexadecimal_com_media = []

            for i in range(0, len(pacote_hexadecimal), 2):
                lista_hexadecimal.append(pacote_hexadecimal[i] + pacote_hexadecimal[i+1])
                
            for i in range(0, 14):
                lista_hexadecimal.append('00')
                
                
            print(lista_hexadecimal)
            print(len(lista_hexadecimal))
                
            # generate_ip(lista_hexadecimal)
            # remove_checksums(lista_hexadecimal)         
            
            for i in range(0, len(lista_hexadecimal)):
                #separar cada byte 
                lista_hexadecimal_com_media.append(lista_hexadecimal[i][:1] + '8')
                lista_hexadecimal_com_media.append(lista_hexadecimal[i][1:] + '8')
                
            # print(len(lista_hexadecimal_com_media))
            print(lista_hexadecimal_com_media)
            print(len(lista_hexadecimal_com_media))
            
            result_matrix = duplicate_and_map_bytes(lista_hexadecimal_com_media, n, d)
            print(result_matrix)
            return
            
            # Divide os dados em 80% treinamento e 20% teste
            if index <= 0.8 * total_packets:
                dataset["x_train"].append(result_matrix.tolist())
                dataset["y_train"].append(result_matrix.tolist())
            else:
                dataset["x_test"].append(result_matrix.tolist())
                dataset["y_test"].append(result_matrix.tolist())
                
            # transformar em imagem                   
            # img = Image.fromarray(np.array(result_matrix, dtype=np.uint8))
            # img.save(f'image_{index}.png')

            # Limpar variáveis
            del pacote_hexadecimal
            del lista_hexadecimal
            del lista_hexadecimal_com_media
            del result_matrix

            print(index)
        index += 1

    # Salvar o dataset em um arquivo NPZ
    np.savez(os.path.join(current_dir, 'dataset.npz'), **dataset)

def generate_ip(lista_hexadecimal):
    ip = []
    for i in range(0, 8):
        ip.append(hex(random.randint(0, 255)).replace('0x', ''))
        lista_hexadecimal[29 + i] = ip[i]

def remove_checksums(lista_hexadecimal):
    lista_hexadecimal[28] = '00'
    lista_hexadecimal[29] = '00'
    
def gerar_matriz_7x7(lista_hexadecimal):
    matriz = []
    for i in range(0, 7):
        linha = []
        for j in range(0, 7):
            linha.append(lista_hexadecimal[i * 7 + j])
        matriz.append(linha)
    return matriz


def duplicate_and_map_bytes(byte_digits, n, d):
    # Inicializar a matriz n x n com zeros do tipo object
    matrix = np.zeros((n, n), dtype=object)

    i, j = 0, 0
    for byte in byte_digits:
        matrix[i][j] = byte
        matrix[i][j + 1] = byte
        matrix[i + 1][j] = byte
        matrix[i + 1][j + 1] = byte
        
        j += 2
        if j >= n:
            j = 0
            i += 2
        if i >= n:
            break
    
    return matrix

main()