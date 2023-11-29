import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'generated_images_packets/generated_images.npz')

d = 2
w = 16
n = 28
n_meios = int(n / d)
n_quartos = int(n / (d * d))

# Carregar o arquivo npz
with np.load(file_path, allow_pickle=True) as f:
    generated_packets = f["generated_packets"]

# Criar uma lista para armazenar os resultados de todas as imagens
all_packets = []

# Carregar cada matriz e fazer o processo inverso
# Primeiro, precisa-se calcular a média dos valores das submatrizes
# Depois, substituir o valor da submatriz pela média

image = np.array(generated_packets)
print(image)

contador = 0

for k in range(generated_packets.shape[0]):    
    packets = np.zeros((n_quartos, n_quartos), dtype=np.uint8)

    # Percorrer a imagem, calculando a média de cada submatriz de tamanho dxd
    for i in range(0, n, d):
        for j in range(0, n, d):
            packets[int(i/4), int(j/4)] = int((image[k, i:i+d, j:j+d].mean()))
            
            
            j += d
        i += d
        
    
    
    


    # Adicionar os resultados da imagem atual à lista geral
    all_packets.append(packets)
    
packets_hex = []
# for packets in all_packets:
    # packets_hex.append([hex(packet).replace('0x', '') for packet in packets])

# Imprimir os resultados de todas as imagens
for i, packets_print in enumerate(all_packets):
    print(f"Image {i + 1}: {packets_print}")
    
    print('---------------------------------------------------------------------------------------')
    
    print(f"Image {i + 1}: {packets}")
    # print(f"Image {i + 1}: {packets_hex}")
    
    print('---------------------------------------------------------------------------------------')
        

from scapy.all import wrpcap
import numpy as np

# Crie um pacote Scapy para cada conjunto de bytes e adicione-os à lista
scapy_packets = [bytes(packets) for packets in all_packets]
print(scapy_packets)

# Nome do arquivo .pcap que você deseja criar
pcap_filename = 'output_file.pcap'

# Salvar os pacotes Scapy em um arquivo .pcap
wrpcap(pcap_filename, scapy_packets)
