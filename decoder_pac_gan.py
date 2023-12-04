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

contador = 0

for k in range(generated_packets.shape[0]):    
    packets = np.zeros((n_meios, n_meios), dtype=np.uint8)

    # Percorrer a imagem, calculando a média de cada submatriz de tamanho dxd
    for i in range(0, n, d):
        for j in range(0, n, d):
            packets[int(i/2), int(j/2)] = int((image[k, i:i+d, j:j+d].mean()))
            
            
        
    print(packets)

    
    lista = []
    
    for i in range(0, n_meios, 1):
        for j in range(0, n_meios, d):
            lista.append(int((str(hex(int(packets[i][j] / 16)).replace('0x', '')) + str(hex(int(packets[i][j+1] / 16)).replace('0x', ''))), 16))
            # print(i, j)

            
            
            

        print(i, j)
        
    lista = lista[0:84]
    
    print(lista)
    print(len(lista))
            
    
    
    
    all_packets.append(lista)
    

# Imprimir os resultados de todas as imagens
for i, packets_print in enumerate(all_packets):
    #print(f"Image {i + 1}: {packets_print}")
    
    print('---------------------------------------------------------------------------------------')
    
    print('---------------------------------------------------------------------------------------')
        

from scapy.all import wrpcap, Ether
import numpy as np

scapy_packets = []

for lista in all_packets:
    eth_packet = Ether(dst="00:11:22:33:44:55", src="66:77:88:99:aa:bb", type=0x800)
    ipv4 = bytes(lista[0:64])
    icmp = bytes(lista[64:84])

    # Adicionar pacote Scapy à lista
    scapy_packets.append(eth_packet / ipv4 / icmp)

# Nome do arquivo .pcap que você deseja criar
pcap_filename = 'esse.pcap'

# Salvar os pacotes Scapy em um arquivo .pcap
wrpcap(pcap_filename, scapy_packets)