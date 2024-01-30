import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
generated_packets_dir = os.path.join(current_dir, 'generated_ip_packets_by_gan')

d = 2
w = 16
n = 28
n_meios = int(n / d)

def decode_packets(generated_packets_dir):
    '''Decodes the generated packets
    
    Args:
        generated_packets_dir (string): Path to the directory containing the generated packets
        
    Returns:
        string: Literal representation of the generated packets
    '''
    file_path = os.path.join(generated_packets_dir, 'generated_packets.npz')
    
    # Carregar o arquivo npz
    with np.load(file_path, allow_pickle=True) as f:
        generated_packets = f["generated_packets"]

    image = np.array(generated_packets)

    for k in range(generated_packets.shape[0]):    
        packets = np.zeros((n_meios, n_meios), dtype=np.uint8)

        # Percorrer a imagem, calculando a média de cada submatriz de tamanho dxd
        for i in range(0, n, d):
            for j in range(0, n, d):
                packets[int(i/2), int(j/2)] = int((image[k, i:i+d, j:j+d].mean()))
        
        packet_in_list = []
        
        for i in range(0, n_meios, 1):
            for j in range(0, n_meios, d):
                packet_in_list.append((str(hex(int(packets[i][j] / 16)).replace('0x', '')) + str(hex(int(packets[i][j+1] / 16)).replace('0x', ''))))
    
        ipv4 = packet_in_list[0:20]
        icmp = packet_in_list[20:84]

        literal_packet_in_list1 = ''.join(ipv4)
        literal_packet_in_list2 = ''.join(icmp)
        
    return literal_packet_in_list1, literal_packet_in_list2