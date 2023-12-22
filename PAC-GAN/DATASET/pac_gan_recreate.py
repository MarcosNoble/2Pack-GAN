import numpy as np
import os
import pyshark

current_dir = os.path.dirname(os.path.abspath(__file__))

# Caminho para o arquivo PCAPNG
file_path = os.path.join(current_dir, 'terceiro_ping.pcap')

def packet_useful_data(packet):
    """Returns the useful data of a packet

    Args:
        packet (string): Packet

    Returns:
        string: Useful data of the packet
    """
    return packet[28:196]  # ICMP


def packet_means(packet):
    """Returns the means of the bytes of a packet
    
    Args:
        packet (string): Packet
        
    Returns:
        list: List of means of the bytes of a packet
    """
    packet_list = []
    packet_with_mean = []
    
    for i in range(0, len(packet), 2):
        packet_list.append(packet[i:i+2])
        
    for i in range(0, 14):
        packet_list.append('00')
                
    for i in range(0, len(packet_list)):
        packet_with_mean.append(packet_list[i][:1] + '8')
        packet_with_mean.append(packet_list[i][1:] + '8')
        
    return packet_with_mean


def duplicate_and_map_bytes(byte_digits, n=28, d=2):
    """Duplicates and maps the bytes of a packet
    
    Args:
        byte_digits (list): List of bytes
        n (integer): Size of the matrix
        d (integer): Size of the submatrix
        
    Returns:
        numpy.ndarray: Matrix of bytes
    """
    
    # Inicializar a matriz n x n com zeros do tipo object
    matrix = np.zeros((n, n), dtype=object)

    i, j = 0, 0
    for byte_digit in byte_digits:
        # Preencher a submatriz d x d com o valor do byte digitado (como string)
        value = int(byte_digit, 16)  # Converter a string hexadecimal para inteiro
        matrix[i:i+d, j:j+d] = value

        # Atualizar os índices para a próxima submatriz
        j += d
        if j >= n:
            j = 0
            i += d
            if i >= n:
                break

    return matrix


def main():
    """Main function
    """
    total_packets = int(input("How many packets do you want to use? "))
    
    dataset = {"x_train": [], "y_train": [], "x_test": [], "y_test": []}
    
    pcap = pyshark.FileCapture(file_path, use_json=True, include_raw=True)
    
    index = 0
    
    for pkt in pcap:
        print(index)
        
        packet = pkt.frame_raw.value
        packet = packet_useful_data(packet)
        packet = packet_means(packet)
        packet = duplicate_and_map_bytes(packet)
        
        if index <= 0.8 * total_packets:
            dataset["x_train"].append(packet.tolist())
            dataset["y_train"].append(packet.tolist())
            
        else:
            dataset["x_test"].append(packet.tolist())
            dataset["y_test"].append(packet.tolist())
        
        index += 1
        
    np.savez(os.path.join(current_dir, 'dataset_ip.npz'), **dataset)
    
if __name__ == "__main__":
    main()