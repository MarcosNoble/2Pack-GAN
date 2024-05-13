from operator import contains
import numpy as np
import os
import pyshark

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

pcap_dir = os.path.join(parent_dir, 'PCAPS')

npz_dir = os.path.join(current_dir, 'NPZ')

if not os.path.exists(f"{npz_dir}"):
    os.makedirs(f"{npz_dir}")


def packet_useful_data(packet, total_packets):
    """Returns the useful data of a packet

    Args:
        packet (string): Packet

    Returns:
        string: Useful data of the packet
    """
    if len(packet) < 42:
        return packet[0:196], 'Other', total_packets
    
    if(packet[46] + packet[47] == '01'):
        total_packets[0] += 1
        return packet[28:196], 'ICMP', total_packets # ICMP
    
    elif(packet[46] + packet[47] == '11' and (packet[74:76] == '35' or packet[70:72] == '35')):
        total_packets[1] += 1
        return packet[28:196], 'DNS', total_packets # DNS
    
    elif(packet[46] + packet[47] == '11'):
        total_packets[4] += 1
        return packet[28:196], 'UDP', total_packets  # UDP
    
    elif (packet[46] + packet[47] == '06' and (packet[70:72] == '15')):
        total_packets[3] += 1
        return packet[28:148], 'FTP', total_packets # FTP
    
    elif(packet[46] + packet[47] == '06'):
        total_packets[2] += 1
        return packet[28:148], 'TCP', total_packets  # TCP
    
    else:
        total_packets[5] += 1
        return packet[28:196], 'Other', total_packets
    
    
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
    
    # Initialize the n x n matrix with zeros of object type
    matrix = np.zeros((n, n), dtype=object)

    i, j = 0, 0
    for byte_digit in byte_digits:
        # Fill the d x d submatrix with the entered byte value (as string)
        value = int(byte_digit, 16)  # Convert the hexadecimal string to integer
        matrix[i:i+d, j:j+d] = value

        # Update the indexes
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
    # Mapeamento de protocolos para números
    protocol_to_label = {'ICMP': 0, 'DNS': 1, 'TCP': 2, 'FTP': 3, 'UDP': 4, 'Other': 5}

    index = 0
    
    dataset = {"x_train": [], "y_train": [], "x_test": [], "y_test": []}
    
    total_packets = [0, 0, 0, 0, 0, 0]
    
    pcap_files = os.listdir(pcap_dir)
    
    for pcap_name in pcap_files:
        if index == 200000:
                break
        
        # pcap_name = input("Enter the name of the pcap file: ")
        pcap_path = os.path.join(pcap_dir, pcap_name)
        pcap = pyshark.FileCapture(pcap_path, use_json=True, include_raw=True)

        print(pcap_name)
        

        for pkt in pcap:
            packet = pkt.frame_raw.value
            packet, protocol, total_packets = packet_useful_data(packet, total_packets)
            
            label = protocol_to_label.get(protocol, protocol_to_label['Other'])

            if packet[2:4] == '00' and total_packets[label] <= 40000 and label != 5: # Verifications for ICMP port reachable.
                index = total_packets[0] + total_packets[1] + total_packets[2] + total_packets[3] + total_packets[4]
                packet = packet_means(packet)
                packet = duplicate_and_map_bytes(packet)
                        
                if index >= 0:
                    dataset["x_train"].append(packet.tolist())
                    dataset["y_train"].append(label)
                    
                else:
                    dataset["x_test"].append(packet.tolist())
                    dataset["y_test"].append(label)
        
                print(f"Packet {index} of {200000}")
                
                print(f"O numero de pacotes do protocolo ICMP é {total_packets[0]}")
                print(f"O numero de pacotes do protocolo DNS é {total_packets[1]}")
                print(f"O numero de pacotes do protocolo TCP é {total_packets[2]}")
                print(f"O numero de pacotes do protocolo FTP é {total_packets[3]}")
                print(f"O numero de pacotes do protocolo UDP é {total_packets[4]}")
            
            else:
                total_packets[label] -= 1
                    
            if index == 200000:
                break
            
            if contains(pcap_name.lower(), protocol.lower()) and total_packets[label] == 40000:
                break
            
    
    npz_file = input("Enter the name of the npz file: ")
    npz_file = npz_file + '.npz'
        
    np.savez(os.path.join(npz_dir, npz_file), **dataset)
    
if __name__ == "__main__":
    main()