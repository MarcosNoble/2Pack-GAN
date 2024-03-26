import numpy as np
import os
import pyshark

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
pcap_dir = os.path.join(parent_dir, 'PCAPS')
npz_dir = os.path.join(parent_dir, 'NPZ')

if not os.path.exists(f"{npz_dir}"):
    os.makedirs(f"{npz_dir}")


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
    pcap_name = input("Enter the name of the pcap file: ")
    pcap_name = pcap_name + '.pcap'
    
    pcap_path = os.path.join(pcap_dir, pcap_name)
    
    dataset = {"x_train": [], "y_train": [], "x_test": [], "y_test": []}
    
    pcap = pyshark.FileCapture(pcap_path, use_json=True, include_raw=True)
    
    total_packets = 0
    for pkt in pcap:
        total_packets += 1
        
    index = 0
    
    for pkt in pcap:
        print(f"Packet {index + 1} of {total_packets}")
        
        packet = pkt.frame_raw.value
        packet = packet_useful_data(packet)

        
        if packet[2:4] == '00': # Verifications for ICMP port reachable.
            packet = packet_means(packet)
            packet = duplicate_and_map_bytes(packet)
            
            if index <= 0.8 * total_packets:
                dataset["x_train"].append(packet.tolist())
                dataset["y_train"].append(packet.tolist())
                
            else:
                dataset["x_test"].append(packet.tolist())
                dataset["y_test"].append(packet.tolist())
        
        index += 1
        
    npz_file = input("Enter the name of the npz file: ")
    npz_file = npz_file + '.npz'
        
    np.savez(os.path.join(npz_dir, npz_file), **dataset)
    
if __name__ == "__main__":
    main()