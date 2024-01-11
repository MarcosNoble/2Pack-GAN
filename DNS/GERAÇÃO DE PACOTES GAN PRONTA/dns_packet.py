from gan_packets_generator import generate_packets_by_gan
from decoder_pac_gan_dns import decode_packets
import sys
import binascii
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
generated_bytes_dir = os.path.join(current_dir, 'generated_dns_bytes_by_gan')
generated_packets_dir = os.path.join(current_dir, 'generated_dns_packets_by_gan')

if not os.path.exists(generated_packets_dir):
    os.makedirs(generated_packets_dir)

#Global header for pcap 2.4
pcap_global_header = ('D4 C3 B2 A1'   
                      '02 00'         #File format major revision (i.e. pcap <2>.4)  
                      '04 00'         #File format minor revision (i.e. pcap 2.<4>)   
                      '00 00 00 00'     
                      '00 00 00 00'     
                      'FF FF 00 00'     
                      '01 00 00 00')

#pcap packet header that must preface every packet
pcap_packet_header = ('AA 77 9F 47'     
                      '90 A2 04 00'     
                      'XX XX XX XX'   #Frame Size (little endian) 
                      'YY YY YY YY')  #Frame Size (little endian)

eth_header = ('00 00 00 00 00 00'     #Source Mac    
              '00 00 00 00 00 00'     #Dest Mac  
              '08 00')                #Protocol (0x0800 = IP)
                
def getByteLength(str1):
    '''Returns the length of a string in bytes
    
    Args:
        str1 (string): String
        
    Returns:
        integer: Length of the string in bytes
    '''
    return int(len(''.join(str1.split())) / 2)

def writeByteStringToFile(bytestring, filename):
    '''Writes a bytestring to a file
    
    Args:
        bytestring (string): Bytestring to be written to the file
        filename (string): Name of the file to be written
    '''
    bytelist = bytestring.split()  
    bytes = binascii.a2b_hex(''.join(bytelist))
    bitout = open(filename, 'wb')
    bitout.write(bytes)
    
#Splits the string into a list of tokens every n characters
def splitN(str1, n):
    '''Splits the string into a list of tokens every n characters
    
    Args:
        str1 (string): String to be split
        n (integer): Number of characters to split the string
        
    Returns:
        list: List of tokens
    '''
    return [str1[start:start+n] for start in range(0, len(str1), n)]
    
#Calculates and returns the IP checksum based on the given IP Header
def ip_checksum(ip):
    '''Calculates and returns the IP checksum based on the given IP Header
    
    Args:
        ip (string): IP Header
        
    Returns:
        integer: IP checksum
            
    '''
    #split into bytes    
    words = splitN(''.join(ip.split()),4)

    csum = 0
    for word in words:
        csum += int(word, base=16)

    csum += (csum >> 16)
    csum = csum & 0xFFFF ^ 0xFFFF

    return csum

def udp_checksum(udp):
    '''Calculates and returns the UDP checksum based on the given UDP Header
    
    Args:
        udp (string): UDP Header
        
    Returns:
        integer: UDP checksum
            
    '''
    #split into bytes    
    words = splitN(''.join(udp.split()),4)

    csum = 0
    for word in words:
        csum += int(word, base=16)

    csum += (csum >> 16)
    csum = csum & 0xFFFF ^ 0xFFFF

    return csum

def generatePcapFile(filename):
    '''Generates a pcap file based on the given filename
    
    Args:
        filename (string): Name of the pcap file to be generated
        
    Returns:
        string: Bytestring of the generated pcap file
    '''
    udp_len = getByteLength(udp_header_data)
    print(udp_len)
    udp = udp_header_data.replace('XXXX',"%04x"%udp_len)
    checksum = udp_checksum(udp.replace('YYYY','00 00'))
    udp = udp.replace('YYYY',"%04x"%checksum)
    
    ip_len = udp_len + getByteLength(ipv4_header)
    ip = ipv4_header.replace('XXXX',"%04x"%ip_len)
    checksum = ip_checksum(ip.replace('YYYY','00 00'))
    ip = ip.replace('YYYY',"%04x"%checksum)
    
    pcap_len = ip_len + getByteLength(eth_header)
    hex_str = "%08x"%pcap_len
    reverse_hex_str = hex_str[6:] + hex_str[4:6] + hex_str[2:4] + hex_str[:2]
    pcaph = pcap_packet_header.replace('XX XX XX XX',reverse_hex_str)
    pcaph = pcaph.replace('YY YY YY YY',reverse_hex_str)
        
    bytestring = pcap_global_header + pcaph + eth_header + ip + udp
    
    output_path = os.path.join(generated_packets_dir, filename)
    
    writeByteStringToFile(bytestring, output_path)
    
number_of_packets = int(input("Type the number of packets to generate: "))

generate_packets_by_gan(number_of_packets, generated_bytes_dir)

literal_lista1, literal_lista2 = decode_packets(generated_bytes_dir)

ipv4_header = literal_lista1[0:4] + 'XX' 'XX' + literal_lista1[8:20] + 'YY' 'YY' + literal_lista1[24:40]
udp_header_data = literal_lista2[0:8] + 'XX' 'XX' + 'YY' 'YY' + literal_lista2[16:]
print(udp_header_data)
        
pcap_name = input("Type the name of the pcap file: ")        
pcapfile = pcap_name + '.pcap'
generatePcapFile(pcapfile)
print("Pcap file generated!")