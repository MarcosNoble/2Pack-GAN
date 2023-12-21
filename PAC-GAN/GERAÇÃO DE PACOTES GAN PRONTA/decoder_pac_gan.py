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
            lista.append((str(hex(int(packets[i][j] / 16)).replace('0x', '')) + str(hex(int(packets[i][j+1] / 16)).replace('0x', ''))))
            # print(i, j)

            
            
        
    lista1 = lista[0:20]
    lista2 = lista[20:84]
    
    print(lista1)
    print(lista2)
    
    literal_lista1 = ''.join(lista1)
    literal_lista2 = ''.join(lista2)
    
        
import sys
import binascii

ipv4_header = literal_lista1[0:4] + 'XX' 'XX' + literal_lista1[8:20] + 'YY' 'YY' + literal_lista1[24:40]


icmp_header_data = literal_lista2[0:4] + 'XX' 'XX' + literal_lista2[8:16] + literal_lista2[48:128]

'''
icmp_data = ('10' '11' '12' '13' '14'
             '15' '16' '17' '18' '19'
             '1a' '1b' '1c' '1d' '1e'
             '1f' '20' '21' '22' '23'
             '24' '25' '26' '27' '28'
             '29' '2a' '2b' '2c' '2d'
             '2e' '2f' '30' '31' '32' 
             '33' '34' '35' '36' '37')
'''

#Global header for pcap 2.4
pcap_global_header =   ('D4 C3 B2 A1'   
                        '02 00'         #File format major revision (i.e. pcap <2>.4)  
                        '04 00'         #File format minor revision (i.e. pcap 2.<4>)   
                        '00 00 00 00'     
                        '00 00 00 00'     
                        'FF FF 00 00'     
                        '01 00 00 00')

#pcap packet header that must preface every packet
pcap_packet_header =   ('AA 77 9F 47'     
                        '90 A2 04 00'     
                        'XX XX XX XX'   #Frame Size (little endian) 
                        'YY YY YY YY')  #Frame Size (little endian)

eth_header =   ('00 00 00 00 00 00'     #Source Mac    
                '00 00 00 00 00 00'     #Dest Mac  
                '08 00')                #Protocol (0x0800 = IP)
                
'''

ip_header =    ('45'                    #IP version and header length (multiples of 4 bytes)   
                '00'                      
                'XX XX'                 #Length - will be calculated and replaced later
                '55 23'                   
                '40 00 40'                
                '01'                    #Protocol (0x01 = ICMP)          
                'YY YY'                 #Checksum - will be calculated and replaced later      
                '8C 11 CD 0B'           #Source IP       
                '59 00 57 87')          #Dest IP 

icmp_header =  ('08'                    #ICMP Type (8 = echo request, 0 = echo reply)
                '00'                    #Code
                'XX XX'                 #Checksum - will be calculated and replaced later
                '02 49'                 #Identifier (arbitrary)
                '00 18')                #Sequence Number (arbitrary)'''


def getByteLength(str1):
    return int(len(''.join(str1.split())) / 2)

def writeByteStringToFile(bytestring, filename):
    bytelist = bytestring.split()  
    bytes = binascii.a2b_hex(''.join(bytelist))
    bitout = open(filename, 'wb')
    bitout.write(bytes)
    
#Splits the string into a list of tokens every n characters
def splitN(str1,n):
    return [str1[start:start+n] for start in range(0, len(str1), n)]
    
#Calculates and returns the IP checksum based on the given IP Header
def ip_checksum(ip):

    #split into bytes    
    words = splitN(''.join(ip.split()),4)

    csum = 0
    for word in words:
        csum += int(word, base=16)

    csum += (csum >> 16)
    csum = csum & 0xFFFF ^ 0xFFFF

    return csum

def icmp_checksum(icmp):
    words = splitN(''.join(icmp.split()),4)
    print(words)

    csum = 0
    for word in words:
        csum += int(word, base=16)
    

    csum += (csum >> 16)
    csum = csum & 0xFFFF ^ 0xFFFF
    
    return csum

    
def generatePcapFile(filename):
    icmp_len = getByteLength(icmp_header_data)
    icmp = icmp_header_data
    checksum = icmp_checksum(icmp.replace('XXXX','00 00'))
    icmp = icmp.replace('XXXX',"%04x"%checksum)
    
    ip_len = icmp_len + getByteLength(ipv4_header)
    ip = ipv4_header.replace('XXXX',"%04x"%ip_len)
    checksum = ip_checksum(ip.replace('YYYY','00 00'))
    ip = ip.replace('YYYY',"%04x"%checksum)
    
    pcap_len = ip_len + getByteLength(eth_header)
    hex_str = "%08x"%pcap_len
    reverse_hex_str = hex_str[6:] + hex_str[4:6] + hex_str[2:4] + hex_str[:2]
    pcaph = pcap_packet_header.replace('XX XX XX XX',reverse_hex_str)
    pcaph = pcaph.replace('YY YY YY YY',reverse_hex_str)
        
    bytestring = pcap_global_header + pcaph + eth_header + ip + icmp
    writeByteStringToFile(bytestring, pcapfile)
        

pcapfile = 'test.pcap'
generatePcapFile(pcapfile)