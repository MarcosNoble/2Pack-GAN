'''
icmp_data = ('10' '11' '12' '13' '14'
             '15' '16' '17' '18' '19'
             '1a' '1b' '1c' '1d' '1e'
             '1f' '20' '21' '22' '23'
             '24' '25' '26' '27' '28'
             '29' '2a' '2b' '2c' '2d'
             '2e' '2f' '30' '31' '32' 
             '33' '34' '35' '36' '37')

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
                '00 18')                #Sequence Number (arbitrary)
'''