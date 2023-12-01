import os
import pandas as pd
import json
from nslookup import Nslookup

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'top500Domains.csv')

# domains_unreachable = []
domains_unreachable = json.load(open('domains_unreachable.json'))

df = pd.DataFrame(pd.read_csv(file_path, sep=','))

dns_query = Nslookup()
dns_query = Nslookup(dns_servers=["1.1.1.1"], verbose=False, tcp=False)

# print(df.head()) # Debug para ver se o arquivo foi lido corretamente

def ping_ip(dominio): # ICMP
    if dominio not in domains_unreachable:
        response = os.system("ping -c 20 " + dominio)
    else:
        return False
    
    if response == 0:
        return True
    else:
        domains_unreachable.append(dominio)
        return False

def nslookup_request(dominio): # DNS
    ips_record = dns_query.dns_lookup(dominio)
    print(ips_record.response_full, ips_record.answer)
    if ips_record.answer is not None:
        return True
    else:
        return False

    
def main():
    os.system("sudo tcpdump -w terceiro_ping.pcap -i eth0 icmp &") # Iniciar o processo de captura icmp
    # os.system("sudo tcpdump -w ajustes.pcap -i eth0 udp port 53 &") # Iniciar o processo de captura dns
    
    for index, row in df.iterrows():
        if ping_ip(row['Root Domain']): # ICMP
        # if nslookup_request(row['Root Domain']): # DNS
            print(row['Root Domain'])
        else:
            print('Não foi possível pingar/DNS o domínio: ' + row['Root Domain'])
        
    os.system("sudo pkill -2 tcpdump") # Matar o processo
    
    # Salvar a lista de domínios não pingados em um arquivo JSON
    with open('domains_unreachable.json', 'w') as f:
        json.dump(domains_unreachable, f)
        
    
if __name__ == '__main__':
    main()
    
