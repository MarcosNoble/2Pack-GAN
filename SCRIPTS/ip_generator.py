import os
import pandas as pd
from nslookup import Nslookup

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'top500Domains.csv')

df = pd.DataFrame(pd.read_csv(file_path, sep=','))

dns_query = Nslookup()
dns_query = Nslookup(dns_servers=["1.1.1.1"], verbose=False, tcp=False)

print(df.head())

def ping_ip(dominio):
    response = os.system("ping -c 10 " + dominio)
    if response == 0:
        return True
    else:
        return False

def nslookup_request(dominio):
    ips_record = dns_query.dns_lookup(dominio)
    print(ips_record.response_full, ips_record.answer)
    if ips_record.answer is not None:
        return True
    else:
        return False

    
def main():
    os.system("sudo tcpdump -w segundo_ip.pcap -i eth0 icmp &")
    
    for index, row in df.iterrows():
        if ping_ip(row['Root Domain']):
        # if nslookup_request(row['Root Domain']):
            print(row['Root Domain'])
        else:
            print('Não foi possível pingar/DNS o domínio: ' + row['Root Domain'])

    
    os.system("pid=$(ps -e | pgrep tcpdump)")
    os.system("echo $pid")
    os.system("sleep 5")
    os.system("sudo kill -2 $pid")
    
if __name__ == '__main__':
    main()
    
