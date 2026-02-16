import requests

def test_aemet_v4():
    print("Testing connectivity to opendata.aemet.es (V4 - Headers)...")
    requests.packages.urllib3.disable_warnings()
    
    url = "https://opendata.aemet.es/opendata/api/avisos_de_fenomenos_meteorologicos_adversos/archivo/hoy"
    
    # Try with headers (simulating a browser/valid client)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    print(f"Requesting {url} with headers...")
    try:
        res = requests.get(url, headers=headers, verify=False, timeout=10)
        print(f"Status: {res.status_code}")
        if res.status_code == 404:
            print("Still 404.")
        elif res.status_code == 401:
            print("Got 401. Success! Endpoint exists.")
        else:
            print(f"Got {res.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_aemet_v4()
