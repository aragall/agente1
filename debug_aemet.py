import requests
import json

def test_aemet_connectivity_v2():
    print("Testing connectivity to opendata.aemet.es (V2)...")
    try:
        # Try 'ultimo' instead of 'archivo/hoy'
        url = "https://opendata.aemet.es/opendata/api/avisos_de_fenomenos_meteorologicos_adversos/ultimo"
        
        requests.packages.urllib3.disable_warnings()
        
        # Test 1: No key (expect 401)
        print(f"Requesting {url} without key...")
        res = requests.get(url, verify=False, timeout=10)
        print(f"Status Code: {res.status_code}")
        
        if res.status_code == 404:
            print("Still 404. Path is wrong.")
        elif res.status_code == 401:
            print("Got 401. This path EXISTS (requires auth). Success!")
        else:
            print(f"Got {res.status_code}. Response: {res.text[:100]}")

    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    test_aemet_connectivity_v2()
