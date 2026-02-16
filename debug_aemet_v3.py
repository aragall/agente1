import requests

def test_aemet_v3():
    print("Testing connectivity to opendata.aemet.es (V3)...")
    requests.packages.urllib3.disable_warnings()
    
    # List of candidate endpoints
    endpoints = [
        "https://opendata.aemet.es/opendata/api/avisos_de_fenomenos_meteorologicos_adversos/ultimo_elaborado",
        "https://opendata.aemet.es/opendata/api/avisos_de_fenomenos_meteorologicos_adversos/archivo/hoy",
        "https://opendata.aemet.es/opendata/api/maestro/municipios" # Should definitely exist
    ]
    
    for url in endpoints:
        print(f"--- Requesting {url} ---")
        try:
            res = requests.get(url, verify=False, timeout=10)
            print(f"Status: {res.status_code}")
            if res.status_code == 401:
                print("Got 401 (Auth required). Endpoint EXISTS.")
            elif res.status_code == 200:
                print("Got 200 OK.")
            elif res.status_code == 404:
                print("Got 404 Not Found.")
            else:
                print(f"Got {res.status_code}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_aemet_v3()
