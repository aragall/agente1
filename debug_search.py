from duckduckgo_search import DDGS
import json

def test_search():
    log = []
    query = "test search"
    
    # Test default/api
    try:
        log.append("Testing API backend...")
        with DDGS() as ddgs:
            res = list(ddgs.text(query, max_results=3))
            log.append(f"API Result: {len(res)} items found.")
    except Exception as e:
        log.append(f"API Failed: {e}")

    # Test html
    try:
        log.append("Testing HTML backend...")
        with DDGS() as ddgs:
            res = list(ddgs.text(query, max_results=3, backend="html"))
            log.append(f"HTML Result: {len(res)} items found.")
    except Exception as e:
        log.append(f"HTML Failed: {e}")
        
    # Test lite
    try:
        log.append("Testing Lite backend...")
        with DDGS() as ddgs:
            res = list(ddgs.text(query, max_results=3, backend="lite"))
            log.append(f"Lite Result: {len(res)} items found.")
    except Exception as e:
        log.append(f"Lite Failed: {e}")

    with open("search_log.txt", "w") as f:
        f.write("\n".join(log))

if __name__ == "__main__":
    test_search()
