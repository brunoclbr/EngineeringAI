import requests

url = "http://127.0.0.1:8000/predict/"

x1 = [0]*61
x1[1] = 0.5
x1[10] = 0.5

data = {
    "catalyst": "PtAg",
    "miller_indices": "[1 1 1 0]",
    "adsorbate": "OH*",
    "x1": x1,
    "x2": [1, 1, 0],
    "graph_data": {
        "x": [[0.1, 0.2], [0.3, 0.4]],
        "edge_index": [[0, 1], [1, 0]],
        "edge_attr": [[0.5, 0.6], [0.7, 0.8]]
    }
}

response = requests.post(url, json=data)
print(response.json())