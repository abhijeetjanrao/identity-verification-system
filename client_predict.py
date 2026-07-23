import sys
import requests

def predict(image_path: str, url: str = "http://127.0.0.1:8000/predict"):
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        resp = requests.post(url, files=files)

    print(f"Status: {resp.status_code}")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_predict.py path/to/image.jpg [url]")
        sys.exit(1)

    image = sys.argv[1]
    endpoint = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000/predict"
    predict(image, endpoint)
