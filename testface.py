import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

img = cv2.imread("96.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w = img.shape[:2]
if max(h, w) > 1000:
    scale = 1000 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

faces = app.get(img)

print("Number of faces detected:", len(faces))