from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(224,224))

img_path = r"C:\Users\Asus\OneDrive\Desktop\face\bollywood_celeb_faces_0\Aamir_Khan\Aamir.62.jpg"

img = cv2.imread(img_path)

print("Image is None:", img is None)

if img is None:
    print("❌ IMAGE NOT LOADING — CHECK PATH")
    exit()

faces = app.get(img)

print("Faces detected:", len(faces))