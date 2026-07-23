from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis(name="buffalo_l")
app.prepare(
    ctx_id=0,
    providers=["CUDAExecutionProvider"]
)

print("Providers:", app.models['recognition'].session.get_providers())