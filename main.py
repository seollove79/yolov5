from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import ffmpeg
import numpy as np
import torch
from pathlib import Path

# YOLOv5를 위한 설정
model = torch.hub.load("ultralytics/yolov5", "custom", path="runs/train/tomato_yolov5s_results3/weights/best.pt")
model.conf = 0.25  # confidence threshold
model.iou = 0.45  # NMS IoU threshold

app = FastAPI()

@app.get("/stream/")
async def video_feed():
    video_path = "tomato.mp4"

    # ffmpeg를 사용하여 동영상을 프레임으로 분할
    process = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run(capture_stdout=True, capture_stderr=True)
    )
    video = np.frombuffer(process[0], np.uint8).reshape([-1, 480, 640, 3])  # 임의의 해상도

    def generate():
        for frame in video:
            results = model(frame)
            frame = results.render()[0]
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
