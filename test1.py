from fastapi import FastAPI, Query
from starlette.responses import StreamingResponse
import cv2
import torch
import uvicorn

# 사용자 정의 모델 로드
model_path = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

app = FastAPI()

def preprocess_frame(frame):
    # 1. 밝기와 대비 조정
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=30)
    # 2. 가우시안 블러
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    # 3. 히스토그램 평활화
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame

@app.get("/video/")
async def analyze_video(video_url: str = Query(..., description="The URL of the video to be analyzed")):
    # URL로부터 cv2.VideoCapture로 동영상 읽기
    cap = cv2.VideoCapture(video_url)

    def gen_frames():
        count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = preprocess_frame(frame)
            results = model(frame)
            text_position = (frame.shape[1] - 300, frame.shape[0] - 30)
            for det in results.xyxy[0].cpu().numpy():
                x_min, y_min, x_max, y_max, _, _ = det
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                if x_center>=930 and x_center<=970:
                    count = count + 1
                half_width = 5
                half_height = 5
                cv2.rectangle(frame, 
                              (int(x_center - half_width), int(y_center - half_height)),
                              (int(x_center + half_width), int(y_center + half_height)),
                              (0, 255, 255), -1)
                count_text = f"COUNT: {int(count)}"
                cv2.putText(frame, count_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
            rendered_frame = results.render()[0]
            (flag, encodedImage) = cv2.imencode(".jpg", rendered_frame)
            if not flag:
                continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')
    
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
