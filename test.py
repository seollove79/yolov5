from fastapi import FastAPI
from starlette.responses import StreamingResponse
import cv2
import torch
import uvicorn
from pathlib import Path

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

    # 추가적인 전처리를 필요에 따라 추가...

    return frame

@app.get("/video/")
async def analyze_video():
    # 파일을 직접 cv2.VideoCapture로 읽기
    cap = cv2.VideoCapture('tomato.mp4')
    #cap = cv2.VideoCapture('http://59.15.133.179:24500/api/downloadfile/20230821140637-11video-4278772458.mp4')
        

    # 비디오를 처리하고 결과를 반환하는 함수를 생성
    def gen_frames():
        count = 0
        while True:
            success, frame = cap.read()  # frame per frame 읽기
            if not success:
                break

            # 프레임 전처리
            frame = preprocess_frame(frame)

            # 사용자 정의 객체 검출 실행
            results = model(frame)

            text_position = (frame.shape[1] - 300, frame.shape[0] - 30)  # 오른쪽 하단 위치
            
            # 인식된 객체의 중심점 계산 및 노란색 사각형 그리기
            for det in results.xyxy[0].cpu().numpy():
                x_min, y_min, x_max, y_max, _, _ = det
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                print(f"x_center : {x_center}")

                if x_center>=930 and x_center<=970:
                    count = count + 1

                half_width = 5
                half_height = 5

                cv2.rectangle(frame, 
                            (int(x_center - half_width), int(y_center - half_height)),
                            (int(x_center + half_width), int(y_center + half_height)),
                            (0, 255, 255), # BGR for yellow
                            -1)  # -1 fills the rectangle
                
                count_text = f"COUNT: {int(count)}"
                cv2.putText(frame, count_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)

            rendered_frame = results.render()[0]  # 객체를 표시한 이미지를 얻습니다.

            (flag, encodedImage) = cv2.imencode(".jpg", rendered_frame)
            if not flag:
                continue

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')  # 현재 프레임을 반환


    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/video1/")
async def analyze_video1():
    return "test"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
