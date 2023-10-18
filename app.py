import torch
import cv2

# YOLOv5 모델을 불러옵니다.
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', trust_repo=True)

def detect_video(model, video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Error: Video file could not be opened!')
        return

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # YOLOv5로 객체를 인식합니다.
        results = model(frame)  # 여기서 img_size 지정
        
        # 확률이 임계값보다 낮은 결과를 필터링합니다.
        results.pred = [x[x[..., 4] > threshold] for x in results.pred]

        # Rendered image를 받아옵니다.
        rendered_img = results.render()[0]

        # 이미지를 보여줍니다.
        cv2.imshow('YOLOv5 Object Detection', rendered_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 동영상 객체 인식을 시작합니다.
video_path = 'tomato.mp4'
detect_video(model, video_path, threshold=0.5)
