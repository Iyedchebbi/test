
from flask import Flask, Response
import cv2
from face_detection import detect_faces

app = Flask(__name__)

# Initialiser la capture vidéo
video_capture = cv2.VideoCapture(0)  # 0 pour la caméra par défaut

def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        # Détecter les visages dans le cadre
        faces = detect_faces(frame)

        # Dessiner des rectangles autour des visages
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Convertir l'image en JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
