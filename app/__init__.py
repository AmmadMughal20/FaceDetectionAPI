from flask import Flask, request, jsonify, Response
from FaceDetection import detect_faces
import cv2

app = Flask(__name__)


@app.route("/")
def home():

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files["image"]

    # Check if the file has a valid extension (you might want to add more checks)
    if not image_file or not image_file.filename.endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "Invalid image file"}), 400

    # Perform face detection on the image
    detected_image = detect_faces(image_file)
    
    # Convert the NumPy array back to an image format
    _, img_encoded = cv2.imencode(".jpg", detected_image)
    img_bytes = img_encoded.tobytes()

    # Set the content type as image/jpeg
    response = Response(img_bytes, content_type="image/jpeg")

    return response


if __name__ == "__main__":
    app.run(debug=True)
