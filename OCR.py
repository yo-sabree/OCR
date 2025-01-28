import keras_ocr
import cv2
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore

pipeline = keras_ocr.pipeline.Pipeline()

genai.configure(api_key="AIzaSyDKjjl8YuWmW07ZIAOLIEqrsJofP9cww5w")

cred = credentials.Certificate("D:/vat-demo-f206b-firebase-adminsdk-fbsvc-e8d3543a25.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prediction_groups = pipeline.recognize([frame_rgb])

    for predictions in prediction_groups[0]:
        box = predictions[1]
        cv2.polylines(frame, [box.astype(int)], True, (0, 255, 0), 2)
        text = predictions[0]
        position = tuple(box[0].astype(int))
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Live OCR', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        extracted_text = " ".join([predictions[0] for predictions in prediction_groups[0]])
        input_text = f"Extracted Text:\n{extracted_text}"
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            [f"Beautify the following text. Organize it with bullet points and headers where appropriate using only given text:\n{input_text}"],
            stream=True
        )
        response.resolve()
        beautified_text = response.text
        doc_ref = db.collection("ocr_texts").document()
        doc_ref.set({"text": beautified_text})
        break

cap.release()
cv2.destroyAllWindows()
