from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import shutil

# ตรวจสอบและสร้างโฟลเดอร์ static หากยังไม่มี
if not os.path.exists("static"):
    os.makedirs("static")

# เส้นทางโมเดลที่เทรนไว้
model_path = './h8/snoring_detection_rnn_model.h5'

# โหลดโมเดลที่เทรนไว้แล้ว
model = load_model(model_path)

# ฟังก์ชันสำหรับโหลดและสกัดคุณลักษณะ (MFCC) จากไฟล์เสียง
def extract_features(audio, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# ฟังก์ชันสำหรับลบช่วงเงียบออกจากไฟล์เสียง
def remove_silence(audio, sr, top_db=20):
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
    non_silent_audio = [audio[start:end] for start, end in non_silent_intervals]
    return non_silent_audio, non_silent_intervals

# ฟังก์ชันสำหรับนับช่วงเสียงกรนและบันทึกเวลา
def count_snoring_segments(audio_file, top_db=20):
    # โหลดไฟล์เสียงทั้งหมด
    audio, sr = librosa.load(audio_file, sr=None)

    # ลบช่วงเงียบออกจากไฟล์เสียง
    non_silent_audio_segments, non_silent_intervals = remove_silence(audio, sr, top_db=top_db)

    # ตรวจสอบว่ามีช่วงเสียงที่ไม่เงียบหรือไม่ ถ้าไม่มีให้คืนค่า 0
    if not non_silent_audio_segments:
        print("No non-silent segments detected.")
        return 0, None

    snoring_intervals = []

    # วิเคราะห์แต่ละช่วงที่ไม่เงียบ
    for i, segment in enumerate(non_silent_audio_segments):
        # สกัดคุณลักษณะจาก segment ที่ไม่เงียบ
        features = extract_features(segment, sr)
        features = features.reshape((1, 1, features.shape[0]))

        # ทำนายผลลัพธ์
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction, axis=1)[0]

        # ถ้าเป็นเสียงกรน (predicted_label == 1)
        if predicted_label == 1:
            start_time = non_silent_intervals[i][0] / sr
            end_time = non_silent_intervals[i][1] / sr
            snoring_intervals.append((start_time, end_time))
            print(f"Snoring detected in non-silent segment {i + 1}")
        else:
            print(f"Non-snoring detected in non-silent segment {i + 1}")

    # สร้างกราฟเวลาแสดงช่วงที่กรน ถ้ามีการตรวจจับเสียงกรน
    if snoring_intervals:
        times = [interval[0] for interval in snoring_intervals]  # เวลาเริ่มต้นของแต่ละช่วงที่กรน
        plt.figure(figsize=(10, 4))
        plt.plot(times, [1] * len(times), 'ro')  # Plot เวลาที่กรนด้วยจุดสีแดง
        plt.xlabel("Time (seconds)")
        plt.yticks([])  # ซ่อนแกน y
        plt.title("Snoring Detection Timeline")
        
        # บันทึกกราฟเป็นไฟล์ภาพในโฟลเดอร์ static
        graph_path = "static/snoring_timeline.png"
        plt.savefig(graph_path)
        plt.close()
        
        print(f"Total snoring segments detected: {len(snoring_intervals)}")
        return len(snoring_intervals), graph_path  # คืนค่าผลการตรวจจับและที่อยู่ของไฟล์กราฟ
    else:
        print("No snoring detected.")
        return 0, None  # ไม่มีเสียงกรนที่ตรวจพบ

app = FastAPI()

# เพิ่ม StaticFiles สำหรับให้บริการไฟล์กราฟ
app.mount("/files", StaticFiles(directory="static"), name="files")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to your FastAPI application!"}

@app.post("/detect-snoring")
async def detect_snoring(file: UploadFile = File(...)):
    print(file.filename)
    temp_file_path = f"./temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # เรียกใช้ฟังก์ชันนับช่วงที่กรนและบันทึกเวลา
    result, graph_path = count_snoring_segments(temp_file_path)

    # ลบไฟล์ชั่วคราวหลังประมวลผลเสร็จ
    os.remove(temp_file_path)

    # ส่งผลลัพธ์กลับไปพร้อมที่อยู่ของกราฟ (ถ้ามี)
    response = {
        "message": "Snoring detection completed." if result > 0 else "No snoring detected.",
        "result": result,
    }
    if graph_path:
        response["graph_url"] = f"/files/{os.path.basename(graph_path)}"

    return response

