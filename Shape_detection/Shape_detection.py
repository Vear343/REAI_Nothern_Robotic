"""
โปรแกรมตรวจจับรูปทรงเรขาคณิต (สามเหลี่ยม / สี่เหลี่ยม / หกเหลี่ยม)
สำหรับหุ่นยนต์แข่งขันจัดส่งชิ้นงาน - ใช้กับ Raspberry Pi Camera หรือ Webcam

ใช้หลักการ: Threshold -> Find Contours -> Polygon Approximation -> นับจำนวนมุม

ติดตั้งไลบรารีที่ต้องใช้ (รันครั้งเดียวบน Raspberry Pi):
    pip install opencv-python numpy picamera2   # ถ้าใช้ Pi Camera
    หรือ
    pip install opencv-python numpy             # ถ้าใช้ USB Webcam
"""

import cv2
import numpy as np
import time

# ----------------------------
# ปรับค่าตรงนี้ให้เหมาะกับสภาพแสงจริงในสนามแข่งขัน
# ----------------------------
MIN_CONTOUR_AREA = 800          # พื้นที่ขั้นต่ำของ contour ที่จะพิจารณา (กรอง noise เล็กๆ ทิ้ง)
APPROX_EPSILON_RATIO = 0.02      # ค่าความละเอียดในการประมาณรูปหลายเหลี่ยม (ยิ่งน้อยยิ่งละเอียด)
STABLE_FRAMES_REQUIRED = 5       # ต้องอ่านรูปทรงเดิมติดกันกี่เฟรมถึงจะถือว่า "ยืนยันผล"


def preprocess(frame):
    """แปลงภาพเป็นภาพขาวดำและทำ threshold เพื่อแยกป้ายรหัส (สีน้ำเงิน) ออกจากพื้นขาว"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ป้ายรหัสสีน้ำเงินเข้มบนพื้นขาว -> ใช้ threshold แบบ inverse
    # ใช้ Otsu's method เพื่อหาค่า threshold อัตโนมัติ (ทนต่อแสงที่เปลี่ยนแปลง)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def classify_shape(contour):
    """จำแนกรูปทรงจากจำนวนมุมของ polygon ที่ประมาณได้"""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, APPROX_EPSILON_RATIO * perimeter, True)
    num_vertices = len(approx)

    if num_vertices == 3:
        return "triangle", approx
    elif num_vertices == 4:
        # ตรวจสอบเพิ่มเติมว่าเป็นสี่เหลี่ยมจริง ไม่ใช่รูปทรงอื่นที่ approx คลาดเคลื่อน
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.2:
            return "square", approx
        else:
            return "rectangle", approx
    elif num_vertices == 6:
        return "hexagon", approx
    else:
        return None, approx


def find_best_shape(frame):
    """หา contour ที่ใหญ่ที่สุดและน่าจะเป็นป้ายรหัส แล้วจำแนกรูปทรง"""
    thresh = preprocess(frame)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_shape = None
    best_contour = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        if area > best_area:
            shape_name, approx = classify_shape(cnt)
            if shape_name is not None:
                best_shape = shape_name
                best_contour = approx
                best_area = area

    return best_shape, best_contour


def shape_to_color(shape_name, color_mapping):
    """
    แปลงชื่อรูปทรงเป็นสีตามที่กรรมการประกาศในวันแข่งขัน
    ตัวอย่าง color_mapping = {"triangle": "red", "hexagon": "yellow", "square": "green"}
    ต้องแก้ dict นี้ให้ตรงกับที่กรรมการแจ้งในวันแข่งจริง (ข้อ 4.11 / คำอธิบายภารกิจข้อ 2)
    """
    return color_mapping.get(shape_name, None)


def detect_with_stability(camera_source=0, color_mapping=None, timeout_sec=5):
    """
    อ่านภาพจากกล้องต่อเนื่อง จนกว่าจะอ่านรูปทรงเดิมได้ติดกันครบ STABLE_FRAMES_REQUIRED เฟรม
    เพื่อป้องกันการอ่านผิดพลาดจากภาพสั่นหรือแสงกระพริบ
    คืนค่า: (shape_name, color) หรือ (None, None) ถ้า timeout
    """
    if color_mapping is None:
        color_mapping = {"triangle": "red", "hexagon": "yellow", "square": "green"}

    cap = cv2.VideoCapture(camera_source)
    # ลด resolution ลงเพื่อความเร็ว (สำคัญมากถ้าใช้ Pi 4 RAM 2GB)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_shape = None
    stable_count = 0
    start_time = time.time()

    try:
        while time.time() - start_time < timeout_sec:
            ret, frame = cap.read()
            if not ret:
                continue

            shape_name, contour = find_best_shape(frame)

            if shape_name == last_shape and shape_name is not None:
                stable_count += 1
            else:
                stable_count = 1
                last_shape = shape_name

            if stable_count >= STABLE_FRAMES_REQUIRED:
                color = shape_to_color(shape_name, color_mapping)
                return shape_name, color

        return None, None  # ไม่พบผลลัพธ์ที่เสถียรภายในเวลาที่กำหนด

    finally:
        cap.release()


# ----------------------------
# ตัวอย่างการใช้งานจริงในโปรแกรมหลักของหุ่นยนต์
# ----------------------------
if __name__ == "__main__":
    # กรรมการจะแจ้งในวันแข่งว่ารูปทรงไหนแทนสีอะไร ต้องแก้ dict นี้ในวันจริง
    COLOR_MAPPING = {
        "triangle": "red",
        "hexagon": "yellow",
        "square": "green",
    }

    print("กำลังอ่านป้ายคำสั่ง...")
    shape, color = detect_with_stability(camera_source=0, color_mapping=COLOR_MAPPING, timeout_sec=5)

    if shape:
        print(f"อ่านได้: รูปทรง = {shape}, สี = {color}")
        # TODO: ส่งคำสั่งไปยัง ESP32 เพื่อเปิดไฟ LED ตามสี (ผ่าน Serial/UART)
        # ตัวอย่าง: send_to_esp32(f"LED:{color}\n")
    else:
        print("ไม่สามารถอ่านป้ายคำสั่งได้ กรุณาตรวจสอบตำแหน่งกล้องและแสง")