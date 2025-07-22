import face_recognition
import cv2
import numpy as np
import os
from flask import Flask, render_template, request, redirect, send_file, jsonify, flash
from datetime import datetime
import csv
import threading

app = Flask(__name__)
app.secret_key = 'supersecretkey'

known_face_encodings = []
known_face_names = []

# Load known faces
def load_known_faces():
    known_face_encodings.clear()
    known_face_names.clear()
    for filename in os.listdir('known_faces'):
        if filename.endswith(('.jpg', '.png')):
            img = face_recognition.load_image_file(f'known_faces/{filename}')
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])

load_known_faces()

# Mark attendance
attendance_records = set()

def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    record = (name, dt_string)
    if record not in attendance_records:
        attendance_records.add(record)
        with open('attendance.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, dt_string])

def read_attendance():
    if not os.path.exists('attendance.csv'):
        return []
    with open('attendance.csv', 'r') as f:
        reader = csv.reader(f)
        return list(reader)

@app.route('/', methods=['GET', 'POST'])
def index():
    recognized_names = []
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            img = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(img)
            face_encodings = face_recognition.face_encodings(img, face_locations)

            for i, face_encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        if name not in recognized_names:
                            recognized_names.append(name)
                            mark_attendance(name)
                    else:
                        unknown_name = f'Unknown_{datetime.now().strftime("%Y%m%d%H%M%S")}_{i}'
                        cv2.imwrite(f'known_faces/{unknown_name}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(unknown_name)
                        recognized_names.append(unknown_name)
                        mark_attendance(unknown_name)

    return render_template('index.html', recognized=recognized_names, attendance_list=read_attendance())

@app.route('/download')
def download_attendance():
    if os.path.exists('attendance.csv'):
        return send_file('attendance.csv', as_attachment=True)
    flash('No attendance records to download.')
    return redirect('/')

@app.route('/clear')
def clear_attendance():
    open('attendance.csv', 'w').close()
    attendance_records.clear()
    flash('Attendance log cleared successfully.')
    return redirect('/')

@app.route('/reload_known_faces')
def reload_known_faces():
    load_known_faces()
    flash('Known faces reloaded successfully.')
    return redirect('/')

@app.route('/api/attendance')
def api_attendance():
    return jsonify(read_attendance())

@app.route('/attendance_table')
def attendance_table():
    return render_template('attendance_table.html', attendance_list=read_attendance())

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
    app.run(debug=True)


