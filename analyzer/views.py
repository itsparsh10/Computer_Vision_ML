import os
import cv2
import numpy as np
import subprocess
import parselmouth
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from deepface import DeepFace
import mediapipe as mp


@csrf_exempt
def home(request):
    return render(request, 'analyzer/upload.html')


@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        video_name = default_storage.save(f'media/{video_file.name}', video_file)
        video_path = os.path.join(settings.MEDIA_ROOT, video_name)

        # Ensure output folders exist
        output_folder = os.path.join(settings.MEDIA_ROOT, 'output_frames')
        audio_folder = os.path.join(settings.MEDIA_ROOT, 'output_audio')
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return JsonResponse({'error': 'Could not open video file.'}, status=400)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_index = 0
        total_frames = 0

        # === Drawing Utils ===
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # === State Tracking ===
        prev_lw = prev_rw = prev_la = prev_ra = prev_lk = prev_rk = None
        initial_nose_x = initial_nose_y = None
        smile_count = head_move_count = hand_move_count = eye_contact_count = 0
        leg_move_count = foot_move_count = 0

        # === Draw Text Helper ===
        def draw_text(image, text, position, color=(255, 255, 255)):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, position, font, 0.6, color, 2)

        # === Pose + Emotion Detection ===
        def detect_pose(image):
            nonlocal prev_lw, prev_rw, prev_la, prev_ra, prev_lk, prev_rk
            nonlocal smile_count, head_move_count, hand_move_count, eye_contact_count
            nonlocal leg_move_count, foot_move_count, initial_nose_x, initial_nose_y

            with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                        enable_segmentation=False, min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as pose, \
                 mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                 refine_landmarks=True, min_detection_confidence=0.5,
                                                 min_tracking_confidence=0.5) as face_mesh:

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb)
                face_results = face_mesh.process(rgb)

                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, pose_results.pose_landmarks,
                                              mp.solutions.pose.POSE_CONNECTIONS,
                                              mp_drawing_styles.get_default_pose_landmarks_style())

                    landmarks = pose_results.pose_landmarks.landmark
                    nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE]
                    lw = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
                    rw = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
                    la = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
                    ra = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
                    lk = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
                    rk = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]

                    # Head Movement
                    if nose.visibility > 0.5:
                        if initial_nose_x is None or initial_nose_y is None:
                            initial_nose_x, initial_nose_y = nose.x, nose.y
                        dx, dy = nose.x - initial_nose_x, initial_nose_y - nose.y
                        movement_msg = []
                        if abs(dx) > 0.02:
                            movement_msg.append("➡️ Right" if dx > 0 else "⬅️ Left")
                        if abs(dy) > 0.02:
                            movement_msg.append("⬆️ Up" if dy > 0 else "⬇️ Down")
                        if movement_msg:
                            head_move_count += 1
                            draw_text(image, "Head: " + " ".join(movement_msg), (10, 100), (0, 255, 255))

                    # Hand Movement
                    if lw.visibility > 0.5 and rw.visibility > 0.5:
                        if prev_lw and prev_rw:
                            if abs(lw.x - prev_lw.x) + abs(lw.y - prev_lw.y) > 0.02 or \
                               abs(rw.x - prev_rw.x) + abs(rw.y - prev_rw.y) > 0.02:
                                hand_move_count += 1
                                draw_text(image, " Hand Movement", (10, 180), (100, 255, 255))
                        prev_lw, prev_rw = lw, rw

                    # Leg Movement
                    if lk.visibility > 0.5 and rk.visibility > 0.5:
                        if prev_lk and prev_rk:
                            if abs(lk.x - prev_lk.x) + abs(lk.y - prev_lk.y) > 0.02 or \
                               abs(rk.x - prev_rk.x) + abs(rk.y - prev_rk.y) > 0.02:
                                leg_move_count += 1
                                draw_text(image, " Leg Movement", (10, 300), (0, 255, 200))
                        prev_lk, prev_rk = lk, rk

                    # Foot Movement
                    if la.visibility > 0.5 and ra.visibility > 0.5:
                        if prev_la and prev_ra:
                            if abs(la.x - prev_la.x) + abs(la.y - prev_la.y) > 0.02 or \
                               abs(ra.x - prev_ra.x) + abs(ra.y - prev_ra.y) > 0.02:
                                foot_move_count += 1
                                draw_text(image, " Foot Movement", (10, 330), (0, 200, 255))
                        prev_la, prev_ra = la, ra

                # Face Detection
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]
                    nose_tip = face_landmarks.landmark[1]
                    eye_center_x = (left_eye.x + right_eye.x) / 2
                    nose_offset = abs(nose_tip.x - eye_center_x)

                    if nose_offset < 0.02:
                        eye_contact_count += 1
                        draw_text(image, " Eye Contact", (10, 260), (255, 255, 0))

                    # Smile Detection
                    try:
                        result = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)
                        if isinstance(result, list) and result[0]['dominant_emotion'].lower() == "happy":
                            smile_count += 1
                            draw_text(image, " Smile Detected", (10, 220), (255, 100, 200))
                    except Exception as e:
                        print(f"DeepFace error: {e}")

                return image

        # === Process Each Frame ===
        print("Running Pose + Emotion Detection...")
        success, frame = cap.read()
        while success:
            if frame_index % fps == 0:  # Every 1 second
                seconds = frame_index // fps
                processed = detect_pose(frame.copy())
                draw_text(processed, f"Time: {seconds}s", (10, 30), (0, 255, 0))
                cv2.imwrite(os.path.join(output_folder, f"frame_{seconds}s.jpg"), processed)
                print(f"Saved frame: {seconds}s")
                total_frames += 1
            success, frame = cap.read()
            frame_index += 1
        cap.release()

        # === Audio Extraction ===
        audio_output_path = os.path.join(audio_folder, "extracted_audio.wav")
        print("\n🎧 Extracting Audio...")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-vn",
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                audio_output_path
            ], check=True)
            print(f"Audio saved: {audio_output_path}")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e}")

        # === Audio Analysis ===
        print("\nExtracting Audio Features...")
        try:
            snd = parselmouth.Sound(audio_output_path)
            duration = snd.get_total_duration()
            intensity = snd.to_intensity()
            mean_volume = intensity.values.T.mean()

            pitch = snd.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]
            mean_pitch = np.mean(pitch_values) if len(pitch_values) else 0
            min_pitch = np.min(pitch_values) if len(pitch_values) else 0
            max_pitch = np.max(pitch_values) if len(pitch_values) else 0

            silence_threshold = 50
            frame_duration = intensity.get_time_step()
            silence_flags = intensity.values[0] < silence_threshold

            num_pauses = 0
            spoken_duration = 0.0
            in_pause = False

            for silent in silence_flags:
                if silent:
                    if not in_pause:
                        num_pauses += 1
                        in_pause = True
                else:
                    spoken_duration += frame_duration
                    in_pause = False

            # === Print Audio Feature Summary ===
            print("\n=== AUDIO FEATURES ===")
            print(f"Duration: {duration:.2f} sec")
            print(f"Volume (avg dB): {mean_volume:.2f}")
            print(f"Frequency (avg): {mean_pitch:.2f} Hz")
            print(f"Pitch Range: {min_pitch:.2f} – {max_pitch:.2f} Hz")
            print(f"Pitch Range Average: {(min_pitch + max_pitch) / 2:.2f} Hz")
            print(f"Number of Pauses: {num_pauses}")
            print(f"Spoken Duration (No silence): {spoken_duration:.2f} sec")

        except Exception as e:
            print(f"Audio processing failed: {e}")

        # === Summary ===
        def percent(value, total):
            return (value / total * 100) if total > 0 else 0

        print("\n=== FINAL SUMMARY ===")
        print(f"Total Frames Processed : {total_frames}")
        print(f"Total Smiles Detected : {smile_count} ({percent(smile_count, total_frames):.2f}%)")
        print(f"Total Head Moves  : {head_move_count} ({percent(head_move_count, total_frames):.2f}%)")
        print(f"Total Hand Moves : {head_move_count} ({percent(hand_move_count, total_frames):.2f}%)")
        print(f"Total Eye Contact  : {eye_contact_count} ({percent(eye_contact_count, total_frames):.2f}%)")
        print(f"Total Leg Moves  : {leg_move_count} ({percent(leg_move_count, total_frames):.2f}%)")
        print(f"Total Foot Moves  : {foot_move_count} ({percent(foot_move_count, total_frames):.2f}%)")

        return JsonResponse({
            "frames_processed": total_frames,
            "smiles": f"{smile_count} ({percent(smile_count, total_frames):.2f}%)",
            "head_moves": f"{head_move_count} ({percent(head_move_count, total_frames):.2f}%)",
            "hand_moves": f"{hand_move_count} ({percent(hand_move_count, total_frames):.2f}%)",
            "eye_contact": f"{eye_contact_count} ({percent(eye_contact_count, total_frames):.2f}%)",
            "leg_moves": f"{leg_move_count} ({percent(leg_move_count, total_frames):.2f}%)",
            "foot_moves": f"{foot_move_count} ({percent(foot_move_count, total_frames):.2f}%)",
            "audio": {
                "duration_sec": round(duration, 2),
                "volume_db": round(mean_volume, 2),
                "mean_pitch_hz": round(mean_pitch, 2),
                "pitch_range": f"{round(min_pitch)}–{round(max_pitch)} Hz",
                "avg_pitch_range": round((min_pitch + max_pitch) / 2, 2) if pitch_values.size else 0,
                "num_pauses": num_pauses,
                "spoken_duration_sec": round(spoken_duration, 2)
            }
        })

    return JsonResponse({'error': 'Invalid request'}, status=400)
