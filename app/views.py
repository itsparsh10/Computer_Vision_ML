"""
Views for handling requests in the Video_Txt Django application.
"""

from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import os
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
import subprocess
import parselmouth
from django.conf import settings
from django.core.files.storage import default_storage
from deepface import DeepFace
import mediapipe as mp

from .models import Upload, Transcript, Analysis
from .transcriber import VideoTranscriber, ffmpeg_available
from .gemini import GeminiProcessor
from .text_analyzer import TextAnalyzer
import re
from collections import Counter

# Initialize the components
transcriber = VideoTranscriber()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    gemini_processor = GeminiProcessor(GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found in .env file!")
    gemini_processor = None
text_analyzer = TextAnalyzer()

# ffmpeg_available is already imported from transcriber.py, no need to check again

# === Drawing Utils ===
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to detect repeated words in transcript
def detect_repeated_words(transcript, min_length=4, min_occurrences=3, max_results=10):
    """Detect words that are frequently repeated in the transcript"""
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', transcript.lower())
    
    # Split into words
    words = [word for word in text.split() if len(word) >= min_length]
    
    # Count occurrences
    word_counts = Counter(words)
    
    # Expanded list of common words to filter out
    common_words = [
        'this', 'that', 'with', 'have', 'from', 'they', 'will', 'would', 
        'there', 'their', 'what', 'when', 'were', 'your', 'which', 'about', 
        'then', 'some', 'these', 'those', 'been', 'being', 'very', 'just',
        'because', 'where', 'through', 'should', 'could', 'other', 'after',
        'before', 'while', 'than', 'them', 'into', 'over', 'under', 'here'
    ]
    
    # Filter out common words and keep only those that appear frequently
    repeated = [
        {'word': word, 'count': count} 
        for word, count in word_counts.most_common(30) 
        if count >= min_occurrences and word not in common_words
    ]
    
    # Sort by frequency (highest to lowest)
    repeated.sort(key=lambda x: x['count'], reverse=True)
    
    return repeated[:max_results]  # Return limited number of results

# Function to detect filler words in transcript
def detect_filler_words(transcript, max_results=10):
    """Detect filler words in the transcript"""
    # Comprehensive list of common filler words and phrases
    filler_words = [
        # Basic fillers
        'um', 'uh', 'ah', 'er', 'eh', 'hmm', 'mmm', 'huh', 
        'uhh', 'umm', 'mm', 'mhm', 'erm',
        
        # Phrase fillers
        'you know', 'i mean', 'like', 'sort of', 'kind of', 
        'basically', 'literally', 'actually', 'obviously',
        
        # Conversational fillers
        'right', 'okay', 'so', 'well', 'just', 'stuff', 'things',
        'yeah', 'you see', 'see what i mean', 'if you will',
        
        # Hesitation indicators
        'anyway', 'anyhow', 'whatever', 'and stuff', 'and things'
    ]
    
    # Convert to lowercase and ensure proper word boundaries
    text = transcript.lower()
    
    # Count filler words with proper word boundary detection
    filler_counts = []
    for filler in filler_words:
        # Create regex pattern with word boundaries
        pattern = r'\b' + re.escape(filler) + r'\b'
        
        # Find all occurrences
        matches = re.findall(pattern, text)
        count = len(matches)
        
        # Only include if found in transcript
        if count > 0:
            filler_counts.append({
                'word': filler, 
                'count': count,
                # Calculate percentage of total words for context
                'percentage': round((count / len(text.split())) * 100, 1)
            })
    
    # Sort by frequency (most frequent first)
    filler_counts.sort(key=lambda x: x['count'], reverse=True)
    
    return filler_counts[:max_results]  # Return limited number of results

@csrf_exempt
def pose_voice_analysis(request):
    """
    Analyze pose and voice from uploaded video
    """
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

        # Counters
        smile_count = head_move_count = hand_move_count = 0
        eye_contact_count = leg_move_count = foot_move_count = 0
        prev_lw = prev_rw = prev_la = prev_ra = prev_lk = prev_rk = None
        initial_nose_x = initial_nose_y = None

        # Init MediaPipe
        mp_pose = mp.solutions.pose
        mp_face = mp.solutions.face_mesh
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
        face_mesh = mp_face.FaceMesh(refine_landmarks=True)

        def detect_pose(frame):
            nonlocal smile_count, head_move_count, hand_move_count
            nonlocal eye_contact_count, leg_move_count, foot_move_count
            nonlocal prev_lw, prev_rw, prev_la, prev_ra, prev_lk, prev_rk
            nonlocal initial_nose_x, initial_nose_y

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb)
            face_results = face_mesh.process(rgb)

            # Body landmarks
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                try:
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                    la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                    rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                except IndexError:
                    return

                # Head movement
                if nose.visibility > 0.5:
                    if initial_nose_x is None:
                        initial_nose_x, initial_nose_y = nose.x, nose.y
                    dx, dy = nose.x - initial_nose_x, initial_nose_y - nose.y
                    if abs(dx) > 0.02 or abs(dy) > 0.02:
                        head_move_count += 1

                # Hand movement
                if lw.visibility > 0.5 and rw.visibility > 0.5:
                    if prev_lw and prev_rw:
                        if abs(lw.x - prev_lw.x) + abs(lw.y - prev_lw.y) > 0.02 or \
                           abs(rw.x - prev_rw.x) + abs(rw.y - prev_rw.y) > 0.02:
                            hand_move_count += 1
                    prev_lw, prev_rw = lw, rw

                # Leg movement
                if lk.visibility > 0.5 and rk.visibility > 0.5:
                    if prev_lk and prev_rk:
                        if abs(lk.x - prev_lk.x) + abs(lk.y - prev_lk.y) > 0.02 or \
                           abs(rk.x - prev_rk.x) + abs(rk.y - prev_rk.y) > 0.02:
                            leg_move_count += 1
                    prev_lk, prev_rk = lk, rk

                # Foot movement
                if la.visibility > 0.5 and ra.visibility > 0.5:
                    if prev_la and prev_ra:
                        if abs(la.x - prev_la.x) + abs(la.y - prev_la.y) > 0.02 or \
                           abs(ra.x - prev_ra.x) + abs(ra.y - prev_ra.y) > 0.02:
                            foot_move_count += 1
                    prev_la, prev_ra = la, ra

            # Face analysis
            if face_results.multi_face_landmarks:
                try:
                    result = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)
                    if isinstance(result, list) and result[0]['dominant_emotion'].lower() == "happy":
                        smile_count += 1
                except Exception:
                    pass

                # Eye contact
                landmarks = face_results.multi_face_landmarks[0].landmark
                left_eye = landmarks[33]
                right_eye = landmarks[263]
                nose_tip = landmarks[1]
                eye_center_x = (left_eye.x + right_eye.x) / 2
                if abs(nose_tip.x - eye_center_x) < 0.02:
                    eye_contact_count += 1

        # Process video frames and save each frame as an image in output_folder
        success, frame = cap.read()
        while success:
            if frame_index % fps == 0:  # Every 1 second
                detect_pose(frame)
                total_frames += 1
                # Save the frame as an image file in output_folder
                frame_filename = f"frame_{frame_index:05d}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
            success, frame = cap.read()
            frame_index += 1
        cap.release()

        # Extract audio using ffmpeg
        audio_path = os.path.join(audio_folder, 'audio.wav')
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path, '-vn',
            '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path
        ], check=True)

        # Audio analysis
        snd = parselmouth.Sound(audio_path)
        duration = snd.get_total_duration()
        intensity = snd.to_intensity()
        mean_volume = intensity.values.T.mean()

        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        mean_pitch = float(np.mean(pitch_values)) if len(pitch_values) else 0
        min_pitch = float(np.min(pitch_values)) if len(pitch_values) else 0
        max_pitch = float(np.max(pitch_values)) if len(pitch_values) else 0

        silence_flags = intensity.values[0] < 50
        frame_duration = intensity.get_time_step()
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

        def percent(val, total):
            return round((val / total * 100), 2) if total > 0 else 0

        return JsonResponse({
            "frames_processed": total_frames,
            "smiles": f"{smile_count} ({percent(smile_count, total_frames)}%)",
            "head_moves": f"{head_move_count} ({percent(head_move_count, total_frames)}%)",
            "hand_moves": f"{hand_move_count} ({percent(hand_move_count, total_frames)}%)",
            "eye_contact": f"{eye_contact_count} ({percent(eye_contact_count, total_frames)}%)",
            "leg_moves": f"{leg_move_count} ({percent(leg_move_count, total_frames)}%)",
            "foot_moves": f"{foot_move_count} ({percent(foot_move_count, total_frames)}%)",
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

@csrf_exempt  # Note: In production, use proper CSRF protection
@require_POST
def upload_video(request):
    """
    Django view to handle video upload and processing
    """
    
    if not gemini_processor:
        return JsonResponse({'error': 'Gemini API key not configured'}, status=500)
    
    # Check if ffmpeg is available
    if not ffmpeg_available:
        return JsonResponse({'error': 'ffmpeg not found. Please install ffmpeg and make sure it is in your system PATH.'}, status=500)
    
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    file = request.FILES['file']
    
    # Check if file is a video/audio
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wav', '.m4a'}
    file_extension = Path(file.name).suffix.lower()
    
    if file_extension not in allowed_extensions:
        return JsonResponse(
            {'error': f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"},
            status=400
        )
    
    # Save uploaded file temporarily
    temp_file_path = None
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            for chunk in file.chunks():
                temp_file.write(chunk)
        
        print(f"Processing file: {file.name}")
        
        # STEP 1: Transcribe video (not generating SRT for web version)
        transcription_result = transcriber.transcribe_video(
            video_path=temp_file_path, 
            generate_srt=False,
            verbose=False
        )
        
        if not transcription_result["success"]:
            return JsonResponse({'error': 'Transcription failed'}, status=500)
        
        transcript = transcription_result["transcript"]
        
        if not transcript or len(transcript.strip()) < 10:
            return JsonResponse({'error': 'No speech detected in the file'}, status=400)
        
        # STEP 2: Process with Gemini
        # Use a random factor between 0.4 and 0.8 to get varied grammar corrections each time
        import random
        random_factor = 0.4 + (random.random() * 0.4)
        gemini_result = gemini_processor.process_transcript(transcript, randomness_factor=random_factor)
        
        # STEP 3: Perform HuggingFace Transformers analysis
        print("Performing advanced text analysis with HuggingFace Transformers...")
        text_analysis = text_analyzer.analyze_transcript(transcript)
        
        # STEP 4: Save to database
        upload = Upload.objects.create(
            file=file,
            filename=file.name,
            file_size=file.size / (1024*1024)  # Convert to MB
        )
        
        transcript_obj = Transcript.objects.create(
            upload=upload,
            raw_transcript=transcript,
            corrected_transcript=gemini_result.get("corrected_text", transcript),
            language=transcription_result.get("language", "unknown"),
            duration_seconds=transcription_result.get("duration_seconds", 0),
            duration_formatted=transcription_result.get("duration_formatted", "00:00:00")
        )
        
        # Format keywords as a proper list if they're in comma-separated format
        keywords = gemini_result.get("keywords", "Keywords not available")
        
        # Detect repeated words (excluding common words)
        repeated_words = detect_repeated_words(transcript)
        
        # Detect filler words
        filler_words = detect_filler_words(transcript)
        
        analysis_obj = Analysis.objects.create(
            transcript=transcript_obj,
            summary=gemini_result.get("summary", "Summary not available"),
            keywords=keywords,
            sentiment_analysis=text_analysis.get("sentiment_analysis", {}),
            content_assessment=text_analysis.get("content_assessment", {}),
            strengths_improvements=text_analysis.get("strengths_improvements", {}),
            emotion_analysis=text_analysis.get("emotion_analysis", {}),
            repeated_words=repeated_words,
            filler_words=filler_words
        )
        
        # STEP 5: Return results with proper formatting
        response_data = {
            "filename": file.name,
            "file_size": f"{file.size / (1024*1024):.2f} MB",
            "language": transcription_result.get("language", "unknown"),
            "duration": transcription_result.get("duration_formatted", "00:00:00"),
            "original_transcript": transcript,
            "corrected_transcript": gemini_result.get("corrected_text", transcript),
            "summary": gemini_result.get("summary", "Summary not available"),
            "keywords": keywords,
            "repeated_words": repeated_words,
            "filler_words": filler_words,
            # Add HuggingFace Transformers analysis results
            "sentiment_analysis": text_analysis.get("sentiment_analysis", {}),
            "content_assessment": text_analysis.get("content_assessment", {}),
            "strengths_improvements": text_analysis.get("strengths_improvements", {}),
            "emotion_analysis": text_analysis.get("emotion_analysis", {})
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {str(e)}")
        print(f"Detailed error: {error_details}")
        return JsonResponse({'error': f"Processing failed: {str(e)}"}, status=500)
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def health_check(request):
    """
    Simple health check endpoint to verify the API is working
    """
    status = {
        "status": "ok",
        "message": "API is running",
        "ffmpeg_available": ffmpeg_available,
        "gemini_api": "configured" if gemini_processor else "not configured"
    }
    return JsonResponse(status)

@csrf_exempt
@require_POST
def generate_coach_feedback(request):
    """
    Generate presentation coaching feedback using Gemini API based on pose and voice analysis data.
    """
    try:
        import json
        data = json.loads(request.body)
        
        # Check if GEMINI_API_KEY is available
        if not GEMINI_API_KEY:
            return JsonResponse({
                'success': False,
                'error': 'Gemini API key not configured'
            }, status=500)
        
        # Initialize the Gemini processor
        gemini_processor = GeminiProcessor(GEMINI_API_KEY)
        
        # Extract metrics from request data
        metrics = data.get('metrics', {})
        
        # Extract pose and voice data
        smile_data = metrics.get('smiles', '0 (0%)')
        head_moves_data = metrics.get('head_moves', '0 (0%)')
        hand_moves_data = metrics.get('hand_moves', '0 (0%)')
        eye_contact_data = metrics.get('eye_contact', '0 (0%)')
        leg_moves_data = metrics.get('leg_moves', '0 (0%)')
        foot_moves_data = metrics.get('foot_moves', '0 (0%)')
        
        # Extract numbers and percentages
        def extract_counts_and_percentages(data_str):
            match = re.search(r'(\d+)\s*\((\d+(?:\.\d+)?)%\)', data_str)
            if match:
                return match.group(1), match.group(2)
            return "0", "0"
        
        smile_count, smile_percentage = extract_counts_and_percentages(smile_data)
        head_count, head_percentage = extract_counts_and_percentages(head_moves_data)
        hand_count, hand_percentage = extract_counts_and_percentages(hand_moves_data)
        eye_contact_count, eye_contact_percentage = extract_counts_and_percentages(eye_contact_data)
        leg_count, leg_percentage = extract_counts_and_percentages(leg_moves_data)
        foot_count, foot_percentage = extract_counts_and_percentages(foot_moves_data)
        
        # Extract audio data
        audio_data = metrics.get('audio', {})
        total_duration = audio_data.get('duration_sec', 0)
        speaking_time = audio_data.get('spoken_duration_sec', 0)
        average_volume = audio_data.get('volume_db', 0)
        mean_pitch = audio_data.get('mean_pitch_hz', 0)
        pitch_range = audio_data.get('pitch_range', '0-0 Hz')
        pitch_range_min, pitch_range_max = "0", "0"
        
        # Extract pitch range values
        pitch_match = re.search(r'(\d+)–(\d+) Hz', str(pitch_range))
        if pitch_match:
            pitch_range_min, pitch_range_max = pitch_match.group(1), pitch_match.group(2)
        
        pause_count = audio_data.get('num_pauses', 0)
        
        # Create the prompt for Gemini API
        prompt = f"""
You are an expert presentation coach specializing in comprehensive communication analysis for educational platforms. Your expertise combines body language interpretation, vocal delivery assessment, and actionable feedback generation to help students develop confident, engaging presentation skills.

### ANALYSIS INSTRUCTIONS:
Analyze the provided data and generate feedback using the following four clearly defined sections with exact HTML formatting:

---

<div class="summary">
<h3>🎯 Overall Assessment</h3>
<p>
Provide a balanced overview of the presenter's communication effectiveness in 2-3 sentences. Highlight their strongest asset (body language OR vocal delivery) and identify the single most important area for improvement. Be encouraging while being honest about performance.
</p>
</div>

---

<div class="body-analysis">
<ul>
<li><strong>Posture & Positioning:</strong> Analyze overall movement patterns based on leg/foot movement data. Assess if the presenter appears confident and grounded or restless and nervous.</li>
<li><strong>Facial Engagement:</strong> Evaluate smile frequency and eye contact patterns. Determine if the presenter connects warmly with the audience or appears distant/uncomfortable.</li>
<li><strong>Gesture Effectiveness:</strong> Assess hand and head movement data for purposeful communication. Identify if gestures enhance the message or appear distracting/excessive.</li>
<li><strong>Confidence Indicators:</strong> Synthesize all body language data to identify overall confidence level and any nervous habits that may detract from the message.</li>
</ul>
</div>

---

<div class="vocal-analysis">
<ul>
<li><strong>Clarity & Audibility:</strong> Analyze volume levels for appropriate projection. Assess if the presenter can be clearly heard without being overwhelming.</li>
<li><strong>Vocal Variety:</strong> Evaluate pitch range for expressiveness and engagement. Determine if the voice maintains audience interest or becomes monotonous.</li>
<li><strong>Pacing & Flow:</strong> Analyze pause patterns and speaking rhythm. Calculate pauses per minute and assess if pacing allows for audience comprehension.</li>
<li><strong>Vocal Confidence:</strong> Assess overall vocal presence, authority, and comfort level based on all vocal metrics combined.</li>
</ul>
</div>

---

<div class="recommendations">
<h3>Priority Improvements</h3>
<ol>
<li><strong>Primary Focus (Body Language):</strong> Identify the most critical physical improvement with a specific, actionable technique the student can practice immediately.</li>
<li><strong>Primary Focus (Vocal Delivery):</strong> Identify the most critical vocal improvement with a specific, actionable technique the student can practice immediately.</li>
<li><strong>Quick Win:</strong> Suggest one easy change that will show immediate results and boost confidence for the next presentation.</li>
<li><strong>Practice Exercise:</strong> Provide a specific, practical exercise or activity that addresses the biggest weakness identified in the analysis.</li>
</ol>
</div>

---

### STUDENT PRESENTATION DATA:

**BODY LANGUAGE METRICS:**
- Smiles Detected: {smile_count} ({smile_percentage}%)
- Head Movements: {head_count} ({head_percentage}%)
- Hand Movements: {hand_count} ({hand_percentage}%)
- Eye Contact: {eye_contact_count} ({eye_contact_percentage}%)
- Leg Movements: {leg_count} ({leg_percentage}%)
- Foot Movements: {foot_count} ({foot_percentage}%)

**VOICE METRICS:**
- Total Duration: {total_duration} seconds
- Speaking Time: {speaking_time} seconds
- Average Volume: {average_volume} dB
- Mean Pitch: {mean_pitch} Hz
- Pitch Range: {pitch_range_min}–{pitch_range_max} Hz
- Number of Pauses: {pause_count}

### QUALITY STANDARDS:
- Use specific data points and percentages in your analysis
- Every suggestion must be actionable and something the student can practice
- Acknowledge strengths before addressing weaknesses
- Frame improvements as opportunities for growth, not failures
- Focus on the 3-4 most impactful changes for maximum improvement
- Use encouraging, supportive coaching language throughout
- Provide equal depth for both body language and vocal delivery sections
- Include specific practice techniques and exercises
- End with an encouraging, forward-looking statement

Generate comprehensive, data-driven feedback that will help this student become a more confident and effective presenter.
"""

        # Call Gemini API
        response = gemini_processor.model.generate_content(prompt)
        generated_text = response.text
        
        # Parse the generated text to extract sections
        summary = ""
        interpretation = ""
        suggestions = ""
        
        # Extract content between <p> and </p> tags for the summary
        summary_match = re.search(r'<p>(.*?)</p>', generated_text, re.DOTALL)
        if summary_match:
            summary = f"<p>{summary_match.group(1)}</p>"
        
        # Extract content between <ul> and </ul> tags for the interpretation
        interpretation_match = re.search(r'<ul>(.*?)</ul>', generated_text, re.DOTALL)
        if interpretation_match:
            interpretation = f"<ul>{interpretation_match.group(1)}</ul>"
        
        # Extract content between <ol> and </ol> tags for the suggestions
        suggestions_match = re.search(r'<ol>(.*?)</ol>', generated_text, re.DOTALL)
        if suggestions_match:
            suggestions = f"<ol>{suggestions_match.group(1)}</ol>"
        
        # If any section couldn't be extracted properly, try simpler parsing
        if not summary or not interpretation or not suggestions:
            parts = generated_text.split("\n\n")
            if len(parts) >= 3:
                # Basic fallback parsing
                summary = f"<p>{parts[0].strip()}</p>"
                
                # Try to extract bullet points for interpretation
                interp_part = parts[1].strip()
                interp_items = re.findall(r'- (.*?)(?:\n|$)', interp_part)
                if interp_items:
                    interpretation = "<ul>" + "".join([f"<li>{item.strip()}</li>" for item in interp_items]) + "</ul>"
                else:
                    interpretation = f"<ul><li>{interp_part}</li></ul>"
                
                # Try to extract numbered items for suggestions
                sugg_part = parts[2].strip()
                sugg_items = re.findall(r'\d+\.\s+(.*?)(?:\n|$)', sugg_part)
                if sugg_items:
                    suggestions = "<ol>" + "".join([f"<li>{item.strip()}</li>" for item in sugg_items]) + "</ol>"
                else:
                    suggestions = f"<ol><li>{sugg_part}</li></ol>"
        
        return JsonResponse({
            'success': True,
            'feedback': {
                'summary': summary,
                'interpretation': interpretation,
                'suggestions': suggestions,
                'raw_text': generated_text  # Including the raw text for debugging
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
