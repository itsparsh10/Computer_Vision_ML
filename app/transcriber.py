try:
    import whisper  # OpenAI's Whisper speech-to-text model
except ImportError:
    print("ERROR: The 'whisper' module is not installed. Please install it using 'pip install openai-whisper'.")
    sys.exit(1)
import os  # For file operations
import sys  # For command-line args
from pathlib import Path  # For file path handling
import subprocess  # For running ffprobe command
import json  # For parsing ffprobe output
import shutil  # For finding executable paths

# Check ffmpeg and set path if needed
def ensure_ffmpeg_available():
    """
    Ensures ffmpeg is available for Whisper to use.
    Returns True if ffmpeg is available, False if not.
    """
    # First check if ffmpeg is in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    
    if not ffmpeg_path:
        # Get LOCALAPPDATA path
        localappdata = os.environ.get('LOCALAPPDATA', '')
        
        # Common ffmpeg locations on Windows, including WinGet installations
        possible_paths = [
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe",
            os.path.join(localappdata, 'Microsoft\\WindowsApps\\ffmpeg.exe'),
            # WinGet installation paths
            os.path.join(localappdata, 'Microsoft\\WinGet\\Packages\\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-essentials_build\\bin\\ffmpeg.exe'),
            os.path.join(localappdata, 'Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe'),
        ]
        
        # Also search for any ffmpeg.exe in WinGet packages directory
        winget_packages_dir = os.path.join(localappdata, 'Microsoft\\WinGet\\Packages')
        if os.path.exists(winget_packages_dir):
            for root, dirs, files in os.walk(winget_packages_dir):
                if 'ffmpeg.exe' in files and 'Gyan.FFmpeg' in root:
                    possible_paths.append(os.path.join(root, 'ffmpeg.exe'))
        
        # Check each possible path
        for path in possible_paths:
            if os.path.isfile(path):
                ffmpeg_path = path
                # Add the directory to PATH so Whisper can find it
                ffmpeg_dir = os.path.dirname(path)
                if ffmpeg_dir not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
                print(f"Found ffmpeg at: {path}")
                
                # Also add ffprobe to PATH if it exists in the same directory
                ffprobe_path = os.path.join(ffmpeg_dir, 'ffprobe.exe')
                if os.path.isfile(ffprobe_path):
                    print(f"Found ffprobe at: {ffprobe_path}")
                
                break
    
    if ffmpeg_path:
        # Test if ffmpeg is working
        try:
            result = subprocess.run([ffmpeg_path, '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ ffmpeg is working correctly!")
                return True
            else:
                print(f"❌ ffmpeg found but not working: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error testing ffmpeg: {e}")
            return False
    else:
        print("ERROR: ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")
        print("You can install it using: winget install Gyan.FFmpeg.Essentials")
        return False

def format_time(seconds):
    """
    Convert seconds to SRT time format: HH:MM:SS,mmm
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def get_video_duration(video_path):
    """
    Get the duration of a video or audio file using ffprobe.
    
    Args:
        video_path (str): Path to the video or audio file
        
    Returns:
        dict with:
            - duration_seconds: Duration in seconds (float)
            - duration_formatted: Duration as HH:MM:SS string
            - success: Whether getting duration was successful
    """
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "json", 
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        duration_seconds = float(data["format"]["duration"])
        
        # Format duration as HH:MM:SS
        mins, secs = divmod(duration_seconds, 60)
        hours, mins = divmod(mins, 60)
        duration_formatted = f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}"
        
        return {
            "success": True,
            "duration_seconds": duration_seconds,
            "duration_formatted": duration_formatted
        }
    except Exception as e:
        return {
            "success": False,
            "duration_seconds": 0,
            "duration_formatted": "00:00:00",
            "error": str(e)
        }

class VideoTranscriber:
    """
    This class handles the transcription of video/audio files using OpenAI's Whisper.
    
    Whisper is a speech recognition model that converts speech in audio to text.
    It can also generate SRT subtitle files from the transcription.
    """
    
    def __init__(self, model_name="base"):
        """
        Initialize the transcriber by loading the Whisper model.
        
        Args:
            model_name (str): Whisper model size (tiny, base, small, medium, large)
                - tiny: fastest but less accurate
                - base: good balance of speed/accuracy (default)
                - medium: more accurate but slower
                - large: most accurate but requires more resources
        """
        self.model = whisper.load_model(model_name)
    
    def transcribe_video(self, video_path: str, generate_srt=False, verbose=False) -> dict:
        """
        Transcribe a video or audio file to text using Whisper.
        
        Args:
            video_path (str): Path to the video or audio file
            generate_srt (bool): Whether to generate an SRT subtitle file
            verbose (bool): Whether to show detailed output during transcription
            
        Returns:
            dict with:
                - success: Whether transcription was successful
                - transcript: The transcribed text
                - language: Detected language of the speech
                - duration_seconds: Duration of the video in seconds
                - duration_formatted: Duration in HH:MM:SS format
                - srt_path: Path to the SRT file (if generated)
                - error: Error message if failed
        """
        try:
            # Verify the file exists
            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "error": "File not found",
                    "transcript": ""
                }
            
            # Get video duration
            duration_info = get_video_duration(video_path)
            
            # Transcribe the video
            if verbose:
                print(f"Transcribing: {video_path}")
                
            result = self.model.transcribe(video_path, verbose=verbose)
            transcript = result["text"]
            
            response = {
                "success": True,
                "transcript": transcript,
                "language": result.get("language", "unknown"),
                "duration_seconds": duration_info.get("duration_seconds", 0),
                "duration_formatted": duration_info.get("duration_formatted", "00:00:00")
            }
            
            # Generate SRT file if requested
            if generate_srt:
                output_srt = Path(video_path).stem + ".srt"
                
                with open(output_srt, "w", encoding="utf-8") as f:
                    for i, segment in enumerate(result["segments"]):
                        start = segment["start"]
                        end = segment["end"]
                        text = segment["text"].strip()
                        f.write(f"{i+1}\n")
                        f.write(f"{format_time(start)} --> {format_time(end)}\n")
                        f.write(f"{text}\n\n")
                
                if verbose:
                    print(f"SRT file created: {output_srt}")
                response["srt_path"] = output_srt
            
            return response
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Transcription error: {str(e)}")
            print(f"Detailed transcription error: {error_details}")
            
            # Check for specific error conditions
            error_msg = str(e)
            if "CUDA" in error_msg or "GPU" in error_msg:
                print("GPU-related error detected. Make sure CUDA is properly configured or use CPU model.")
            elif "memory" in error_msg.lower():
                print("Memory error detected. The video may be too large or complex for current resources.")
            elif "ffmpeg" in error_msg.lower():
                print("FFmpeg error detected. Make sure ffmpeg is installed and accessible in PATH.")
            
            return {
                "success": False,
                "error": str(e),
                "transcript": "",
                "language": "unknown"
            }

# Check if ffmpeg is available
ffmpeg_available = ensure_ffmpeg_available()

# Function to make it easier to use as a standalone script
def transcribe_video_standalone(video_path, model_name="base", generate_srt=False, verbose=False):
    """
    Standalone function to transcribe a video file using Whisper
    
    Args:
        video_path (str): Path to the video file
        model_name (str): Whisper model size (tiny, base, small, medium, large)
        generate_srt (bool): Whether to generate an SRT subtitle file
        verbose (bool): Whether to print segment details during transcription
    
    Returns:
        str: The transcription text
    """
    transcriber = VideoTranscriber(model_name)
    result = transcriber.transcribe_video(video_path, generate_srt=generate_srt, verbose=False)
    
    if not result["success"]:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return None
    
    # Print the transcription
    print(f"Transcription completed: {len(result['transcript'])} characters")
    
    return result["transcript"]

def run_examples(video_path):
    """
    Run a basic transcription example
    
    Args:
        video_path (str): Path to the video file to use for example
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    
    print("\n===== AI VIDEO TRANSCRIPTION EXAMPLE =====\n")
    print(f"Video file: {video_path}")
    
    # Run basic transcription example
    print("\nRunning transcription with SRT generation...")
    transcribe_video_standalone(video_path, generate_srt=True)
    
    print("\n===== EXAMPLE COMPLETED =====")
    print("\nYou can also run the transcriber directly with:")
    print(f"python -m app.transcriber {video_path} [model_size] [--srt] [--verbose]")

# This allows the file to be used as both a module (import) and a standalone script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcriber.py video_file.mp4 [model_size] [--srt]")
        print("Model size options: tiny, base, small, medium, large (default: base)")
        print("Add --srt to generate subtitle file")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_name = "base"
    generate_srt = False
    
    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if arg == "--srt":
                generate_srt = True
            elif arg in ["tiny", "base", "small", "medium", "large"]:
                model_name = arg
    
    # Simple standalone usage
    transcribe_video_standalone(video_path, model_name, generate_srt)