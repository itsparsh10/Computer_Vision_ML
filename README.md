# AI Video Transcriber (Django Version)

This application transcribes video and audio files using OpenAI's Whisper and enhances the transcript using Google's Gemini AI and HuggingFace Transformers, all within a robust Django backend.

## Features

- **Video/Audio Transcription**: Utilizes OpenAI's Whisper for highly accurate speech-to-text conversion.
- **AI-Powered Analysis**:
    - **Summarization**: Generates concise summaries of transcripts with Google Gemini.
    - **Sentiment Analysis**: Analyzes the emotional tone using HuggingFace Transformers.
- **Web Interface**: A user-friendly interface for uploading files and viewing results.
- **Static & Media File Handling**: Properly serves the frontend and manages user uploads through Django.

## Tech Stack

- **Backend**: Django, Python
- **AI Models**: OpenAI Whisper, Google Gemini, HuggingFace Transformers
- **Frontend**: HTML, CSS, JavaScript

## Setup and Installation

For detailed instructions, please refer to `docs.txt`.

### 1. Prerequisites
- Python 3.9+
- FFmpeg

### 2. Clone the Repository
```bash
git clone <repository-url>
cd "Video_Txt 5"
```

### 3. Create a Virtual Environment
**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```
**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables
Create a `.env` file in the root directory and add your Gemini API key:
```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

## How to Run

### Method 1: Using the Custom Entry Point
Execute the main script to start the application:
```bash
python3 main.py
```

### Method 2: Using Django Management Command
Start the Django server directly:
```bash
python3 manage.py runserver
```

### Method 3: Using VS Code Task
If using VS Code, you can run the predefined task:
- Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
- Type "Tasks: Run Task"
- Select "Run Django Server"

The application will be available at **http://127.0.0.1:8000**.

## Usage

1.  Open your browser and navigate to the application URL.
2.  Use the form to upload a video or audio file.
3.  The system will transcribe the file and perform AI analysis.
4.  Results, including the transcript, summary, and sentiment, will be displayed on the page.

## Project Structure

```
Video_Txt 5/                     # Root project directory
│
├── Project Core Files           # Django project configuration
│   ├── manage.py               # Django management script
│   ├── settings.py             # Django settings and configuration
│   ├── urls.py                 # Main URL routing configuration
│   ├── wsgi.py                 # WSGI application entry point
│   ├── asgi.py                 # ASGI application entry point
│   └── main.py                 # Custom application entry point
│
├── app/                        # Django application directory
│   ├── __init__.py            # Python package marker
│   ├── views.py               # HTTP request handlers and API endpoints
│   ├── models.py              # Database models (Upload, Transcript, Analysis)
│   ├── urls.py                # App-specific URL routing
│   ├── transcriber.py         # Video transcription using Whisper
│   ├── gemini.py              # Google Gemini AI integration
│   ├── text_analyzer.py       # HuggingFace Transformers analysis
│   └── migrations/            # Database migration files
│       ├── __init__.py
│       └── 0001_initial.py    # Initial database schema
│
├── frontend/                   # Static frontend assets
│   ├── index.html             # Main interface template (reference)
│   ├── style.css              # Main application styling
│   ├── script.js              # Frontend JavaScript logic
│   ├── nav-styles.css         # Navigation styling
│   ├── transformers-styles.css # Analysis components styling
│   ├── visualizations.js      # Chart and graph logic
│   ├── scroll-indicator.js    # Scroll tracking functionality
│   └── section-history.js     # Navigation history management
│
├── templates/                  # Django templates
│   └── index.html             # Main interface (served by Django)
│
├── media/                      # User uploaded content
│   └── uploads/               # Video/audio files storage
│
├── uploads/                    # Legacy upload directory
├── outputs/                    # Processed results and SRT files
│
├── Configuration & Documentation
│   ├── .env                   # Environment variables (API keys)
│   ├── requirements.txt       # Python dependencies
│   ├── package-lock.json      # Node.js dependencies lock
│   ├── installed_packages.txt # Currently installed packages
│   ├── docs.txt              # Migration and technical documentation
│   ├── README.md             # Project documentation
│   └── test_emotion.py       # Emotion analysis testing script
│
├── Development Tools
│   ├── .vscode/              # VS Code configuration
│   │   └── tasks.json        # VS Code tasks (Run Django Server)
│   ├── __pycache__/          # Python bytecode cache
│   ├── venv/                 # Python virtual environment (if using venv)
│   ├── venv311/              # Python 3.11 virtual environment
│   └── db.sqlite3            # SQLite database file
```

## Backend Architecture and Processing Logic

This section explains the backend components and processing workflow in detail.

### Core Processing Pipeline

The AI Video Transcriber executes a multi-stage processing pipeline:

1. **Video/Audio Upload**: The frontend sends the media file to the Django backend.
2. **Transcription**: Whisper AI model converts speech to text.
3. **AI Enhancement**: The transcript is processed through Google's Gemini and HuggingFace Transformers.
4. **Result Composition**: All analysis is combined into a structured JSON response.
5. **Frontend Rendering**: The frontend displays the results in various sections.

### Key Components

#### 1. Video Transcriber (`app/transcriber.py`)

The `VideoTranscriber` class handles converting speech to text using OpenAI's Whisper:

- **Model Loading**: Initializes the Whisper model (tiny, base, small, medium, or large)
- **FFmpeg Integration**: Automatically detects and configures FFmpeg for audio processing
- **Transcription Process**: Processes video/audio files and extracts text
- **Features**:
  - Language detection
  - Duration calculation using ffprobe
  - Optional SRT subtitle file generation with timecodes
  - Error handling for common issues (CUDA, memory, ffmpeg)
  - Windows-specific FFmpeg path detection for WinGet installations

```python
# Example workflow
transcriber = VideoTranscriber(model_name="base")
result = transcriber.transcribe_video(video_path, generate_srt=False)
transcript = result["transcript"]
```

#### 2. Gemini Processor (`app/gemini.py`)

The `GeminiProcessor` class enhances the raw transcript using Google's Gemini AI:

- **Text Cleaning**: Pre-processes transcripts to remove artifacts and fix issues
- **Grammar Correction**: Improves readability and fixes grammatical errors with randomness control
- **Summarization**: Creates concise summaries of the content
- **Keyword Extraction**: Identifies key topics and themes
- **Error Handling**: Graceful fallback when API is unavailable

```python
# Example workflow
gemini_processor = GeminiProcessor(api_key)
gemini_result = gemini_processor.process_transcript(transcript, randomness_factor=0.6)
corrected_text = gemini_result["corrected_text"]
summary = gemini_result["summary"]
keywords = gemini_result["keywords"]
```

#### 3. Text Analyzer (`app/text_analyzer.py`)

The `TextAnalyzer` class uses HuggingFace Transformers for advanced NLP analysis:

- **Sentiment Analysis**: Determines emotional tone (positive/negative/neutral) using DistilBERT
- **Emotion Analysis**: Extended emotion categorization with confidence scores
- **Content Assessment**: Evaluates quality metrics like vocabulary diversity
- **Strengths & Improvements**: Identifies content strengths and areas for improvement
- **Device Optimization**: Utilizes MPS (Metal Performance Shaders) on Apple Silicon

```python
# Example workflow
text_analyzer = TextAnalyzer()
analysis_results = text_analyzer.analyze_transcript(transcript)
sentiment = analysis_results["sentiment_analysis"]
content_assessment = analysis_results["content_assessment"]
emotion_analysis = analysis_results["emotion_analysis"]
```

#### 4. Django Views (`app/views.py`)

Orchestrates the entire processing pipeline:

- **File Upload Handling**: Validates and processes uploaded media files
- **Database Integration**: Stores results in SQLite using Django ORM
- **API Endpoints**: RESTful endpoints for upload and health checking
- **Helper Functions**:
  - `detect_repeated_words()`: Identifies overused words in the transcript
  - `detect_filler_words()`: Finds common filler words like "um", "uh", etc.

#### 5. Database Models (`app/models.py`)

Django ORM models for data persistence:

- **Upload Model**: Stores file metadata and upload information
- **Transcript Model**: Stores transcription results and language detection
- **Analysis Model**: Stores all AI analysis results in structured format

### Data Flow

1. **Frontend → Backend**:
   - User uploads video/audio file via form
   - File is sent to `/upload-video/` endpoint

2. **Processing Pipeline**:
   ```
   Upload → Whisper Transcription → Gemini Processing → HuggingFace Analysis → Database Storage → Combined Results
   ```

3. **Backend → Frontend**:
   - JSON response with all analysis results
   - Frontend renders each section with visualizations
   - Results stored persistently in SQLite database

### Analysis Components Details

#### Sentiment Analysis

- Uses DistilBERT model fine-tuned on SST-2 dataset
- Calculates positive, negative, and neutral scores
- Provides confidence percentage and overall sentiment classification
- Handles long text by chunking and averaging results

#### Content Quality Assessment

- Evaluates overall quality score (0-100%)
- Measures vocabulary diversity and clarity
- Determines complexity level (basic, intermediate, advanced)
- Visualizes metrics using circular progress indicators

#### Strengths & Improvements Analysis

- Identifies content strengths with percentage scores
- Suggests specific areas for improvement
- Provides detailed metrics for:
  - Vocabulary quality (unique word ratio)
  - Fluency (filler word percentage)
  - Sentence structure (average sentence length)
  - Content length assessment

#### Repeated & Filler Words Detection

- Identifies frequently repeated words (excluding common words)
- Detects filler words with occurrence percentages
- Visualizes word frequency with proportional bars

### Error Handling

The application implements comprehensive error handling:

- File format validation
- Transcription failure detection
- API connection error recovery
- User-friendly error messages with suggestions
- Detailed error logging for troubleshooting

### Performance Optimization

- **Efficient File Handling**: Uses temporary files to avoid memory overload
- **Model Optimization**: Loads models once at startup
- **Resource Management**: Cleans up temporary files after processing
- **Response Structure**: Organized JSON for efficient frontend rendering




<!-- python manage.py runserver -->