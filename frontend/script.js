document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const videoFileInput = document.getElementById('videoFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileLabel = document.querySelector('.file-label .file-text');
    
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');

    // Function to update the Presentation Coach feedback with dynamic content from Gemini API
    function updatePresentationCoachFeedback(poseData) {
        // Get references to the coach elements
        const coachSummary = document.getElementById('coachSummary');
        const coachInterpretation = document.getElementById('coachInterpretation');
        const coachSuggestions = document.getElementById('coachSuggestions');

        // Show loading state
        coachSummary.innerHTML = `
            <div class="analysis-placeholder">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Analyzing presentation skills...</p>
            </div>
        `;
        coachInterpretation.innerHTML = `
            <div class="analysis-placeholder">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Interpreting body language and vocal data...</p>
            </div>
        `;
        coachSuggestions.innerHTML = `
            <div class="analysis-placeholder">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Generating personalized suggestions...</p>
            </div>
        `;

        // Make an API call to the backend to generate coach feedback using Gemini API
        fetch('/generate-coach-feedback/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                metrics: {
                    smiles: poseData.smiles,
                    head_moves: poseData.head_moves,
                    hand_moves: poseData.hand_moves,
                    eye_contact: poseData.eye_contact,
                    leg_moves: poseData.leg_moves,
                    foot_moves: poseData.foot_moves,
                    audio: poseData.audio
                }
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to generate coaching feedback');
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.feedback) {
                // Update the coach elements with the generated content
                coachSummary.innerHTML = data.feedback.summary || 'Summary not available';
                // Extract and display body analysis in coachInterpretation
                const bodyMatch = data.feedback.raw_text.match(/<div class=\"body-analysis\">([\s\S]*?)<\/div>/);
                if (bodyMatch) {
                    coachInterpretation.innerHTML = bodyMatch[0];
                } else {
                    coachInterpretation.innerHTML = data.feedback.interpretation || 'Interpretation not available';
                }
                coachSuggestions.innerHTML = data.feedback.suggestions || 'Suggestions not available';
                // Update the Voice Analysis section in Presentation Coach using Gemini feedback
                updateCoachVoiceAnalysisFromGemini(data.feedback.raw_text);
            } else {
                throw new Error(data.error || 'Unknown error generating feedback');
            }
        })
        .catch(error => {
            console.error('Error generating coach feedback:', error);
            
            // Fallback to basic feedback if API call fails
            generateFallbackFeedback(poseData);
            // Also clear the coach voice analysis section on error
            updateCoachVoiceAnalysis(null);
        });
    }
    
    // Fallback function if the API call fails
    function generateFallbackFeedback(poseData) {
        const coachSummary = document.getElementById('coachSummary');
        const coachInterpretation = document.getElementById('coachInterpretation');
        const coachSuggestions = document.getElementById('coachSuggestions');
        
        // Extract key metrics
        const smiles = poseData.smiles ? parseInt(poseData.smiles) : 0;
        const eyeContact = poseData.eye_contact || '0%';
        const eyeContactPercent = parseInt(eyeContact.replace('%', '')) || 0;
        const handMoves = poseData.hand_moves ? parseInt(poseData.hand_moves) : 0;
        const pitchRange = poseData.audio.pitch_range || '0-0 Hz';
        const numPauses = poseData.audio.num_pauses || 0;
        const speakingTime = poseData.audio.spoken_duration_sec || 0;
        const totalTime = poseData.audio.duration_sec || 0;
        
        // Generate a basic summary
        coachSummary.innerHTML = `
            <p>Your presentation shows a combination of strengths and areas for improvement in body language and vocal delivery. 
            With practice on a few key aspects, you can significantly enhance your overall presentation effectiveness.</p>
        `;
        
        // Generate basic interpretation
        coachInterpretation.innerHTML = `
            <ul>
                <li><strong>Body Language:</strong> You maintained eye contact for ${eyeContact} of your presentation time.
                You smiled ${smiles} times and used hand gestures ${handMoves} times during your presentation.</li>
                <li><strong>Voice Delivery:</strong> Your pitch range was ${pitchRange}, and you used ${numPauses} pauses.
                You actively spoke for ${Math.round((speakingTime/totalTime)*100)}% of your total presentation time.</li>
            </ul>
        `;
        
        // Generate basic suggestions
        coachSuggestions.innerHTML = `
            <ol>
                <li><strong>Improve Eye Contact:</strong> Try to maintain consistent eye contact with your audience by dividing the room into sections and giving each section equal attention.</li>
                <li><strong>Use More Gestures:</strong> Incorporate deliberate hand movements to emphasize key points in your presentation.</li>
                <li><strong>Vary Your Vocal Delivery:</strong> Practice changing your pitch and using strategic pauses to make your presentation more engaging and to emphasize important points.</li>
            </ol>
        `;
        
        // Update voice analysis with fallback data
        if (poseData && poseData.audio) {
            updateCoachVoiceAnalysis(poseData.audio);
        }
    }

    // Function to update the Presentation Coach Voice Analysis section dynamically
    function updateCoachVoiceAnalysis(audioData) {
        const coachVoiceAnalysis = document.getElementById('coachVoiceAnalysis');
        if (!audioData) {
            coachVoiceAnalysis.innerHTML = `<div class="analysis-placeholder">
                <i class="fas fa-microphone-alt"></i>
                <p>No voice data available.</p>
            </div>`;
            return;
        }
        // Build dynamic metrics table
        let html = `<div class="voice-results coach-voice-results">`;
        if (audioData.duration_sec !== undefined) {
            html += `<div class="voice-metric"><h5>Duration</h5><div class="metric-value">${audioData.duration_sec}s</div><div class="metric-unit">Total Length</div></div>`;
        }
        if (audioData.spoken_duration_sec !== undefined) {
            html += `<div class="voice-metric"><h5>Speaking Time</h5><div class="metric-value">${audioData.spoken_duration_sec}s</div><div class="metric-unit">Active Speaking</div></div>`;
        }
        if (audioData.volume_db !== undefined) {
            html += `<div class="voice-metric"><h5>Volume</h5><div class="metric-value">${audioData.volume_db} dB</div><div class="metric-unit">Average Volume</div></div>`;
        }
        if (audioData.mean_pitch_hz !== undefined) {
            html += `<div class="voice-metric"><h5>Mean Pitch</h5><div class="metric-value">${audioData.mean_pitch_hz} Hz</div><div class="metric-unit">Average Pitch</div></div>`;
        }
        if (audioData.pitch_range !== undefined) {
            html += `<div class="voice-metric"><h5>Pitch Range</h5><div class="metric-value">${audioData.pitch_range}</div><div class="metric-unit">Frequency Range</div></div>`;
        }
        if (audioData.num_pauses !== undefined) {
            html += `<div class="voice-metric"><h5>Pauses</h5><div class="metric-value">${audioData.num_pauses}</div><div class="metric-unit">Number of Pauses</div></div>`;
        }
        html += `</div>`;
        coachVoiceAnalysis.innerHTML = html;
    }

    // Function to update the Presentation Coach Voice Analysis section dynamically
    function updateCoachVoiceAnalysisFromGemini(feedbackText) {
        const coachVoiceAnalysis = document.getElementById('coachVoiceAnalysis');
        // Extract the <div class="vocal-analysis">...</div> section from Gemini feedback
        // Try both escaped and unescaped quotes
        const vocalMatch = feedbackText.match(/<div class=["']vocal-analysis["']>([\s\S]*?)<\/div>/);
        if (vocalMatch) {
            coachVoiceAnalysis.innerHTML = vocalMatch[0];
        } else {
            // Debug: Log the feedback text to see what's being received
            console.log('Gemini feedback text:', feedbackText);
            console.log('No vocal analysis section found in Gemini response');
            
            // If no vocal analysis found in Gemini response, show a fallback message
            coachVoiceAnalysis.innerHTML = `<div class="analysis-placeholder">
                <i class="fas fa-microphone-alt"></i>
                <p>Vocal analysis will be available after processing completes.</p>
            </div>`;
        }
    }

    // Add dropdown functionality for navigation
    document.querySelectorAll('.dropdown-toggle').forEach(toggle => {
        toggle.addEventListener('click', function(e) {
            e.preventDefault();
            const dropdown = this.closest('.nav-dropdown');
            const isActive = dropdown.classList.contains('active');
            
            // Close all other dropdowns
            document.querySelectorAll('.nav-dropdown').forEach(dd => {
                dd.classList.remove('active');
            });
            
            // Toggle current dropdown
            if (!isActive) {
                dropdown.classList.add('active');
            }
        });
    });

    // Add smooth scrolling functionality for navigation links
    document.querySelectorAll('.nav-links a').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1); // Remove the #
            
            // If results aren't displayed yet, just scroll to the upload section
            if (resultsSection.style.display === 'none' && targetId !== 'uploadSection') {
                scrollToSection('uploadSection');
            } else {
                scrollToSection(targetId);
                
                // Highlight the active nav item
                document.querySelectorAll('.nav-links li').forEach(item => {
                    item.classList.remove('active');
                });
                this.parentElement.classList.add('active');
            }
        });
    });
    
    // Add event listeners for section navigation buttons
    document.querySelectorAll('.section-nav-btn').forEach(button => {
        // Make buttons accessible via keyboard
        button.setAttribute('role', 'button');
        button.setAttribute('tabindex', '0');
        
        // Handle click events
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-scroll-to');
            scrollToSection(targetId);
            
            // Also update the sidebar navigation
            document.querySelectorAll('.nav-links li').forEach(item => {
                const link = item.querySelector('a');
                if (link && link.getAttribute('href') === `#${targetId}`) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            });
        });
        
        // Handle keyboard events (Enter and Space)
        button.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.click();
            }
        });
    });
    
    // Update file label when file is selected
    videoFileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const fileName = this.files[0].name;
            const fileSize = (this.files[0].size / (1024 * 1024)).toFixed(2);
            fileLabel.textContent = `${fileName} (${fileSize} MB)`;
        } else {
            fileLabel.textContent = 'Choose Video/Audio File';
        }
    });
    
    // Add button functionality to copy text from transcript and corrected sections
    document.querySelectorAll('.control-btn[title="Copy Text"]').forEach(btn => {
        btn.addEventListener('click', function() {
            // Find the closest card and then the text content
            const card = this.closest('.result-card');
            const textBox = card.querySelector('.transcript-box, .summary-box');
            
            if (textBox && textBox.textContent) {
                navigator.clipboard.writeText(textBox.textContent)
                    .then(() => {
                        // Show a temporary success message
                        const originalIcon = this.innerHTML;
                        this.innerHTML = '<i class="fas fa-check"></i>';
                        setTimeout(() => {
                            this.innerHTML = originalIcon;
                        }, 2000);
                    })
                    .catch(err => console.error('Could not copy text: ', err));
            }
        });
    });
    
    // Add button functionality to expand text sections
    document.querySelectorAll('.control-btn[title="Expand"]').forEach(btn => {
        btn.addEventListener('click', function() {
            // Find the closest card and then the text box
            const card = this.closest('.result-card');
            const textBox = card.querySelector('.transcript-box, .summary-box');
            
            if (textBox) {
                if (textBox.style.maxHeight === 'none') {
                    // Collapse
                    textBox.style.maxHeight = '300px';
                    this.innerHTML = '<i class="fas fa-expand"></i>';
                    this.title = "Expand";
                } else {
                    // Expand
                    textBox.style.maxHeight = 'none';
                    this.innerHTML = '<i class="fas fa-compress"></i>';
                    this.title = "Collapse";
                }
            }
        });
    });

    // Handle form submission when the user clicks "Transcribe & Analyze"
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Make sure a file was selected
        const fileInput = document.getElementById('videoFile');
        if (!fileInput.files || !fileInput.files[0]) {
            alert('Please select a video or audio file first!');
            return;
        }

        // Hide any previous results or errors
        resultsSection.style.display = 'none';
        errorSection.style.display = 'none';
        
        // Show the loading spinner
        loadingSection.style.display = 'block';
        uploadBtn.disabled = true;
        uploadBtn.querySelector('.btn-text').textContent = '⏳ Processing...';

        try {
            // Create form data with the selected file
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            // Function to get CSRF token (for future use when CSRF is enabled)
            function getCookie(name) {
                const value = `; ${document.cookie}`;
                const parts = value.split(`; ${name}=`);
                if (parts.length === 2) return parts.pop().split(';').shift();
            }
            
            // Send the file to the backend for processing
            const response = await fetch('/upload-video/', {
                method: 'POST',
                body: formData,
                // Uncomment the following line when CSRF protection is enabled
                // headers: {'X-CSRFToken': getCookie('csrftoken')}
            });

            // Try to parse the response as JSON
            const result = await response.json().catch(err => {
                console.error("Failed to parse JSON response:", err);
                return { detail: "Server error: Failed to process response" };
            });

            if (!response.ok) {
                throw new Error(result.detail || 'Upload failed');
            }

            displayResults(result);
            
        } catch (error) {
            console.error('Error:', error);
            displayError(error.message);
        } finally {
            // Hide loading
            loadingSection.style.display = 'none';
            uploadBtn.disabled = false;
            uploadBtn.querySelector('.btn-text').textContent = '🚀 Transcribe & Analyze';
        }
    });

    // Helper function for smooth scrolling to elements
    function scrollToSection(elementId, skipHistory) {
        const element = document.getElementById(elementId);
        if (element) {
            // Get the element's position relative to the viewport
            const rect = element.getBoundingClientRect();
            
            // Calculate position to scroll to (with offset for fixed header and sticky navigation)
            const offset = 120; // Increased offset for better visibility with sticky nav
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            
            // Scroll to the element with smooth behavior
            window.scrollTo({
                top: rect.top + scrollTop - offset,
                behavior: 'smooth'
            });
            
            // Update the active nav item in sidebar
            document.querySelectorAll('.nav-links li').forEach(item => {
                const link = item.querySelector('a');
                if (link && link.getAttribute('href') === `#${elementId}`) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            });
            
            // Update the active button in section navigation
            document.querySelectorAll('.section-nav-btn').forEach(btn => {
                if (btn.getAttribute('data-scroll-to') === elementId) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
            
            // Add a temporary highlight effect to the section
            element.classList.add('highlight');
            setTimeout(() => {
                element.classList.remove('highlight');
            }, 1500);
            
            // Update the URL hash without jumping
            if (!skipHistory) {
                history.pushState(null, null, `#${elementId}`);
            }
        }
    }

    // Add event listeners to all buttons/links that should trigger scrolling
    document.querySelectorAll('[data-scroll-to]').forEach(button => {
        // Make the element accessible via keyboard if it's not already a button
        if (button.tagName !== 'BUTTON') {
            button.setAttribute('role', 'button');
            button.setAttribute('tabindex', '0');
        }
        
        // Handle click events
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-scroll-to');
            scrollToSection(targetId);
        });
        
        // Handle keyboard events for elements that aren't buttons
        if (button.tagName !== 'BUTTON') {
            button.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.click();
                }
            });
        }
    });

    // Add scroll spy functionality - highlights the current section in nav
    function handleScrollSpy() {
        // Only run this if results are visible
        if (resultsSection.style.display === 'none') return;
        
        const sections = [
            'uploadSection',
            'transcriptSection',
            'summarySection',
            'sentimentSection',
            'wordsAnalysisSection',
            'enhancedAnalysisSection'
        ];
        
        // Find which section is currently most visible in the viewport
        let currentSection = sections[0];
        let maxVisibility = 0;
        
        sections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            if (section) {
                const rect = section.getBoundingClientRect();
                const sectionHeight = rect.height;
                
                // Calculate how much of the section is visible
                const visibleTop = Math.max(0, rect.top);
                const visibleBottom = Math.min(window.innerHeight, rect.bottom);
                const visibleHeight = Math.max(0, visibleBottom - visibleTop);
                
                // Calculate visibility ratio
                const visibilityRatio = visibleHeight / sectionHeight;
                
                if (visibilityRatio > maxVisibility) {
                    maxVisibility = visibilityRatio;
                    currentSection = sectionId;
                }
            }
        });
        
        // Update active nav link in sidebar
        document.querySelectorAll('.nav-links li').forEach(item => {
            const link = item.querySelector('a');
            if (link && link.getAttribute('href') === `#${currentSection}`) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        
        // Update active section navigation button
        document.querySelectorAll('.section-nav-btn').forEach(btn => {
            if (btn.getAttribute('data-scroll-to') === currentSection) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }
    
    // Listen for scroll events to update active section
    window.addEventListener('scroll', function() {
        handleScrollSpy();
        
        // Handle scroll-to-top button visibility
        const scrollToTopBtn = document.getElementById('scrollToTopBtn');
        if (window.scrollY > 500) {
            scrollToTopBtn.classList.add('visible');
        } else {
            scrollToTopBtn.classList.remove('visible');
        }
    });
    
    // Add click handler for scroll-to-top button
    const scrollToTopBtn = document.getElementById('scrollToTopBtn');
    scrollToTopBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
        
        // Update active state in navigation
        document.querySelectorAll('.nav-links li').forEach(item => {
            const link = item.querySelector('a');
            if (link && link.getAttribute('href') === '#uploadSection') {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        
        // Update active section navigation button
        document.querySelectorAll('.section-nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Update URL without triggering scroll
        history.pushState(null, null, '#uploadSection');
    });
    
    // Add keyboard support for scroll-to-top button
    scrollToTopBtn.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            this.click();
        }
    });
    
    function displayResults(data) {
        // Display file information
        document.getElementById('fileInfo').innerHTML = `
            <p><strong>📁 Filename:</strong> ${data.filename}</p>
            <p><strong>💾 Size:</strong> ${data.file_size}</p>
            <p><strong>🌐 Language:</strong> ${data.language || 'unknown'}</p>
            <p><strong>⏱️ Duration:</strong> ${data.duration || '00:00:00'}</p>
        `;

        // Display original transcript from Whisper
        document.getElementById('originalTranscript').textContent = data.original_transcript;

        // Display grammar-corrected transcript from Gemini with error highlighting
        displayCorrectedTranscriptWithHighlighting('correctedTranscript', data.original_transcript, data.corrected_transcript);

        // Display summary from Gemini
        document.getElementById('summary').textContent = data.summary;
        
        // Display repeated words with improved visualization
        if (data.repeated_words && data.repeated_words.length > 0) {
            // Add explanatory text
            const explanation = `<p class="analysis-explanation">These words appear frequently in your speech. 
            Consider using synonyms for variety.</p>`;
            
            // Calculate max count for visualization
            const maxCount = Math.max(...data.repeated_words.map(item => item.count));
            
            const repeatedWordsHtml = data.repeated_words.map(item => {
                // Calculate percentage width for the bar visualization
                const percentWidth = Math.max((item.count / maxCount) * 100, 20);
                
                return `<div class="word-item">
                    <div class="word-info">
                        <span class="word-name">${item.word}</span>
                        <div class="word-bar" style="width: ${percentWidth}%"></div>
                    </div>
                    <span class="word-count">${item.count}</span>
                </div>`;
            }).join('');
            
            document.getElementById('repeatedWords').innerHTML = explanation + repeatedWordsHtml;
        } else {
            document.getElementById('repeatedWords').innerHTML = '<p>No significant repeated words detected</p>';
        }
        
        // Display filler words with percentage
        if (data.filler_words && data.filler_words.length > 0) {
            // Create word items with improved spacing
            const fillerWordsHtml = data.filler_words.map(item => 
                `<div class="word-item">
                    <div class="word-info">
                        <span class="word-name">${item.word}</span>
                        ${item.percentage ? `<span class="word-percentage">${item.percentage}%</span>` : ''}
                    </div>
                    <span class="word-count">${item.count}</span>
                </div>`
            ).join('');
            
            // Add explanatory text
            const explanation = `<p class="analysis-explanation">These are filler words that appear in your speech. 
            Reducing them can make your delivery more impactful.</p>`;
            
            document.getElementById('fillerWords').innerHTML = explanation + fillerWordsHtml;
        } else {
            document.getElementById('fillerWords').innerHTML = '<p class="analysis-explanation">No significant filler words detected.</p>';
        }

        // Create keyword badges from the comma-separated list
        try {
            const keywordsArray = data.keywords.split(',').map(k => k.trim()).filter(k => k);
            const keywordsHtml = keywordsArray.map(keyword => 
                `<span class="keyword">${keyword}</span>`
            ).join('');
            document.getElementById('keywords').innerHTML = keywordsHtml;
        } catch (e) {
            document.getElementById('keywords').textContent = data.keywords;
        }
        
        // Display Sentiment Analysis from HuggingFace Transformers
        if (data.sentiment_analysis) {
            const sentiment = data.sentiment_analysis;
            const sentimentClass = sentiment.overall_sentiment === 'positive' ? 'positive' : 'negative';
            
            // Create a visual representation of sentiment score
            const sentimentHtml = `
                <div class="sentiment-result ${sentimentClass}">
                    <div class="sentiment-icon">${sentiment.overall_sentiment === 'positive' ? '😊' : '😔'}</div>
                    <div class="sentiment-label">
                        <span>Overall Sentiment: ${sentiment.overall_sentiment.toUpperCase()}</span>
                        <div class="sentiment-confidence">Confidence: ${sentiment.confidence}%</div>
                    </div>
                </div>
                
                <div class="sentiment-scores">
                    <div class="score-container">
                        <div class="score-label">Positive</div>
                        <div class="score-bar-container">
                            <div class="score-bar positive" style="width: ${sentiment.positive_score}%"></div>
                        </div>
                        <div class="score-value">${sentiment.positive_score}%</div>
                    </div>
                    
                    <div class="score-container">
                        <div class="score-label">Negative</div>
                        <div class="score-bar-container">
                            <div class="score-bar negative" style="width: ${sentiment.negative_score}%"></div>
                        </div>
                        <div class="score-value">${sentiment.negative_score}%</div>
                    </div>
                    
                    <div class="score-container">
                        <div class="score-label">Neutral</div>
                        <div class="score-bar-container">
                            <div class="score-bar neutral" style="width: ${sentiment.neutral_score}%"></div>
                        </div>
                        <div class="score-value">${sentiment.neutral_score}%</div>
                    </div>
                </div>
            `;
            
            document.getElementById('sentimentAnalysis').innerHTML = sentimentHtml;
        }
        
        // Display Emotion Analysis
        if (data.emotion_analysis) {
            const emotion = data.emotion_analysis;
            
            // Create emotion display with dominant emotion and confidence
            const emotionHtml = `
                <div class="emotion-result">
                    <div class="emotion-main">
                        <div class="emotion-icon">${emotion.emoji}</div>
                        <div class="emotion-label">
                            <span class="emotion-name">${emotion.dominant_emotion}</span>
                            <div class="emotion-confidence">Confidence: ${emotion.confidence}%</div>
                        </div>
                    </div>
                    
                    <div class="emotion-description">
                        ${getEmotionDescription(emotion.dominant_emotion)}
                    </div>
                    
                    ${emotion.detected_keywords && emotion.detected_keywords.length > 0 ? `
                        <div class="emotion-keywords">
                            <div class="keywords-title">Detected Keywords:</div>
                            <div class="keywords-list">
                                ${emotion.detected_keywords.slice(0, 5).map(([emotionType, keyword]) => 
                                    `<span class="keyword-tag">${keyword}</span>`
                                ).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    ${Object.keys(emotion.emotion_scores).filter(key => emotion.emotion_scores[key] > 0).length > 1 ? `
                        <div class="emotion-breakdown">
                            <div class="breakdown-title">Emotion Breakdown:</div>
                            <div class="emotion-scores">
                                ${Object.entries(emotion.emotion_scores)
                                    .filter(([_, score]) => score > 0)
                                    .sort(([_, a], [__, b]) => b - a)
                                    .slice(0, 5)
                                    .map(([emotionType, score]) => {
                                        const emoji = getEmotionEmoji(emotionType);
                                        const percentage = emotion.emotion_scores ? 
                                            Math.round((score / Math.max(1, Object.values(emotion.emotion_scores).reduce((a, b) => a + b, 0))) * 100) : 0;
                                        return `
                                            <div class="emotion-score-item">
                                                <span class="emotion-emoji">${emoji}</span>
                                                <span class="emotion-type">${emotionType}</span>
                                                <div class="emotion-bar-container">
                                                    <div class="emotion-bar" style="width: ${percentage}%"></div>
                                                </div>
                                                <span class="emotion-percentage">${percentage}%</span>
                                            </div>
                                        `;
                                    }).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            `;
            
            document.getElementById('emotionAnalysis').innerHTML = emotionHtml;
        }
        
        // Helper function to get emotion descriptions
        function getEmotionDescription(emotion) {
            const descriptions = {
                'Joy / Happiness': 'Your content expresses positive emotions, cheerfulness, and enthusiasm.',
                'Sadness': 'Your content contains expressions of sadness, disappointment, or melancholy.',
                'Anger': 'Your content shows signs of frustration, annoyance, or strong negative feelings.',
                'Fear': 'Your content expresses anxiety, worry, or concern about something.',
                'Surprise': 'Your content contains expressions of amazement, shock, or unexpected reactions.',
                'Disgust': 'Your content shows aversion, disapproval, or strong negative reactions.',
                'Trust': 'Your content expresses confidence, reliability, and positive assurance.',
                'Anticipation': 'Your content shows hope, curiosity, and forward-looking excitement.',
                'Love': 'Your content expresses affection, care, and positive interpersonal feelings.',
                'Optimism': 'Your content demonstrates hopefulness and positive outlook on the future.',
                'Pessimism': 'Your content expresses negativity or doubtful views about outcomes.',
                'Neutral': 'Your content maintains a balanced, objective emotional tone.'
            };
            return descriptions[emotion] || 'Emotional tone detected in your content.';
        }
        
        // Helper function to get emotion emojis
        function getEmotionEmoji(emotion) {
            const emojis = {
                'Joy / Happiness': '😊',
                'Sadness': '😢',
                'Anger': '😠',
                'Fear': '😰',
                'Surprise': '😲',
                'Disgust': '🤢',
                'Trust': '🤝',
                'Anticipation': '🤔',
                'Love': '❤️',
                'Optimism': '🌟',
                'Pessimism': '😔',
                'Neutral': '😐'
            };
            return emojis[emotion] || '😐';
        }
        
        // Display Content Assessment from HuggingFace Transformers
        if (data.content_assessment) {
            const assessment = data.content_assessment;
            
            // Create circular progress indicator for quality score
            const qualityScore = assessment.quality_score;
            const qualityColor = qualityScore > 75 ? '#4CAF50' : qualityScore > 50 ? '#FF9800' : '#F44336';
            
            const contentHtml = `
                <div class="quality-meter">
                    <div class="circular-progress" style="background: conic-gradient(${qualityColor} ${qualityScore * 3.6}deg, #f0f0f0 0deg);">
                        <div class="inner-circle">
                            <span class="quality-score">${qualityScore}%</span>
                        </div>
                    </div>
                    <div class="quality-label">Overall Quality</div>
                </div>
                
                <div class="content-metrics">
                    <div class="metric">
                        <span class="metric-name">Vocabulary Diversity</span>
                        <div class="metric-bar-container">
                            <div class="metric-bar" style="width: ${assessment.vocabulary_diversity}%"></div>
                        </div>
                        <span class="metric-value">${assessment.vocabulary_diversity}%</span>
                    </div>
                    
                    <div class="metric">
                        <span class="metric-name">Clarity Score</span>
                        <div class="metric-bar-container">
                            <div class="metric-bar" style="width: ${assessment.clarity_score}%"></div>
                        </div>
                        <span class="metric-value">${assessment.clarity_score}%</span>
                    </div>
                    
                    <div class="complexity-level">
                        <span class="complexity-label">Complexity Level</span>
                        <span class="complexity-value">${assessment.complexity_level}</span>
                    </div>
                </div>
            `;
            
            document.getElementById('contentAssessment').innerHTML = contentHtml;
        }
        
        // Display Strengths & Improvements from HuggingFace Transformers
        if (data.strengths_improvements) {
            const strengthsImprovement = data.strengths_improvements;
            const detailedMetrics = strengthsImprovement.detailed_metrics || {};
            
            // Create sections for strengths and areas to improve
            let strengthsHtml = '<div class="strengths-section">';
            strengthsHtml += '<h4>Strengths</h4>';
            
            if (strengthsImprovement.strengths && strengthsImprovement.strengths.length > 0) {
                strengthsHtml += '<h4><i class="fas fa-check-circle"></i> Strengths</h4>';
                strengthsHtml += '<ul class="strengths-list">';
                strengthsImprovement.strengths.forEach(item => {
                    // Add score badge if available
                    const scoreBadge = item.score ? 
                        `<span class="metric-badge ${item.score >= 80 ? 'high' : item.score >= 60 ? 'medium' : 'low'}">${item.score}%</span>` : '';
                        
                    strengthsHtml += `<li class="strength-item">
                        <div class="strength-header">
                            <span class="strength-area">${item.area}</span> 
                            ${scoreBadge}
                        </div>
                        <span class="strength-desc">${item.description}</span>
                    </li>`;
                });
                strengthsHtml += '</ul>';
            } else {
                strengthsHtml += '<h4><i class="fas fa-check-circle"></i> Strengths</h4>';
                strengthsHtml += '<p>No significant strengths identified.</p>';
            }
            strengthsHtml += '</div>';
            
            let improvementsHtml = '<div class="improvements-section">';
            improvementsHtml += '<h4><i class="fas fa-arrow-circle-up"></i> Areas for Improvement</h4>';
            
            if (strengthsImprovement.improvements && strengthsImprovement.improvements.length > 0) {
                improvementsHtml += '<ul class="improvements-list">';
                strengthsImprovement.improvements.forEach(item => {
                    // Add score badge if available
                    const scoreBadge = item.score ? 
                        `<span class="metric-badge ${item.score >= 80 ? 'high' : item.score >= 60 ? 'medium' : 'low'}">${item.score}%</span>` : '';
                        
                    improvementsHtml += `<li class="improvement-item">
                        <div class="improvement-header">
                            <span class="improvement-area">${item.area}</span>
                            ${scoreBadge}
                        </div>
                        <span class="improvement-desc">${item.description}</span>
                    </li>`;
                });
                improvementsHtml += '</ul>';
            } else {
                improvementsHtml += '<p>No significant areas for improvement identified.</p>';
            }
            improvementsHtml += '</div>';
            
            // Add detailed metrics visualization if available
            let detailedMetricsHtml = '';
            if (Object.keys(detailedMetrics).length > 0) {
                detailedMetricsHtml = `
                    <div class="detailed-metrics">
                        <h4>Detailed Metrics</h4>
                        <div class="metrics-grid">
                            ${detailedMetrics.vocabulary_score ? `
                            <div class="metric-card">
                                <div class="metric-title">Vocabulary</div>
                                <div class="metric-circle ${getScoreClass(detailedMetrics.vocabulary_score)}">
                                    <span>${detailedMetrics.vocabulary_score}%</span>
                                </div>
                                ${detailedMetrics.unique_word_ratio ? 
                                  `<div class="metric-detail">${detailedMetrics.unique_word_ratio}% unique words</div>` : ''}
                            </div>` : ''}
                            
                            ${detailedMetrics.fluency_score ? `
                            <div class="metric-card">
                                <div class="metric-title">Fluency</div>
                                <div class="metric-circle ${getScoreClass(detailedMetrics.fluency_score)}">
                                    <span>${detailedMetrics.fluency_score}%</span>
                                </div>
                                ${detailedMetrics.filler_word_percentage ? 
                                  `<div class="metric-detail">${detailedMetrics.filler_word_percentage}% filler words</div>` : ''}
                            </div>` : ''}
                            
                            ${detailedMetrics.sentence_score ? `
                            <div class="metric-card">
                                <div class="metric-title">Sentence Structure</div>
                                <div class="metric-circle ${getScoreClass(detailedMetrics.sentence_score)}">
                                    <span>${detailedMetrics.sentence_score}%</span>
                                </div>
                                ${detailedMetrics.avg_sentence_length ? 
                                  `<div class="metric-detail">~${detailedMetrics.avg_sentence_length} words/sentence</div>` : ''}
                            </div>` : ''}
                            
                            ${detailedMetrics.length_score ? `
                            <div class="metric-card">
                                <div class="metric-title">Content Length</div>
                                <div class="metric-circle ${getScoreClass(detailedMetrics.length_score)}">
                                    <span>${detailedMetrics.length_score}%</span>
                                </div>
                            </div>` : ''}
                        </div>
                    </div>
                `;
            }
            
            // Add score visualization with improved design
            const scoresHtml = `
                <div class="score-visualization">
                    <div class="score-summary">
                        <div class="summary-heading">Content Analysis Summary</div>
                        <div class="summary-desc">Based on vocabulary, fluency, sentence structure, and other metrics</div>
                        
                        <div class="key-insights">
                            <h4><i class="fas fa-lightbulb"></i> Key Insights</h4>
                            <ul class="insights-list">
                                ${generateKeyInsights(data)}
                            </ul>
                        </div>
                    </div>
                    
                    <div class="score-item">
                        <div class="score-label">
                            <span class="score-icon">💪</span>
                            <span>Overall Strength</span>
                        </div>
                        <div class="score-value">${strengthsImprovement.strength_score}%</div>
                        <div class="score-bar-container">
                            <div class="score-bar strength" style="width: ${strengthsImprovement.strength_score}%"></div>
                        </div>
                        <div class="score-explanation">${getScoreExplanation(strengthsImprovement.strength_score)}</div>
                    </div>
                    
                    <div class="score-item">
                        <div class="score-label">
                            <span class="score-icon">📈</span>
                            <span>Growth Potential</span>
                        </div>
                        <div class="score-value">${strengthsImprovement.improvement_areas_score}%</div>
                        <div class="score-bar-container">
                            <div class="score-bar improvement" style="width: ${strengthsImprovement.improvement_areas_score}%"></div>
                        </div>
                        <div class="score-explanation">${getImprovementExplanation(strengthsImprovement.improvement_areas_score)}</div>
                    </div>
                </div>
            `;
            
            // Generate key insights summary function
            function generateKeyInsights(data) {
                let insights = [];
                
                // Sentiment Analysis Insights
                if (data.sentiment_analysis) {
                    const sentiment = data.sentiment_analysis;
                    const sentimentIcon = sentiment.overall_sentiment === 'positive' ? '😊' : 
                                         sentiment.overall_sentiment === 'negative' ? '😔' : '😐';
                    insights.push(`<li><strong>Sentiment:</strong> ${sentimentIcon} ${sentiment.overall_sentiment.toUpperCase()} tone (${sentiment.confidence}% confidence)</li>`);
                }
                
                // Emotion Analysis Insights
                if (data.emotion_analysis) {
                    const emotion = data.emotion_analysis;
                    insights.push(`<li><strong>Dominant Emotion:</strong> ${emotion.emoji} ${emotion.dominant_emotion} (${emotion.confidence}% confidence)</li>`);
                }
                
                // Content Quality Insights
                if (data.content_assessment) {
                    const assessment = data.content_assessment;
                    const qualityIcon = assessment.quality_score >= 80 ? '🌟' : 
                                       assessment.quality_score >= 60 ? '👍' : '📈';
                    insights.push(`<li><strong>Content Quality:</strong> ${qualityIcon} ${assessment.quality_score}% overall quality score</li>`);
                    insights.push(`<li><strong>Vocabulary Diversity:</strong> 📚 ${assessment.vocabulary_diversity}% unique words</li>`);
                    insights.push(`<li><strong>Complexity Level:</strong> 🎯 ${assessment.complexity_level}</li>`);
                }
                
                // Strengths and Improvements Insights
                if (data.strengths_improvements) {
                    const si = data.strengths_improvements;
                    
                    // Overall scores
                    const strengthIcon = si.strength_score >= 80 ? '💪' : 
                                        si.strength_score >= 60 ? '👌' : '🔧';
                    insights.push(`<li><strong>Overall Strength:</strong> ${strengthIcon} ${si.strength_score}% content strength</li>`);
                    
                    // Top strengths
                    if (si.strengths && si.strengths.length > 0) {
                        const topStrength = si.strengths[0];
                        insights.push(`<li><strong>Top Strength:</strong> ✅ ${topStrength.area} - ${topStrength.description}</li>`);
                    }
                    
                    // Key improvement area
                    if (si.improvements && si.improvements.length > 0) {
                        const topImprovement = si.improvements[0];
                        insights.push(`<li><strong>Key Improvement:</strong> 🎯 ${topImprovement.area} - ${topImprovement.description}</li>`);
                    }
                    
                    // Detailed metrics insights
                    if (si.detailed_metrics) {
                        const metrics = si.detailed_metrics;
                        
                        // Fluency insight
                        if (metrics.filler_word_percentage !== undefined) {
                            const fillerIcon = metrics.filler_word_percentage <= 2 ? '🎙️' : '⚠️';
                            insights.push(`<li><strong>Speaking Fluency:</strong> ${fillerIcon} ${metrics.filler_word_percentage}% filler words detected</li>`);
                        }
                        
                        // Sentence structure insight
                        if (metrics.avg_sentence_length !== undefined) {
                            const sentenceIcon = (metrics.avg_sentence_length >= 12 && metrics.avg_sentence_length <= 20) ? '📝' : '✏️';
                            insights.push(`<li><strong>Sentence Structure:</strong> ${sentenceIcon} Average ${metrics.avg_sentence_length} words per sentence</li>`);
                        }
                    }
                }
                
                // Word Analysis Insights
                if (data.repeated_words && data.repeated_words.length > 0) {
                    const topRepeated = data.repeated_words[0];
                    insights.push(`<li><strong>Most Repeated Word:</strong> 🔄 "${topRepeated.word}" used ${topRepeated.count} times</li>`);
                }
                
                if (data.filler_words && data.filler_words.length > 0) {
                    const topFiller = data.filler_words[0];
                    insights.push(`<li><strong>Top Filler Word:</strong> 🗣️ "${topFiller.word}" (${topFiller.percentage}% of content)</li>`);
                }
                
                // Content length insight
                if (data.original_transcript) {
                    const wordCount = data.original_transcript.split(' ').length;
                    const lengthIcon = wordCount >= 300 ? '📄' : '📝';
                    insights.push(`<li><strong>Content Length:</strong> ${lengthIcon} ${wordCount} words total</li>`);
                }
                
                return insights.join('');
            }
            
            // Helper function for score class
            function getScoreClass(score) {
                if (score >= 80) return 'high-score';
                if (score >= 60) return 'medium-score';
                return 'low-score';
            }
            
            // Helper function for strength score explanations
            function getScoreExplanation(score) {
                if (score >= 85) return 'Excellent - Your content demonstrates outstanding quality in most areas';
                if (score >= 70) return 'Good - Your content shows strength in several important areas';
                if (score >= 50) return 'Fair - Your content has some strong elements to build upon';
                return 'Needs work - Focus on addressing the improvement suggestions';
            }
            
            // Helper function for improvement score explanations
            function getImprovementExplanation(score) {
                if (score >= 85) return 'Very high potential - Implementing suggestions will greatly enhance your content';
                if (score >= 70) return 'Good potential - Several specific areas can be improved';
                if (score >= 50) return 'Some potential - Focus on the key improvement areas identified';
                return 'Limited potential - Major revisions recommended';
            }
            
            document.getElementById('strengthsImprovements').innerHTML = 
                strengthsHtml + improvementsHtml + detailedMetricsHtml + scoresHtml;
        }

        // Pose and Voice Analysis
        // Trigger pose analysis after the main analysis is complete
        if (data.filename) {
            // Create a new FormData for pose analysis
            const poseFormData = new FormData();
            poseFormData.append('video', videoFileInput.files[0]);
            
            // Show loading state for pose analysis
            const poseAnalysisContent = document.getElementById('poseAnalysis');
            const voiceAnalysisContent = document.getElementById('voiceAnalysis');
            
            poseAnalysisContent.innerHTML = `
                <div class="analysis-placeholder">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Analyzing body language and facial expressions...</p>
                </div>
            `;
            
            voiceAnalysisContent.innerHTML = `
                <div class="analysis-placeholder">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Processing voice characteristics...</p>
                </div>
            `;
            
            // Perform pose and voice analysis
            fetch('/pose-voice-analysis/', {
                method: 'POST',
                body: poseFormData
            })
            .then(response => response.json())
            .then(poseData => {
                if (poseData.error) {
                    throw new Error(poseData.error);
                }
                
                // Display pose analysis results
                const poseResults = `
                    <div class="pose-results">
                        <div class="pose-metric">
                            <h5>Smiles Detected</h5>
                            <div class="metric-value">${poseData.smiles}</div>
                        </div>
                        <div class="pose-metric">
                            <h5>Head Movements</h5>
                            <div class="metric-value">${poseData.head_moves}</div>
                        </div>
                        <div class="pose-metric">
                            <h5>Hand Movements</h5>
                            <div class="metric-value">${poseData.hand_moves}</div>
                        </div>
                        <div class="pose-metric">
                            <h5>Eye Contact</h5>
                            <div class="metric-value">${poseData.eye_contact}</div>
                        </div>
                        <div class="pose-metric">
                            <h5>Leg Movements</h5>
                            <div class="metric-value">${poseData.leg_moves}</div>
                        </div>
                        <div class="pose-metric">
                            <h5>Foot Movements</h5>
                            <div class="metric-value">${poseData.foot_moves}</div>
                        </div>
                    </div>
                `;
                
                // Display voice analysis results
                const voiceResults = `
                    <div class="voice-results">
                        <div class="voice-metric">
                            <h5>Duration</h5>
                            <div class="metric-value">${poseData.audio.duration_sec}s</div>
                            <div class="metric-unit">Total Length</div>
                        </div>
                        <div class="voice-metric">
                            <h5>Volume</h5>
                            <div class="metric-value">${poseData.audio.volume_db} dB</div>
                            <div class="metric-unit">Average Volume</div>
                        </div>
                        <div class="voice-metric">
                            <h5>Mean Pitch</h5>
                            <div class="metric-value">${poseData.audio.mean_pitch_hz} Hz</div>
                            <div class="metric-unit">Average Pitch</div>
                        </div>
                        <div class="voice-metric">
                            <h5>Pitch Range</h5>
                            <div class="metric-value">${poseData.audio.pitch_range}</div>
                            <div class="metric-unit">Frequency Range</div>
                        </div>
                        <div class="voice-metric">
                            <h5>Pauses</h5>
                            <div class="metric-value">${poseData.audio.num_pauses}</div>
                            <div class="metric-unit">Number of Pauses</div>
                        </div>
                        <div class="voice-metric">
                            <h5>Speaking Time</h5>
                            <div class="metric-value">${poseData.audio.spoken_duration_sec}s</div>
                            <div class="metric-unit">Active Speaking</div>
                        </div>
                    </div>
                `;
                
                poseAnalysisContent.innerHTML = poseResults;
                voiceAnalysisContent.innerHTML = voiceResults;
                
                // Update the Presentation Coach feedback with the new data
                updatePresentationCoachFeedback(poseData);
            })
            .catch(error => {
                console.error('Pose analysis error:', error);
                poseAnalysisContent.innerHTML = `
                    <div class="analysis-placeholder">
                        <i class="fas fa-exclamation-triangle"></i>
                        <p>Pose analysis failed: ${error.message}</p>
                    </div>
                `;
                voiceAnalysisContent.innerHTML = `
                    <div class="analysis-placeholder">
                        <i class="fas fa-exclamation-triangle"></i>
                        <p>Voice analysis failed: ${error.message}</p>
                    </div>
                `;
            });
        }

        // Show results
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

        // Function to display corrected transcript with highlighted grammar corrections
    function displayCorrectedTranscriptWithHighlighting(elementId, originalText, correctedText) {
        const element = document.getElementById(elementId);
        if (!element || !originalText || !correctedText) return;

        // If the texts are identical, add a message indicating no grammar corrections were needed
        if (originalText === correctedText) {
            // Remove any existing tooltips first
            const existingTooltips = element.parentNode.querySelectorAll('.grammar-info-tooltip');
            existingTooltips.forEach(tooltip => tooltip.remove());
            
            // Create an information message
            const infoTooltip = document.createElement('div');
            infoTooltip.className = 'grammar-info-tooltip';
            infoTooltip.innerHTML = '<i class="fas fa-check-circle"></i> No grammar corrections needed - the text is grammatically correct';
            
            // Add the message before the element
            element.parentNode.insertBefore(infoTooltip, element);
            
            // Set the content
            element.innerHTML = correctedText;
            return;
        }

        // Simple tokenization by splitting on spaces and punctuation
        function tokenize(text) {
            // Replace commas, periods, etc. with a space before them to ensure proper splitting
            const preparedText = text.replace(/([.,!?;:])/g, ' $1');
            // Split by spaces and filter out empty strings
            return preparedText.split(/\s+/).filter(token => token);
        }

        const originalTokens = tokenize(originalText);
        const correctedTokens = tokenize(correctedText);

        // Use the Levenshtein distance algorithm to find the minimum edit operations
        const m = originalTokens.length;
        const n = correctedTokens.length;

        // Initialize the dp matrix
        const dp = Array(m + 1).fill().map(() => Array(n + 1).fill(0));
        
        // Fill the first row and column
        for (let i = 0; i <= m; i++) dp[i][0] = i;
        for (let j = 0; j <= n; j++) dp[0][j] = j;
        
        // Fill the dp matrix
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (originalTokens[i - 1].toLowerCase() === correctedTokens[j - 1].toLowerCase()) {
                    dp[i][j] = dp[i - 1][j - 1]; // No operation needed
                } else {
                    dp[i][j] = Math.min(
                        dp[i - 1][j] + 1,      // Deletion
                        dp[i][j - 1] + 1,      // Insertion
                        dp[i - 1][j - 1] + 1   // Substitution
                    );
                }
            }
        }
        
        // Backtrack to find the operations
        let i = m;
        let j = n;
        const operations = [];
        
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && originalTokens[i - 1].toLowerCase() === correctedTokens[j - 1].toLowerCase()) {
                // No change
                operations.unshift({ 
                    type: 'keep', 
                    value: correctedTokens[j - 1]
                });
                i--;
                j--;
            } else if (j > 0 && (i === 0 || dp[i][j] === dp[i][j - 1] + 1)) {
                // Insertion
                operations.unshift({ 
                    type: 'insert', 
                    value: correctedTokens[j - 1]
                });
                j--;
            } else if (i > 0 && (j === 0 || dp[i][j] === dp[i - 1][j] + 1)) {
                // Deletion
                operations.unshift({ 
                    type: 'delete', 
                    value: originalTokens[i - 1]
                });
                i--;
            } else {
                // Substitution
                operations.unshift({ 
                    type: 'substitute', 
                    oldValue: originalTokens[i - 1], 
                    newValue: correctedTokens[j - 1]
                });
                i--;
                j--;
            }
        }
        
        // Build the HTML with highlights
        let html = '';
        for (const op of operations) {
            if (op.type === 'keep') {
                html += op.value + ' ';
            } else if (op.type === 'insert') {
                html += `<span class="grammar-correction grammar-insertion">${op.value}</span> `;
            } else if (op.type === 'delete') {
                // Skip deleted tokens from the original text
                continue;
            } else if (op.type === 'substitute') {
                html += `<span class="grammar-correction grammar-substitution" title="Original: '${op.oldValue}'">${op.newValue}</span> `;
            }
        }
        
        // Set the HTML content
        element.innerHTML = html;
        
        // Remove any existing tooltips before adding a new one
        const existingTooltips = element.parentNode.querySelectorAll('.grammar-info-tooltip');
        existingTooltips.forEach(tooltip => tooltip.remove());
        
        // Add a tooltip explaining the highlighting
        const infoTooltip = document.createElement('div');
        infoTooltip.className = 'grammar-info-tooltip';
        infoTooltip.innerHTML = '<i class="fas fa-info-circle"></i> Text in <span class="grammar-correction-example">red</span> shows grammar corrections';
        element.parentNode.insertBefore(infoTooltip, element);
    }

    function displayError(message) {
        const errorMessageEl = document.getElementById('errorMessage');
        
        // Check for common error patterns and provide helpful guidance
        let helpfulMessage = message;
        
        if (message.includes("Transcription failed")) {
            helpfulMessage = `${message}
            
            Possible solutions:
            1. Check that the video has clear audio
            2. Try a smaller or shorter video file
            3. Make sure ffmpeg is installed on the server
            4. Check server logs for more details`;
        } else if (message.includes("Processing failed")) {
            helpfulMessage = `${message}
            
            Possible solutions:
            1. The video may be too large - try a smaller file
            2. The server might be out of memory - restart it
            3. Check that the video format is supported`;
        }
        
        // Handle different types of errors with more context
        errorMessageEl.innerHTML = `<p><strong>Error:</strong> ${helpfulMessage}</p>
                                   <p>If the problem persists, please try with a different file or contact support.</p>`;
        errorSection.style.display = 'block';
        errorSection.scrollIntoView({ behavior: 'smooth' });
    }
});