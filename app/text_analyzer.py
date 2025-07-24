"""
Text analysis module using HuggingFace Transformers for the AI Video Transcriber.
This module provides sentiment analysis, summarization evaluation, and content assessment.
"""

from transformers import pipeline
import numpy as np
import re
from typing import Dict, List, Any, Tuple

class EmotionAnalyzer:
    """
    A class for analyzing emotions in text using keyword matching and pattern recognition.
    Maps emotions to the Extended Emotion Categories for NLP & Social Media Models.
    """
    
    def __init__(self):
        """Initialize emotion keywords and mappings"""
        self.emotion_keywords = {
            'Joy / Happiness': {
                'keywords': ['happy', 'joy', 'excited', 'cheerful', 'delighted', 'pleased', 'glad', 
                           'wonderful', 'amazing', 'fantastic', 'great', 'awesome', 'love', 'enjoy', 
                           'fun', 'laugh', 'smile', 'celebrate', 'brilliant', 'excellent', 'perfect',
                           'thrilled', 'ecstatic', 'elated', 'blissful', 'overjoyed'],
                'emoji': '😊'
            },
            'Sadness': {
                'keywords': ['sad', 'grief', 'disappointed', 'lonely', 'regret', 'sorry', 'hurt', 
                           'pain', 'cry', 'tears', 'upset', 'down', 'depressed', 'miserable', 
                           'melancholy', 'heartbroken', 'devastated', 'gloomy', 'sorrow', 'despair'],
                'emoji': '😢'
            },
            'Anger': {
                'keywords': ['angry', 'frustrated', 'rage', 'annoyed', 'mad', 'furious', 'hate', 
                           'irritated', 'pissed', 'outraged', 'hostile', 'aggressive', 'livid', 
                           'infuriated', 'enraged', 'resentful', 'bitter', 'indignant'],
                'emoji': '😠'
            },
            'Fear': {
                'keywords': ['afraid', 'scared', 'worried', 'nervous', 'panic', 'anxious', 'terrified', 
                           'frightened', 'concern', 'stress', 'dread', 'horror', 'apprehensive', 
                           'timid', 'fearful', 'alarmed', 'uneasy', 'troubled'],
                'emoji': '😰'
            },
            'Surprise': {
                'keywords': ['surprised', 'shocked', 'amazed', 'wow', 'incredible', 'unbelievable', 
                           'unexpected', 'astonished', 'stunned', 'bewildered', 'startled', 'astounded',
                           'flabbergasted', 'speechless', 'mind-blown', 'remarkable'],
                'emoji': '😲'
            },
            'Disgust': {
                'keywords': ['disgusted', 'gross', 'awful', 'terrible', 'horrible', 'nasty', 'sick', 
                           'revolting', 'appalled', 'repulsed', 'vile', 'loathsome', 'abhorrent',
                           'repugnant', 'offensive', 'distasteful'],
                'emoji': '🤢'
            },
            'Trust': {
                'keywords': ['trust', 'confident', 'reliable', 'believe', 'faith', 'sure', 'certain', 
                           'dependable', 'loyal', 'honest', 'sincere', 'genuine', 'credible',
                           'trustworthy', 'faithful', 'devoted'],
                'emoji': '🤝'
            },
            'Anticipation': {
                'keywords': ['hope', 'expect', 'anticipate', 'curious', 'interested', 'eager', 
                           'looking forward', 'excited', 'optimistic', 'expectant', 'keen',
                           'enthusiastic', 'anticipating', 'awaiting', 'yearning'],
                'emoji': '🤔'
            },
            'Love': {
                'keywords': ['love', 'affection', 'romance', 'intimate', 'adore', 'cherish', 'treasure',
                           'devoted', 'passionate', 'caring', 'tender', 'warm', 'compassionate',
                           'affectionate', 'loving', 'dear', 'beloved'],
                'emoji': '❤️'
            },
            'Optimism': {
                'keywords': ['optimistic', 'hopeful', 'positive', 'confident', 'bright', 'promising',
                           'encouraging', 'upbeat', 'cheerful', 'buoyant', 'constructive',
                           'favorable', 'promising', 'rosy', 'sunny'],
                'emoji': '🌟'
            },
            'Pessimism': {
                'keywords': ['pessimistic', 'hopeless', 'negative', 'bleak', 'grim', 'dark', 'doubtful',
                           'cynical', 'despairing', 'gloomy', 'defeatist', 'discouraging',
                           'dismal', 'foreboding', 'ominous'],
                'emoji': '😔'
            },
            'Neutral': {
                'keywords': ['neutral', 'okay', 'fine', 'normal', 'average', 'standard', 'typical',
                           'ordinary', 'regular', 'usual', 'moderate', 'balanced'],
                'emoji': '😐'
            }
        }
        
    def analyze_emotions(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotions in the given text and return the dominant emotion with emoji
        
        Args:
            text: The text to analyze for emotions
            
        Returns:
            Dictionary containing emotion analysis results
        """
        if not text:
            return {
                "dominant_emotion": "Neutral",
                "emoji": "😐",
                "confidence": 0,
                "emotion_scores": {},
                "detected_keywords": []
            }
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        
        # Score each emotion based on keyword matches
        emotion_scores = {}
        detected_keywords = []
        
        for emotion, data in self.emotion_keywords.items():
            score = 0
            emotion_keywords = []
            
            for keyword in data['keywords']:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    score += matches
                    emotion_keywords.extend([keyword] * matches)
            
            emotion_scores[emotion] = score
            if emotion_keywords:
                detected_keywords.extend([(emotion, kw) for kw in emotion_keywords])
        
        # Find dominant emotion
        if not any(emotion_scores.values()):
            dominant_emotion = "Neutral"
            confidence = 0
        else:
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            total_keywords = sum(emotion_scores.values())
            confidence = round((emotion_scores[dominant_emotion] / total_keywords) * 100, 1) if total_keywords > 0 else 0
        
        # Get emoji for dominant emotion
        emoji = self.emotion_keywords[dominant_emotion]['emoji']
        
        return {
            "dominant_emotion": dominant_emotion,
            "emoji": emoji,
            "confidence": confidence,
            "emotion_scores": emotion_scores,
            "detected_keywords": detected_keywords[:10]  # Limit to top 10 for readability
        }

class TextAnalyzer:
    """
    A class for analyzing transcribed text using HuggingFace models to extract:
    - Sentiment analysis
    - Key points assessment
    - Strength/improvement metrics
    - Emotion analysis
    """
    
    def __init__(self):
        """Initialize the text analysis pipelines"""
        print("Loading HuggingFace text analysis models...")
        
        # Initialize sentiment analysis pipeline
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            print("✅ Sentiment analysis model loaded")
        except Exception as e:
            print(f"❌ Error loading sentiment model: {str(e)}")
            self.sentiment_analyzer = None
            
        # Initialize text classification for content quality
        try:
            self.quality_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                return_all_scores=True
            )
            print("✅ Content quality model loaded")
        except Exception as e:
            print(f"❌ Error loading quality model: {str(e)}")
            self.quality_classifier = None
            
        # Initialize emotion analyzer
        self.emotion_analyzer = EmotionAnalyzer()
        print("✅ Emotion analyzer loaded")
        
    def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on the transcript text
        
        Args:
            transcript: The text transcript to analyze
            
        Returns:
            Dictionary with various analysis results
        """
        # Initialize results dictionary
        results = {
            "sentiment_analysis": self._analyze_sentiment(transcript),
            "content_assessment": self._assess_content(transcript),
            "strengths_improvements": self._identify_strengths_improvements(transcript),
            "emotion_analysis": self._analyze_emotions(transcript)
        }
        
        return results
        
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the transcript
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment scores and interpretation
        """
        if not self.sentiment_analyzer or not text:
            return {
                "overall_sentiment": "unknown",
                "positive_score": 0,
                "negative_score": 0,
                "neutral_score": 0,
                "confidence": 0
            }
        
        # Break text into chunks if too long (models typically have token limits)
        chunks = self._split_into_chunks(text, max_length=500)
        
        # Process each chunk
        positive_scores = []
        negative_scores = []
        for chunk in chunks:
            try:
                # The sentiment analyzer with return_all_scores=True returns a list of dictionaries with scores for each label
                # Structure: [{'label': 'LABEL1', 'score': X.X}, {'label': 'LABEL2', 'score': X.X}]
                result = self.sentiment_analyzer(chunk)
                if result:
                    # First item is for the chunk, which contains a list of label/score pairs
                    for score_data in result[0]:
                        if score_data['label'] == 'POSITIVE':
                            positive_scores.append(score_data['score'])
                        elif score_data['label'] == 'NEGATIVE':
                            negative_scores.append(score_data['score'])
            except Exception as e:
                print(f"Error in sentiment analysis: {str(e)}")
                continue
                
        # If we have no results, return default values
        if not positive_scores and not negative_scores:
            return {
                "overall_sentiment": "unknown",
                "positive_score": 0,
                "negative_score": 0,
                "neutral_score": 0,
                "confidence": 0
            }
            
        # Calculate average scores
        pos_score = sum(positive_scores) / max(len(positive_scores), 1)
        neg_score = sum(negative_scores) / max(len(negative_scores), 1)
        
        # Determine overall sentiment
        if pos_score > neg_score:
            overall = "positive"
            confidence = pos_score
        else:
            overall = "negative"
            confidence = neg_score
            
        return {
            "overall_sentiment": overall,
            "positive_score": round(pos_score * 100, 1),
            "negative_score": round(neg_score * 100, 1),
            "neutral_score": round((1 - (pos_score + neg_score)) * 100, 1) if (pos_score + neg_score) < 1 else 0,
            "confidence": round(confidence * 100, 1)
        }
        
    def _assess_content(self, text: str) -> Dict[str, Any]:
        """
        Assess the content quality of the transcript
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with content assessment metrics
        """
        # Perform text analysis based on characteristics
        word_count = len(text.split())
        avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
        
        # Calculate vocabulary diversity (unique words / total words)
        words = [word.lower() for word in re.findall(r'\b[a-zA-Z]+\b', text)]
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / max(1, len(words))
        
        # Assess based on sentence structure
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        # Calculate "clarity score" - a heuristic for readability
        # Lower avg sentence length (10-20 words) is more readable
        clarity_factor = 1.0 if (10 <= avg_sentence_length <= 20) else 0.7
        
        # Calculate overall content quality score (0-100)
        quality_score = min(100, max(0, 
            40 * min(1.0, vocabulary_diversity * 2) +  # Vocabulary diversity (40%)
            30 * min(1.0, clarity_factor) +            # Clarity factor (30%)
            30 * min(1.0, min(word_count, 500) / 500)  # Length factor (30%)
        ))
        
        return {
            "quality_score": round(quality_score, 1),
            "vocabulary_diversity": round(vocabulary_diversity * 100, 1),
            "clarity_score": round(clarity_factor * 100, 1),
            "complexity_level": self._determine_complexity_level(avg_sentence_length, avg_word_length)
        }
        
    def _determine_complexity_level(self, avg_sentence_length: float, avg_word_length: float) -> str:
        """Determine text complexity level based on sentence and word length"""
        if avg_sentence_length > 25 and avg_word_length > 5.5:
            return "Advanced"
        elif avg_sentence_length > 18 and avg_word_length > 5.0:
            return "Intermediate"
        else:
            return "Basic"
            
    def _identify_strengths_improvements(self, text: str) -> Dict[str, Any]:
        """
        Identify strengths and areas for improvement in the transcript
        with detailed analysis and precise percentage scoring
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with strengths and improvement suggestions with detailed metrics
        """
        strengths = []
        improvements = []
        detailed_metrics = {}
        
        # Check word count/length
        word_count = len(text.split())
        min_ideal_length = 300
        max_ideal_length = 1500
        
        # Calculate content length score (0-100)
        if word_count <= min_ideal_length:
            length_score = (word_count / min_ideal_length) * 100
        elif word_count <= max_ideal_length:
            length_score = 100
        else:
            # Gradually reduce score for extremely lengthy content
            length_score = max(70, 100 - ((word_count - max_ideal_length) / 1000) * 30)
            
        detailed_metrics["length_score"] = round(length_score, 1)
        
        if word_count > min_ideal_length:
            strengths.append({
                "area": "Content Length", 
                "description": f"Good content length with {word_count} words",
                "score": detailed_metrics["length_score"]
            })
        else:
            target_words = min_ideal_length - word_count
            improvements.append({
                "area": "Content Length", 
                "description": f"Consider adding approximately {target_words} more words for better depth",
                "score": detailed_metrics["length_score"]
            })
            
        # Check vocabulary diversity
        words = [word.lower() for word in re.findall(r'\b[a-zA-Z]+\b', text)]
        unique_word_ratio = len(set(words)) / max(1, len(words))
        
        # Calculate vocabulary diversity score (0-100)
        # Ideal unique word ratio is typically 0.4-0.6 for engaging content
        if unique_word_ratio < 0.3:
            vocab_score = (unique_word_ratio / 0.3) * 70
        elif unique_word_ratio <= 0.6:
            vocab_score = 70 + ((unique_word_ratio - 0.3) / 0.3) * 30
        else:
            vocab_score = 100
            
        detailed_metrics["vocabulary_score"] = round(vocab_score, 1)
        detailed_metrics["unique_word_ratio"] = round(unique_word_ratio * 100, 1)
        
        if unique_word_ratio > 0.4:
            strengths.append({
                "area": "Vocabulary", 
                "description": f"Strong vocabulary diversity with {detailed_metrics['unique_word_ratio']}% unique words",
                "score": detailed_metrics["vocabulary_score"]
            })
        else:
            improvements.append({
                "area": "Vocabulary", 
                "description": f"Aim for at least 40% unique words (currently {detailed_metrics['unique_word_ratio']}%)",
                "score": detailed_metrics["vocabulary_score"]
            })
            
        # Check for filler words with more comprehensive analysis
        filler_words = ['um', 'uh', 'like', 'you know', 'sort of', 'kind of', 'basically', 'actually', 'literally', 'just']
        filler_counts = {filler: text.lower().count(f" {filler} ") for filler in filler_words}
        filler_count = sum(filler_counts.values())
        filler_ratio = filler_count / max(1, word_count)
        
        # Calculate fluency score (0-100)
        if filler_ratio >= 0.05:
            fluency_score = max(0, 100 - (filler_ratio - 0.05) * 1000)
        elif filler_ratio >= 0.02:
            fluency_score = 80 + (0.05 - filler_ratio) * 400
        else:
            fluency_score = 100
            
        detailed_metrics["fluency_score"] = round(fluency_score, 1)
        detailed_metrics["filler_word_percentage"] = round(filler_ratio * 100, 2)
        
        # Find most common filler words
        top_fillers = sorted(filler_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_fillers = [f for f, count in top_fillers if count > 0]
        
        if filler_ratio < 0.02:
            strengths.append({
                "area": "Fluency", 
                "description": "Excellent speaking fluency with minimal filler words",
                "score": detailed_metrics["fluency_score"]
            })
        else:
            filler_suggestion = f"Reduce filler words ({', '.join(top_fillers) if top_fillers else 'general fillers'})"
            improvements.append({
                "area": "Fluency", 
                "description": f"{filler_suggestion} which make up {detailed_metrics['filler_word_percentage']}% of content",
                "score": detailed_metrics["fluency_score"]
            })
            
        # Check sentence structure and readability
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, sentence_count)
        
        # Calculate sentence structure score (0-100)
        if 12 <= avg_sentence_length <= 20:
            # Ideal range
            sentence_score = 100
        elif 8 <= avg_sentence_length < 12 or 20 < avg_sentence_length <= 25:
            # Acceptable but not ideal
            distance = min(abs(avg_sentence_length - 12), abs(avg_sentence_length - 20))
            sentence_score = 85 - (distance * 3)
        else:
            # Too short or too long
            sentence_score = max(50, 70 - abs(avg_sentence_length - 16) * 2)
            
        detailed_metrics["sentence_score"] = round(sentence_score, 1)
        detailed_metrics["avg_sentence_length"] = round(avg_sentence_length, 1)
        detailed_metrics["sentence_count"] = sentence_count
        
        if 12 <= avg_sentence_length <= 20:
            strengths.append({
                "area": "Sentence Structure", 
                "description": f"Well-balanced sentences averaging {detailed_metrics['avg_sentence_length']} words each",
                "score": detailed_metrics["sentence_score"]
            })
        elif avg_sentence_length > 25:
            improvements.append({
                "area": "Sentence Structure", 
                "description": f"Shorten sentences from current average of {detailed_metrics['avg_sentence_length']} words to 15-20 words",
                "score": detailed_metrics["sentence_score"]
            })
        elif avg_sentence_length < 8:
            improvements.append({
                "area": "Sentence Structure", 
                "description": f"Combine short sentences to reach an average of 12-20 words (currently {detailed_metrics['avg_sentence_length']})",
                "score": detailed_metrics["sentence_score"]
            })
        
        # Add transition words analysis
        transition_words = ['however', 'therefore', 'consequently', 'furthermore', 'moreover', 'nevertheless',
                           'thus', 'meanwhile', 'subsequently', 'alternatively', 'specifically', 'similarly']
        transition_count = sum(text.lower().count(f" {word} ") for word in transition_words)
        transition_ratio = transition_count / max(1, sentence_count)
        
        # Calculate transition score (0-100)
        if transition_ratio >= 0.2:
            transition_score = 100
        else:
            transition_score = (transition_ratio / 0.2) * 100
            
        detailed_metrics["transition_score"] = round(transition_score, 1)
        detailed_metrics["transition_word_count"] = transition_count
        
        if transition_ratio >= 0.15:
            strengths.append({
                "area": "Flow & Cohesion", 
                "description": "Good use of transition words to connect ideas",
                "score": detailed_metrics["transition_score"]
            })
        elif sentence_count > 5:  # Only suggest if there are enough sentences
            improvements.append({
                "area": "Flow & Cohesion", 
                "description": "Add more transition words to improve flow between ideas",
                "score": detailed_metrics["transition_score"]
            })
            
        # Calculate overall scores - weighted average of component scores
        # Define weights for different aspects
        weights = {
            "length_score": 0.15,
            "vocabulary_score": 0.25,
            "fluency_score": 0.25,
            "sentence_score": 0.25,
            "transition_score": 0.10
        }
        
        weighted_score = sum(weights[metric] * score for metric, score in detailed_metrics.items() if metric in weights)
        
        # Calculate strength and improvement scores based on the weighted components
        strength_score = round(weighted_score, 1)
        
        # Improvement score is inversely related but not just (100 - strength_score)
        # A high strength score still leaves room for specific improvements
        improvement_score = round(max(40, min(95, 100 - (len(improvements) * 15))), 1)
        
        # Ensure scores are not unexpectedly low
        strength_score = max(strength_score, len(strengths) * 20)
        
        return {
            "strengths": strengths,
            "improvements": improvements,
            "strength_score": strength_score,
            "improvement_areas_score": improvement_score,
            "detailed_metrics": detailed_metrics
        }
        
    def _analyze_emotions(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotions in the given text and return the dominant emotion with emoji
        
        Args:
            text: The text to analyze for emotions
            
        Returns:
            Dictionary containing emotion analysis results
        """
        if not text:
            return {
                "dominant_emotion": "Neutral",
                "emoji": "😐",
                "confidence": 0,
                "emotion_scores": {},
                "detected_keywords": []
            }
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        
        # Score each emotion based on keyword matches
        emotion_scores = {}
        detected_keywords = []
        
        for emotion, data in self.emotion_analyzer.emotion_keywords.items():
            score = 0
            emotion_keywords = []
            
            for keyword in data['keywords']:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    score += matches
                    emotion_keywords.extend([keyword] * matches)
            
            emotion_scores[emotion] = score
            if emotion_keywords:
                detected_keywords.extend([(emotion, kw) for kw in emotion_keywords])
        
        # Find dominant emotion
        if not any(emotion_scores.values()):
            dominant_emotion = "Neutral"
            confidence = 0
        else:
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            total_keywords = sum(emotion_scores.values())
            confidence = round((emotion_scores[dominant_emotion] / total_keywords) * 100, 1) if total_keywords > 0 else 0
        
        # Get emoji for dominant emotion
        emoji = self.emotion_analyzer.emotion_keywords[dominant_emotion]['emoji']
        
        return {
            "dominant_emotion": dominant_emotion,
            "emoji": emoji,
            "confidence": confidence,
            "emotion_scores": emotion_scores,
            "detected_keywords": detected_keywords[:10]  # Limit to top 10 for readability
        }

    def _split_into_chunks(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into chunks of approximately equal size"""
        words = text.split()
        if len(words) <= max_length:
            return [text]
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + 1 > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = 1
            else:
                current_chunk.append(word)
                current_length += 1
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
