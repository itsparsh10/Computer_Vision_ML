#!/usr/bin/env python3

# Simple test script to debug emotion analysis
from app.text_analyzer import EmotionAnalyzer

def test_emotion_analysis():
    analyzer = EmotionAnalyzer()
    
    # Test with various emotional texts
    test_texts = [
        "I am so happy and excited about this wonderful day!",
        "I feel sad and disappointed about what happened.",
        "This is absolutely amazing and fantastic!",
        "I trust you completely and have faith in your abilities.",
        "I'm really worried and anxious about the situation.",
        "What a surprising and incredible turn of events!",
        "This is usual and ordinary content about regular topics."
    ]
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        result = analyzer.analyze_emotions(text)
        print(f"Dominant Emotion: {result['dominant_emotion']} {result['emoji']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Detected Keywords: {result['detected_keywords'][:3]}")  # Show first 3
        print(f"Emotion Scores: {result['emotion_scores']}")

if __name__ == "__main__":
    test_emotion_analysis()
