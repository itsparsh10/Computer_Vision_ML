import google.generativeai as genai  # Google's Gemini AI API
from typing import Dict, Optional, List  # For type hints
import re  # For regular expressions
from collections import Counter  # For counting word occurrences

class GeminiProcessor:
    """
    Enhanced class for processing video transcripts with Google's Gemini API.
    Provides improved transcript processing, grammar correction, summarization, and keyword extraction.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Gemini API client with your API key
        
        Args:
            api_key (str): Your Google Gemini API key from .env file
        """
        # Configure the Gemini API with your key
        genai.configure(api_key=api_key)
        
        # Configure the generative AI model
        self.model = genai.GenerativeModel('gemini-2.0-flash') # Use the requested model
        
        # Test the API key by making a simple request
        try:
            genai.GenerativeModel('gemini-2.0-flash').generate_content("Test")
            print("✅ Gemini API initialized successfully (gemini-2.0-flash)")
        except Exception as e:
            print(f"❌ Error initializing Gemini API: {str(e)}")
            raise RuntimeError("Gemini API initialization failed - please check your API key and model availability") from e
    
    def clean_transcript(self, transcript: str) -> str:
        """
        Clean and prepare the raw transcript for processing
        
        Args:
            transcript (str): Raw transcript text
            
        Returns:
            str: Cleaned transcript
        """
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', transcript)
        
        # Remove common transcript artifacts
        cleaned = re.sub(r'\[.*?\]', '', cleaned)  # Remove [music], [applause], etc.
        cleaned = re.sub(r'\(.*?\)', '', cleaned)  # Remove (inaudible), etc.
        
        # Fix common OCR/transcription errors
        cleaned = cleaned.replace(' um ', ' ')
        cleaned = cleaned.replace(' uh ', ' ')
        cleaned = cleaned.replace(' ah ', ' ')
        
        # Clean up punctuation spacing
        cleaned = re.sub(r'\s+([,.!?])', r'\1', cleaned)
        cleaned = re.sub(r'([,.!?])\s*', r'\1 ', cleaned)
        
        return cleaned.strip()
    
    def process_transcript(self, transcript: str, randomness_factor: float = 0.5) -> Dict[str, str]:
        """
        Process the transcript with Gemini AI in three steps:
        1. Grammar correction - Fix any grammar errors and improve readability
        2. Summary generation - Create a concise summary of the content
        3. Keyword extraction - Identify important topics as keywords
        
        Args:
            transcript (str): The raw transcript text from Whisper
            randomness_factor (float): Controls the variation in grammar correction (0.1-1.0)
                                      Higher values produce more variation between runs
            
        Returns:
            Dict with three keys: corrected_text, summary, and keywords
        """
        # Ensure randomness_factor is within valid range
        randomness = max(0.1, min(1.0, randomness_factor))
        
        # For backward compatibility, call the enhanced processing method with randomness factor
        result = self.process_video_transcript(transcript, randomness)
        
        # Adjust the keys to match what main.py expects
        return {
            "corrected_text": result.get("corrected_transcript", transcript),
            "summary": result.get("summary", "Summary not available"),
            "keywords": result.get("keywords", "Keywords not available")
        }
        
    def process_video_transcript(self, raw_transcript: str, randomness_factor: float = 0.5) -> Dict[str, str]:
        """
        Process the video transcript with enhanced Gemini AI processing:
        1. Clean the transcript
        2. Grammar correction with proper punctuation
        3. Generate structured 2-3 paragraph summary
        4. Extract relevant keywords for video content
        
        Args:
            raw_transcript (str): The raw transcript text from video
            randomness_factor (float): Controls the variation in generation (0.1-1.0)
            
        Returns:
            Dict with keys: original_transcript, corrected_transcript, summary, keywords
        """
        try:
            print("🎬 Starting enhanced video transcript processing...")
            
            # Clean the transcript first
            cleaned_transcript = self.clean_transcript(raw_transcript)
            print("✅ Transcript cleaned and prepared")
            
            # Check if transcript is too long and needs chunking
            is_long_transcript = len(cleaned_transcript) > 5000
            
            # STEP 1: Enhanced Grammar Correction
            print("📝 Step 1/3: Correcting grammar and improving readability...")
            
            # For very long transcripts, we'll skip full correction to avoid token limits
            if is_long_transcript:
                # For long transcripts, just do basic cleaning and correction
                corrected_transcript = cleaned_transcript
                print("⚠️ Long transcript detected - using cleaned version without full grammar correction")
            else:
                # Set temperature based on randomness factor - higher for more variation
                grammar_temperature = 0.2 + (randomness_factor * 0.8)
                
                # Include a style variation suggestion based on randomness factor
                style_variations = [
                    "Make it sound natural and conversational",
                    "Use a more professional and formal tone",
                    "Make the language crisp and direct",
                    "Slightly elevate the language while maintaining clarity",
                    "Maintain a friendly and accessible tone"
                ]
                
                # Use the randomness factor to pick a style variation
                import random
                random.seed()  # Ensure randomness
                style_index = random.randint(0, len(style_variations) - 1)
                style_suggestion = style_variations[style_index]
                
                grammar_prompt = f"""
                You are a professional transcript editor. Please improve this video transcript by:

                1. Correcting all grammar, spelling, and punctuation errors
                2. Adding proper capitalization and sentence structure
                3. Removing filler words and verbal tics (um, uh, like, you know)
                4. Making the text flow naturally while preserving the original meaning
                5. Adding paragraph breaks for better readability
                6. Maintaining the speaker's core ideas and key points
                7. {style_suggestion}

                Original transcript:
                {cleaned_transcript}

                Return ONLY the corrected transcript with proper formatting and paragraphs.
                """
                
                corrected_response = self.model.generate_content(grammar_prompt, 
                                                              generation_config={"temperature": grammar_temperature})
                corrected_transcript = corrected_response.text.strip()
                
                # If the resulting text is too similar to the original, try once more with higher temperature
                if self._similarity_score(cleaned_transcript, corrected_transcript) > 0.9 and randomness_factor > 0.3:
                    print("⚠️ Correction too similar to original - trying with higher variation...")
                    grammar_prompt += "\n\nPlease rephrase significantly while preserving meaning."
                    corrected_response = self.model.generate_content(grammar_prompt, 
                                                                  generation_config={"temperature": min(1.0, grammar_temperature * 1.5)})
                    corrected_transcript = corrected_response.text.strip()
            
            # STEP 2: Enhanced Summary Generation
            print("📋 Step 2/3: Generating structured summary...")
            
            # For long transcripts, use chunking for summary
            if is_long_transcript:
                chunks = self._split_into_chunks(cleaned_transcript, max_length=2000)
                print(f"⚠️ Using chunked processing for long transcript ({len(chunks)} chunks)")
                
                # Process each chunk for summary data
                chunk_summaries = []
                for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks
                    print(f"  Processing chunk {i+1}/{min(len(chunks), 3)}...")
                    
                    chunk_prompt = f"""
                    Summarize the key points from this transcript segment:
                    
                    {chunk}
                    
                    Provide 3-5 bullet points covering the main ideas in this segment.
                    """
                    
                    chunk_response = self.model.generate_content(chunk_prompt, 
                                                               generation_config={"temperature": 0.2})
                    chunk_summaries.append(chunk_response.text.strip())
                
                # Combine chunk summaries into comprehensive summary
                combined_summary = "\n\n".join(chunk_summaries)
                
                # Final pass to create numbered, structured summary
                summary_prompt = f"""
                Create a comprehensive, well-structured summary based on these key points:
                
                {combined_summary}
                
                Your summary MUST follow this EXACT format:
                
                # 📝 SUMMARY
                
                ## 🎯 Overview
                [Provide a 2-3 sentence introduction to the main topic, purpose, and context of the content.]
                
                ## 📋 Key Highlights
                • [Highlight 1 - Most important point or finding]
                • [Highlight 2 - Second most important point]
                • [Highlight 3 - Third key point]
                
                ## 🔍 Detailed Breakdown
                ### Main Topic 1
                - Key point 1
                - Key point 2
                
                ### Main Topic 2
                - Key point 1
                - Key point 2
                
                ## 💡 Key Insights
                • [Insight 1 - Important observation or analysis]
                • [Insight 2 - Another important observation]
                
                ## 🚀 Action Items/Recommendations
                • [Action 1 - Specific recommendation]
                • [Action 2 - Additional recommendation]
                
                Formatting Requirements:
                - Use clear, concise language
                - Keep bullet points to one line each
                - Use emojis as shown for visual hierarchy
                - Ensure proper markdown formatting
                - Focus on actionable insights
                - Maintain a professional yet engaging tone
                """
                
                summary_response = self.model.generate_content(summary_prompt, 
                                                             generation_config={"temperature": 0.3})
                summary = summary_response.text.strip()
                
            else:
                # Standard processing for normal-length transcripts - using numbered format
                summary_prompt = f"""
                You are a professional content strategist and summarization expert. Analyze this transcript and create a comprehensive, well-structured summary that captures the essence of the content.
                
                Transcript:
                {corrected_transcript}
                
                Your summary MUST follow this EXACT format and structure:
                
                # 📋 EXECUTIVE SUMMARY
                [A concise 2-3 sentence overview of the main topic and its significance]
                
                ## 🎯 Key Objectives
                • [Primary goal or purpose 1]
                • [Primary goal or purpose 2]
                
                ## 🔍 Core Content
                ### Main Topic 1
                - Key point or finding 1
                - Supporting detail or example
                
                ### Main Topic 2
                - Key point or finding 2
                - Supporting detail or example
                
                ## 💡 Critical Insights
                • [Important observation or analysis 1]
                • [Important observation or analysis 2]
                
                ## 🚀 Implementation/Next Steps
                • [Actionable recommendation 1]
                • [Actionable recommendation 2]
                
                Formatting Guidelines:
                - Use clear, professional language
                - Keep bullet points concise (1 line each)
                - Use emojis as shown for visual hierarchy
                - Maintain consistent markdown formatting
                - Focus on the most valuable information
                - Ensure the summary is scannable and easy to digest
                - Highlight any statistics, data points, or specific examples
                - Maintain a professional yet engaging tone
                """
                
                summary_response = self.model.generate_content(summary_prompt, 
                                                             generation_config={"temperature": 0.3})
                summary = summary_response.text.strip()
            
            # STEP 3: Enhanced Keyword Extraction (use first 2000 chars for keywords)
            print("🔍 Step 3/3: Extracting relevant keywords...")
            
            # Use beginning of transcript for keywords (most important content often at start)
            keyword_text = cleaned_transcript[:2000]
            
            keywords_prompt = f"""
            You are an expert SEO analyst and video content specialist.
            
            Extract the TOP 12 most relevant keywords and key phrases from this video transcript:
            
            {keyword_text}
            
            Focus on extracting:
            1. Main topics and themes discussed
            2. Important names, products, or brands mentioned
            3. Technical terms and concepts
            4. Industry-specific terminology
            5. Action words and processes described
            6. Location names or specific references
            
            Guidelines:
            - Return ONLY a comma-separated list
            - Mix of single words and 2-3 word phrases
            - Prioritize specific, descriptive terms over generic words
            - Avoid common words like "video", "discuss", "important"
            - Focus on what makes this video unique and searchable
            - Ensure keywords reflect the actual content discussed
            
            Format: keyword1, keyword2, keyword phrase, etc.
            """
            
            keywords_response = self.model.generate_content(keywords_prompt, 
                                                          generation_config={"temperature": 0.2})
            raw_keywords = keywords_response.text.strip()
            
            # Clean up keywords
            keywords = self._clean_keywords(raw_keywords)
            
            print("✅ Enhanced processing completed successfully!")
            
            return {
                "original_transcript": raw_transcript,
                "corrected_transcript": corrected_transcript,
                "summary": summary,
                "keywords": keywords
            }
            
        except Exception as e:
            print(f"❌ Error during processing: {str(e)}")
            return self._handle_processing_error(raw_transcript, cleaned_transcript)
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Simple similarity measure based on word sets
        # More sophisticated methods could be used (like cosine similarity)
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def _split_into_chunks(self, text: str, max_length: int = 2000) -> List[str]:
        """
        Split long text into chunks of approximately equal size
        
        Args:
            text (str): Text to split
            max_length (int): Maximum chunk length
            
        Returns:
            List[str]: List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _clean_keywords(self, raw_keywords: str) -> str:
        """
        Clean and format the extracted keywords
        
        Args:
            raw_keywords (str): Raw keywords from Gemini
            
        Returns:
            str: Cleaned, formatted keywords
        """
        # Remove common prefixes
        keywords = raw_keywords
        prefixes_to_remove = ["Keywords:", "Key phrases:", "Tags:", "Topics:"]
        for prefix in prefixes_to_remove:
            if keywords.startswith(prefix):
                keywords = keywords[len(prefix):].strip()
        
        # Remove bullet points, numbers, and special characters
        keywords = re.sub(r'^[\d\.\-\*\•]+\s*', '', keywords, flags=re.MULTILINE)
        keywords = re.sub(r'\n', ', ', keywords)
        
        # Split, clean, and rejoin
        keyword_list = []
        for keyword in keywords.split(','):
            clean_keyword = keyword.strip().strip('"').strip("'")
            if clean_keyword and len(clean_keyword) > 1:
                keyword_list.append(clean_keyword)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keyword_list:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword)
        
        return ', '.join(unique_keywords[:12])  # Limit to 12 keywords
    
    def _handle_processing_error(self, original: str, cleaned: str) -> Dict[str, str]:
        """
        Handle errors gracefully by providing robust fallback processing
        
        Args:
            original (str): Original transcript
            cleaned (str): Cleaned transcript
            
        Returns:
            Dict: Results with best-effort processing
        """
        return {
            "original_transcript": original,
            "corrected_transcript": cleaned,  # Use cleaned version as fallback
            "summary": "Summary processing failed - please try again",
            "keywords": "Keywords extraction failed - please try again"
        }
