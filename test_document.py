#!/usr/bin/env python3
"""
Create a test PDF document for testing the Smart Audiobook Reader
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os

def create_test_pdf(filename="test_document.pdf"):
    """Create a test PDF document with sample content."""
    
    # Create document
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = styles['Title']
    story.append(Paragraph("The Smart Audiobook Reader Test Document", title_style))
    story.append(Spacer(1, 12))
    
    # Sample content
    content_style = styles['Normal']
    
    content = """
    Chapter 1: Introduction to Smart Reading
    
    The Smart Audiobook Reader represents a breakthrough in document accessibility and 
    convenience. By leveraging advanced artificial intelligence and neural text-to-speech 
    technology, it transforms static PDF documents into dynamic, engaging audio experiences.
    
    This innovative approach combines the power of large language models for text 
    processing with state-of-the-art voice synthesis. The result is a natural, 
    human-like reading experience that makes consuming content more accessible than ever.
    
    Chapter 2: Technology Overview
    
    At the heart of this system lies Ollama, a platform for running large language models 
    locally. This ensures privacy and reduces latency by processing text enhancement on 
    your own machine. The system uses the Llama 3.2 model, optimized for efficiency and 
    performance on modern hardware.
    
    For voice generation, we employ Piper, a fast neural text-to-speech system that 
    produces remarkably natural speech. Piper's neural networks create voice patterns 
    that closely mimic human speech patterns, including proper intonation, pacing, and 
    emotional expression.
    
    Chapter 3: Intel Core Ultra 7 Optimization
    
    This application is specifically optimized for Intel Core Ultra 7 processors, 
    featuring the revolutionary Meteor Lake architecture. The optimization takes 
    advantage of several key features:
    
    First, the integrated Intel Arc Graphics provides powerful GPU acceleration for 
    neural network computations. This significantly speeds up both the language model 
    processing and text-to-speech generation.
    
    Second, the hybrid architecture of Performance-cores and Efficient-cores allows 
    for intelligent workload distribution. Heavy computational tasks are assigned to 
    P-cores while background processes run efficiently on E-cores.
    
    Third, the advanced memory controller and high-speed RAM interfaces ensure 
    smooth data flow between the processor, GPU, and memory subsystems.
    
    Chapter 4: User Experience
    
    The user experience is designed to be seamless and intuitive. Simply provide a 
    PDF file, and the system handles everything else automatically. The text is 
    extracted, processed for optimal readability, converted to natural speech, and 
    played back immediately.
    
    The intelligent chunking system ensures that even large documents are processed 
    efficiently without overwhelming system resources. Each chunk is processed 
    independently, allowing for concurrent operations and optimal resource utilization.
    
    Chapter 5: Future Developments
    
    Looking forward, we plan to expand language support, add more voice options, 
    and implement additional customization features. The modular architecture allows 
    for easy integration of new models and technologies as they become available.
    
    We're also exploring real-time processing capabilities, which would enable 
    live transcription and narration of streaming content. This could revolutionize 
    how we interact with digital information in real-time scenarios.
    
    Conclusion
    
    The Smart Audiobook Reader represents more than just a convenience tool; 
    it's a gateway to making information more accessible and engaging for everyone. 
    By combining cutting-edge AI technology with thoughtful optimization for modern 
    hardware, we've created a solution that's both powerful and practical.
    
    Whether you're a student looking to review course materials during your commute, 
    a professional catching up on industry reports, or someone who simply prefers 
    auditory learning, this system opens up new possibilities for how we consume 
    written content in our daily lives.
    """
    
    # Split content into paragraphs and add to story
    paragraphs = content.split('\n\n')
    for para in paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), content_style))
            story.append(Spacer(1, 6))
    
    # Build the PDF
    doc.build(story)
    print(f"✅ Test PDF created: {filename}")
    return filename

if __name__ == "__main__":
    create_test_pdf()