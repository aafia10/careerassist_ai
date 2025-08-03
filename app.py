import streamlit as st
import PyPDF2
from openai import OpenAI
import io
import json
import os
from datetime import datetime
import re
from typing import List, Dict
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure page
st.set_page_config(
    page_title="📚 EduInsights - PDF Intelligence for Learning",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #f8f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f5f7ff 0%, #e8f2ff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #d1e7dd;
        margin: 1rem 0;
    }
    
    .question-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    
    .answer-box {
        background: #d1e7dd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #198754;
        margin: 0.5rem 0;
    }
    
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class PDFInsightsApp:
    def __init__(self):
        self.setup_openai()
        
    def setup_openai(self):
        """Setup OpenAI client"""
        # Only use environment variable for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            st.session_state.openai_api_key = api_key
        else:
            st.session_state.openai_api_key = None
            
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 3000) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1
            if current_size > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def call_gpt(self, prompt: str, api_key: str) -> str:
        """Make API call to GPT-3.5-turbo using new OpenAI client"""
        try:
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an educational AI assistant specializing in document analysis and creating study materials for students and teachers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error calling GPT API: {str(e)}")
            return ""
    
    def generate_summary(self, text: str, api_key: str, user_type: str) -> str:
        """Generate comprehensive summary"""
        prompt = f"""
        As an educational AI assistant, create a comprehensive summary of the following text for {user_type}.
        
        Please provide:
        1. Main Topic Overview
        2. Key Concepts and Ideas
        3. Important Details
        4. Educational Value
        
        Text to summarize:
        {text[:4000]}
        
        Make the summary detailed but accessible for educational purposes.
        """
        
        return self.call_gpt(prompt, api_key)
    
    def extract_key_points(self, text: str, api_key: str) -> str:
        """Extract key points for study"""
        prompt = f"""
        Extract the most important key points from this educational content. Format as a clear, organized list:
        
        Text:
        {text[:4000]}
        
        Provide 8-12 key points that capture the essential information for study purposes.
        Format each point clearly and concisely.
        """
        
        return self.call_gpt(prompt, api_key)
    
    def generate_study_questions(self, text: str, api_key: str) -> str:
        """Generate study questions"""
        prompt = f"""
        Create 10 educational study questions based on this content. Include:
        - 4 factual/recall questions
        - 3 analytical/understanding questions  
        - 3 application/critical thinking questions
        
        Text:
        {text[:4000]}
        
        Format as numbered questions suitable for student assessment.
        """
        
        return self.call_gpt(prompt, api_key)
    
    def answer_question(self, question: str, context: str, api_key: str) -> str:
        """Answer specific question about the PDF"""
        prompt = f"""
        Based on the provided context, answer the following question clearly and comprehensively:
        
        Question: {question}
        
        Context:
        {context[:3500]}
        
        Provide a detailed, educational answer. If the answer isn't directly in the context, say so and provide the best related information available.
        """
        
        return self.call_gpt(prompt, api_key)
    
    def create_teaching_notes(self, text: str, api_key: str) -> str:
        """Generate teaching notes for educators"""
        prompt = f"""
        Create comprehensive teaching notes for educators based on this content. Include:
        
        1. Learning Objectives
        2. Key Teaching Points
        3. Discussion Questions
        4. Activity Suggestions
        5. Assessment Ideas
        
        Content:
        {text[:4000]}
        
        Format professionally for classroom use.
        """
        
        return self.call_gpt(prompt, api_key)

def main():
    app = PDFInsightsApp()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 EduInsights - PDF Intelligence for Learning</h1>
        <p>Transform any PDF into comprehensive study materials and insights using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Check if API key is set via environment
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            st.success("✅ OpenAI API Key configured!")
        else:
            st.error("❌ OpenAI API Key not found!")
            st.info("💡 Please set OPENAI_API_KEY in your .env file")
        
        st.header("👥 User Type")
        user_type = st.radio(
            "I am a:",
            ["Student", "Teacher", "Researcher"],
            help="This helps customize the analysis for your needs"
        )
        
        st.header("📋 Features")
        st.markdown("""
        - **Smart Summarization**
        - **Key Points Extraction** 
        - **Study Questions**
        - **Q&A System**
        - **Teaching Notes**
        - **Educational Insights**
        """)
    
    # Main content area
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("🔑 **OpenAI API Key Required**")
        st.markdown("""
        <div class="feature-card">
            <h3>⚙️ Setup Instructions</h3>
            <p><strong>1.</strong> Create a <code>.env</code> file in your project directory</p>
            <p><strong>2.</strong> Add your OpenAI API key:</p>
            <pre><code>OPENAI_API_KEY=your_openai_api_key_here</code></pre>
            <p><strong>3.</strong> Restart the application</p>
            <br>
            <p><strong>Get your API key:</strong> <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI API Keys</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Demo information
        st.markdown("### 🎯 What you can do with EduInsights:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🎯 For Students</h3>
                <p>Get summaries, key points, and study questions from any PDF material</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>👩‍🏫 For Teachers</h3>
                <p>Generate teaching notes, discussion questions, and assessment materials</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h3>🔍 For Researchers</h3>
                <p>Extract insights, analyze content, and get detailed summaries</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # File upload
    st.header("📄 Upload Your PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any educational PDF document for analysis"
    )
    
    if uploaded_file is not None:
        # Extract text
        with st.spinner("🔍 Extracting text from PDF..."):
            pdf_text = app.extract_text_from_pdf(uploaded_file)
        
        if pdf_text:
            # Display PDF stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="stats-card">
                    <h3>📊 Words</h3>
                    <h2>{}</h2>
                </div>
                """.format(len(pdf_text.split())), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="stats-card">
                    <h3>📝 Characters</h3>
                    <h2>{}</h2>
                </div>
                """.format(len(pdf_text)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="stats-card">
                    <h3>📄 File Size</h3>
                    <h2>{:.1f} KB</h2>
                </div>
                """.format(uploaded_file.size / 1024), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="stats-card">
                    <h3>👤 Mode</h3>
                    <h2>{}</h2>
                </div>
                """.format(user_type), unsafe_allow_html=True)
            
            # Analysis tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Summary", "🎯 Key Points", "❓ Study Questions", "💬 Q&A", "📚 Teaching Notes"
            ])
            
            with tab1:
                st.header("📋 Comprehensive Summary")
                
                if st.button("Generate Summary", type="primary"):
                    with st.spinner("🤖 Generating comprehensive summary..."):
                        summary = app.generate_summary(pdf_text, api_key, user_type.lower())
                    
                    if summary:
                        summary_html = summary.replace('\n', '<br>')
                        st.markdown(f"""
                        <div class="insight-box">
                            <h3>📖 Summary for {user_type}s</h3>
                            {summary_html}
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab2:
                st.header("🎯 Key Points & Concepts")
                
                if st.button("Extract Key Points", type="primary"):
                    with st.spinner("🔍 Extracting key educational points..."):
                        key_points = app.extract_key_points(pdf_text, api_key)
                    
                    if key_points:
                        key_points_html = key_points.replace('\n', '<br>')
                        st.markdown(f"""
                        <div class="insight-box">
                            <h3>🎯 Essential Key Points</h3>
                            {key_points_html}
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab3:
                st.header("❓ Study Questions")
                
                if st.button("Generate Study Questions", type="primary"):
                    with st.spinner("📝 Creating study questions..."):
                        questions = app.generate_study_questions(pdf_text, api_key)
                    
                    if questions:
                        questions_html = questions.replace('\n', '<br>')
                        st.markdown(f"""
                        <div class="insight-box">
                            <h3>📚 Study Questions</h3>
                            {questions_html}
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab4:
                st.header("💬 Ask Questions About the PDF")
                
                user_question = st.text_input(
                    "Ask any question about the PDF content:",
                    placeholder="What are the main themes discussed in this document?"
                )
                
                if st.button("Get Answer", type="primary") and user_question:
                    with st.spinner("🤔 Finding the answer..."):
                        answer = app.answer_question(user_question, pdf_text, api_key)
                    
                    if answer:
                        answer_html = answer.replace('\n', '<br>')
                        st.markdown(f"""
                        <div class="question-box">
                            <strong>❓ Question:</strong> {user_question}
                        </div>
                        <div class="answer-box">
                            <strong>💡 Answer:</strong><br>{answer_html}
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab5:
                if user_type == "Teacher":
                    st.header("📚 Teaching Notes & Materials")
                    
                    if st.button("Generate Teaching Notes", type="primary"):
                        with st.spinner("👩‍🏫 Creating teaching materials..."):
                            teaching_notes = app.create_teaching_notes(pdf_text, api_key)
                        
                        if teaching_notes:
                            teaching_notes_html = teaching_notes.replace('\n', '<br>')
                            st.markdown(f"""
                            <div class="insight-box">
                                <h3>📋 Teaching Notes & Materials</h3>
                                {teaching_notes_html}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("🎓 Teaching Notes are available when you select 'Teacher' as your user type in the sidebar.")
            
            # Footer with tips
            st.markdown("---")
            st.markdown("""
            <div class="feature-card">
                <h3>💡 Pro Tips</h3>
                <ul>
                    <li><strong>Students:</strong> Use the Summary and Key Points for quick review</li>
                    <li><strong>Teachers:</strong> Generate teaching notes and discussion questions</li>
                    <li><strong>Everyone:</strong> Use Q&A to clarify specific concepts</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.error("❌ Could not extract text from the PDF. Please try a different file.")

if __name__ == "__main__":
    main()