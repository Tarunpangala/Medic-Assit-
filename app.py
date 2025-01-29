import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found.")
    st.stop()

# Configure Gemini models
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

class MedicalAssistant:
    def __init__(self):
        self.conversation_history = []

    def safe_generate_content(self, prompt, model_type='text', image=None):
        """Generate content with error handling."""
        try:
            if model_type == 'text':
                response = model.generate_content(prompt, safety_settings={
                    'HARASSMENT': 'BLOCK_NONE',
                    'HATE': 'BLOCK_NONE',
                    'SEXUAL': 'BLOCK_NONE',
                    'DANGEROUS': 'BLOCK_NONE'
                })
            elif model_type == 'vision':
                response = model.generate_content([prompt, image], safety_settings={
                    'HARASSMENT': 'BLOCK_NONE',
                    'HATE': 'BLOCK_NONE',
                    'SEXUAL': 'BLOCK_NONE',
                    'DANGEROUS': 'BLOCK_NONE'
                })
            return response.text
        except Exception as e:
            st.error(f"Content generation error: {e}")
            return None

    def interactive_symptom_diagnosis(self):
        """Interactive symptom diagnosis workflow with focused questioning."""
        st.title("🩺 Virtual Doctor Consultation")
        
        # Initialize session state for conversation
        if 'conversation_state' not in st.session_state:
            st.session_state.conversation_state = [
                {"role": "Virtual Doctor", "message": "Hello! I'll help diagnose your health concern through a few key questions."},
                {"role": "Virtual Doctor", "message": "What is your age?"}
            ]
            st.session_state.patient_data = {
                'age': None,
                'gender': None,
                'primary_symptom': None,
                'symptom_duration': None,
                'severity': None,
                'medical_history': None
            }
            st.session_state.analysis_complete = False

        # Display conversation history
        for msg in st.session_state.conversation_state:
            if msg['role'] == 'Virtual Doctor':
                st.markdown(f"**🩺 Doctor:** {msg['message']}")
            else:
                st.markdown(f"**👤 You:** {msg['message']}")

        # Check if analysis is complete
        if st.session_state.analysis_complete:
            analysis = self.generate_focused_medical_analysis(st.session_state.patient_data)
            st.markdown("## 📋 Medical Analysis Report")
            st.write(analysis)
            return

        # User input
        user_response = st.text_input("Your response:", key="user_input")

        if st.button("Send"):
            if user_response:
                # Add user response to conversation
                st.session_state.conversation_state.append({
                    "role": "Patient", 
                    "message": user_response
                })

                # Update patient data
                if not st.session_state.patient_data['age']:
                    st.session_state.patient_data['age'] = user_response
                elif not st.session_state.patient_data['gender']:
                    st.session_state.patient_data['gender'] = user_response
                elif not st.session_state.patient_data['primary_symptom']:
                    st.session_state.patient_data['primary_symptom'] = user_response
                elif not st.session_state.patient_data['symptom_duration']:
                    st.session_state.patient_data['symptom_duration'] = user_response
                elif not st.session_state.patient_data['severity']:
                    st.session_state.patient_data['severity'] = user_response
                elif not st.session_state.patient_data['medical_history']:
                    st.session_state.patient_data['medical_history'] = user_response
                    # When medical history is filled, mark analysis as complete
                    st.session_state.analysis_complete = True

                # Determine next step in diagnosis
                next_step = self.generate_focused_diagnostic_step(
                    st.session_state.conversation_state, 
                    st.session_state.patient_data
                )

                # Add doctor's next message
                st.session_state.conversation_state.append({
                    "role": "Virtual Doctor", 
                    "message": next_step['message']
                })
        
                # Rerun to refresh the page
                st.experimental_rerun()

    def generate_focused_diagnostic_step(self, conversation_history, patient_data):
        """Generate focused diagnostic steps with strategic questioning."""
        if not patient_data['age']:
            return {
                "type": "question",
                "message": "What is your age?"
            }
        
        if not patient_data['gender']:
            return {
                "type": "question", 
                "message": "What is your gender?"
            }
        
        if not patient_data['primary_symptom']:
            return {
                "type": "question",
                "message": "Can you describe your primary symptom in more detail? (e.g., location, type of pain/discomfort)"
            }
        
        if not patient_data['symptom_duration']:
            return {
                "type": "question",
                "message": "How long have you been experiencing this symptom? (Days, weeks, months)"
            }
        
        if not patient_data['severity']:
            return {
                "type": "question",
                "message": "On a scale of 1-10, how would you rate the severity of your symptom?"
            }
        
        if not patient_data['medical_history']:
            return {
                "type": "question",
                "message": "Do you have any pre-existing medical conditions or ongoing health issues?"
            }
        
        return {
            "type": "question",
            "message": "Thank you for providing all details. Generating medical analysis..."
        }

    def generate_focused_medical_analysis(self, patient_data):
        """Generate detailed, structured medical analysis with comprehensive report."""
        # Validate all required fields are present
        required_fields = ['age', 'gender', 'primary_symptom', 'symptom_duration', 'severity', 'medical_history']
        for field in required_fields:
            if not patient_data.get(field):
                return f"Error: Missing {field} information for analysis."

        # Construct detailed prompt for medical analysis
        prompt = f"""Comprehensive Medical Report:

Patient Profile:
- Age: {patient_data['age']} years
- Gender: {patient_data['gender']}
- Primary Symptom: {patient_data['primary_symptom']}
- Symptom Duration: {patient_data['symptom_duration']}
- Symptom Severity: {patient_data['severity']}/10
- Medical History: {patient_data['medical_history']}

Provide a detailed medical report with:
1. Definition of potential condition
2. Possible Causes
3. Types of Condition
4. Prevention Strategies
5. Top 5 Recommended Medicines
6. Top 5 Home Remedies
7. When to Consult a Doctor
8. Immediate Medical Attention Guidelines"""

        # Generate analysis using Gemini
        analysis = self.safe_generate_content(prompt)
        return analysis if analysis else "Unable to generate medical analysis."

    def medical_image_analysis(self):
        """Advanced medical image analysis with comprehensive reporting."""
        st.title("🖼 Medical Image Analysis")
        
        # Initialize session state for image analysis
        if 'image_analysis_result' not in st.session_state:
            st.session_state.image_analysis_result = None
            st.session_state.uploaded_image = None
            st.session_state.previous_questions = []
        
        uploaded_image = st.file_uploader("Upload Medical Image:", type=["jpg", "jpeg", "png"])
        
        if uploaded_image and uploaded_image != st.session_state.uploaded_image:
            st.session_state.uploaded_image = uploaded_image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Medical Image", use_column_width=True)
            
            image_analysis_prompt = """Perform comprehensive medical image analysis:
            1. Identify potential medical conditions
            2. Describe visible anatomical features
            3. Highlight any critical anomalies
            4. Provide diagnostic insights
            5. Recommend follow-up actions"""
            
            try:
                initial_response = self.safe_generate_content(
                    image_analysis_prompt, 
                    model_type='vision', 
                    image=image
                )
                
                st.session_state.image_analysis_result = initial_response
            
            except Exception as e:
                st.error(f"Image analysis error: {e}")
        
        # Display Image Analysis Result
        if st.session_state.image_analysis_result:
            st.markdown("## 🔬 Image Analysis Results")
            st.write(st.session_state.image_analysis_result)
            
            # Question Section
            st.markdown("### 💬 Ask About the Image")
            user_question = st.text_input("Ask a specific question about the medical image:", key="image_question")
            
            if st.button("Get Detailed Answer"):
                if user_question:
                    # Store question for history tracking
                    st.session_state.previous_questions.append(user_question)
                    
                    # Construct a detailed query based on previous analysis and user's question
                    detailed_query = f"""Based on the previous medical image analysis:
                    {st.session_state.image_analysis_result}

                    Specific User Question: {user_question}

                    Provide a comprehensive, medically precise answer addressing:
                    1. Direct response to the question
                    2. Medical context
                    3. Potential implications
                    4. Recommended actions"""
                    
                    # Generate answer using Gemini
                    answer = self.safe_generate_content(detailed_query)
                    
                    if answer:
                        st.markdown("## 🩺 Detailed Medical Insight")
                        st.write(answer)
                        
                        # Optional: Show previous questions
                        if st.session_state.previous_questions:
                            st.markdown("### Previous Questions")
                            for idx, prev_q in enumerate(st.session_state.previous_questions, 1):
                                st.write(f"{idx}. {prev_q}")
                    else:
                        st.error("Unable to generate a detailed answer.")

    def vital_sign_monitoring(self):
        """Advanced vital sign monitoring."""
        st.title("📊 Comprehensive Vital Sign Monitoring")
        
        heart_rate = st.number_input("Heart Rate (bpm):", min_value=0, max_value=300)
        oxygen_saturation = st.number_input("Oxygen Saturation (%):", min_value=0, max_value=100)
        systolic_bp = st.number_input("Systolic BP (mmHg):", min_value=0, max_value=300)
        diastolic_bp = st.number_input("Diastolic BP (mmHg):", min_value=0, max_value=200)
        
        if st.button("Analyze Vital Signs"):
            vitals_prompt = f"""Comprehensive Vital Signs Analysis:
            Heart Rate: {heart_rate} bpm
            Oxygen Saturation: {oxygen_saturation}%
            Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg

            Provide:
            1. Health status interpretation
            2. Comparison with healthy ranges
            3. Potential health risks
            4. Lifestyle recommendations
            5. When to seek medical attention"""
            
            result = self.safe_generate_content(vitals_prompt)
            if result:
                st.markdown("## 🩺 Vital Signs Health Assessment")
                st.write(result)
            else:
                st.error("Vital sign analysis failed.")

    def first_aid_guidance(self):
        """Comprehensive first aid guidance."""
        st.title("🚑 Emergency First Aid Guidance")
        emergency_type = st.text_input("Describe the emergency situation:")
        
        if st.button("Get Emergency Guidance"):
            first_aid_prompt = f"""Detailed First Aid Guidance for {emergency_type}:
            Provide:
            1. Immediate emergency response
            2. Action protocol
            3. Risk management
            4. Warning signs
            5. Professional help steps"""
            
            result = self.safe_generate_content(first_aid_prompt)
            if result:
                st.markdown("## 🆘 First Aid Protocol")
                st.write(result)
            else:
                st.error("Emergency guidance unavailable.")

def main():
    st.set_page_config(
        page_title="AI Medical Assistant", 
        page_icon="🩺", 
        layout="wide"
    )
    
    st.sidebar.title("Medical AI Assistant")
    medical_assistant = MedicalAssistant()
    
    app_mode = st.sidebar.selectbox(
        "Choose Service", 
        [
            "Symptom Diagnosis", 
            "Vital Sign Monitoring", 
            "First-Aid Guidance",
            "Medical Image Analysis"
        ]
    )

    if app_mode == "Symptom Diagnosis":
        medical_assistant.interactive_symptom_diagnosis()
    elif app_mode == "Vital Sign Monitoring":
        medical_assistant.vital_sign_monitoring()
    elif app_mode == "First-Aid Guidance":
        medical_assistant.first_aid_guidance()
    elif app_mode == "Medical Image Analysis":
        medical_assistant.medical_image_analysis()

if __name__ == "__main__":
    main()