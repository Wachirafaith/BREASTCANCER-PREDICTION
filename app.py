import streamlit as st
import numpy as np
import pickle
from datetime import datetime
import google.generativeai as genai

SYSTEM_PROMPT = """
You are a specialized AI Health Assistant focused on breast cancer education and emotional support.

ROLE:
- You help patients and healthcare users understand breast cancer-related information.
- You explain concepts clearly, calmly, and compassionately.
- You adapt your tone based on whether the user seems anxious, confused, or technical.

KNOWLEDGE:
- Breast cancer basics (benign vs malignant)
- Diagnostic terms (biopsy, cytology, pathology reports)
- Test result explanations (non-diagnostic)
- Treatment options (surgery, chemotherapy, radiotherapy, hormone therapy)
- Lifestyle guidance, prevention, and follow-up care
- Emotional reassurance and coping support

MEDICAL SAFETY RULES:
- You do NOT provide medical diagnoses.
- You do NOT predict outcomes.
- You ALWAYS encourage consulting qualified healthcare professionals.
- You never alarm the user.

STYLE:
- Warm, empathetic, human-like
- Simple language by default
- More technical only if the user asks
- Calm, supportive, and respectful

RESPONSE STRUCTURE:
1. Acknowledge the user’s concern or question
2. Explain clearly and simply
3. Offer helpful context or next steps
4. End with reassurance and professional guidance
"""

# Configure Gemini API key from Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except:
    GEMINI_API_KEY = None
    st.error("API key not found. Please configure secrets.toml")

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Diagnostic Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark green and black theme
st.markdown("""
<style>
    .main {background-color: #0E1117;}
    .stButton>button {background-color: #161B22; color: white; height: 45px; border: none; border-radius: 6px;}
    .stButton>button:hover {background-color: #343632;}
    h1, h2, h3, h4, h5, h6 {color: #ffffff;}
    p, li, label {color: #d0d0d0;}
    .stTextInput input, .stNumberInput input, .stDateInput input {background-color: black; color: #ffffff;}
    .stExpander {background-color: black;}
    div[data-baseweb="select"] > div {background-color: #102613; color: #ffffff;}
</style>
""", unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load the pre-trained SVM model and StandardScaler"""
    try:
        with open('svm.pkl', 'rb') as file:
            svm_model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            standard_scaler = pickle.load(file)
        return svm_model, standard_scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'svm.pkl' and 'scaler.pkl' are in the application directory.")
        st.stop()


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}
if 'measurements' not in st.session_state:
    st.session_state.measurements = {}
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Navigation function
def go_to_page(page_name):
    st.session_state.page = page_name
    st.rerun()


# Feature definitions
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

feature_labels = {
    'radius_mean': 'Radius', 'texture_mean': 'Texture', 'perimeter_mean': 'Perimeter', 'area_mean': 'Area',
    'smoothness_mean': 'Smoothness', 'compactness_mean': 'Compactness', 'concavity_mean': 'Concavity',
    'concave_points_mean': 'Concave Points', 'symmetry_mean': 'Symmetry', 'fractal_dimension_mean': 'Fractal Dimension',
    'radius_se': 'Radius SE', 'texture_se': 'Texture SE', 'perimeter_se': 'Perimeter SE', 'area_se': 'Area SE',
    'smoothness_se': 'Smoothness SE', 'compactness_se': 'Compactness SE', 'concavity_se': 'Concavity SE',
    'concave_points_se': 'Concave Points SE', 'symmetry_se': 'Symmetry SE',
    'fractal_dimension_se': 'Fractal Dimension SE',
    'radius_worst': 'Radius (Worst)', 'texture_worst': 'Texture (Worst)', 'perimeter_worst': 'Perimeter (Worst)',
    'area_worst': 'Area (Worst)',
    'smoothness_worst': 'Smoothness (Worst)', 'compactness_worst': 'Compactness (Worst)',
    'concavity_worst': 'Concavity (Worst)',
    'concave_points_worst': 'Concave Points (Worst)', 'symmetry_worst': 'Symmetry (Worst)',
    'fractal_dimension_worst': 'Fractal Dimension (Worst)'
}

feature_ranges = {
    'radius_mean': (6.0, 28.0, 14.0), 'texture_mean': (9.0, 40.0, 19.0), 'perimeter_mean': (43.0, 189.0, 92.0),
    'area_mean': (143.0, 2501.0, 655.0), 'smoothness_mean': (0.05, 0.16, 0.10), 'compactness_mean': (0.02, 0.35, 0.10),
    'concavity_mean': (0.0, 0.43, 0.09), 'concave_points_mean': (0.0, 0.20, 0.05), 'symmetry_mean': (0.11, 0.30, 0.18),
    'fractal_dimension_mean': (0.05, 0.10, 0.06), 'radius_se': (0.1, 2.9, 0.4), 'texture_se': (0.4, 4.9, 1.2),
    'perimeter_se': (0.8, 22.0, 2.9), 'area_se': (6.0, 542.0, 40.0), 'smoothness_se': (0.002, 0.031, 0.007),
    'compactness_se': (0.002, 0.135, 0.025), 'concavity_se': (0.0, 0.40, 0.03), 'concave_points_se': (0.0, 0.05, 0.01),
    'symmetry_se': (0.008, 0.079, 0.020), 'fractal_dimension_se': (0.001, 0.030, 0.004),
    'radius_worst': (7.0, 36.0, 16.0),
    'texture_worst': (12.0, 50.0, 25.0), 'perimeter_worst': (50.0, 251.0, 107.0), 'area_worst': (185.0, 4254.0, 880.0),
    'smoothness_worst': (0.07, 0.22, 0.13), 'compactness_worst': (0.03, 1.06, 0.21),
    'concavity_worst': (0.0, 1.25, 0.27),
    'concave_points_worst': (0.0, 0.29, 0.11), 'symmetry_worst': (0.16, 0.66, 0.29),
    'fractal_dimension_worst': (0.055, 0.208, 0.084)
}

# Sidebar navigation
with st.sidebar:
    st.markdown("### Navigation")

    if st.button("Home", use_container_width=True):
        go_to_page('home')

    if st.session_state.prediction is not None:
        if st.button("Results", use_container_width=True):
            go_to_page('results')
        if st.button("Recommendations", use_container_width=True):
            go_to_page('recommendations')
        if st.button("AI Assistant", use_container_width=True):
            go_to_page('chatbot')

    st.markdown("---")

    if st.button("About", use_container_width=True):
        go_to_page('about')

    st.markdown("---")

    st.caption("**Current User:**")
    if st.session_state.user_type:
        st.info(f"{st.session_state.user_type}")
    else:
        st.info("Not selected")

# ===========================
# HOME PAGE
# ===========================
if st.session_state.page == 'home':
    st.title("🏥 Breast Cancer Diagnostic Support System")

    st.info("Welcome to the Diagnostic Support System")

    st.markdown("""
    This professional system assists in analyzing breast cell measurements from laboratory reports 
    to predict whether tissue characteristics are benign or malignant using advanced machine learning.

    **For:** Laboratory assistants, healthcare professionals, and patients with lab reports
    """)

    st.markdown("### Please select your user type:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Lab Assistant / Healthcare Professional")
        st.caption("Enter measurements from laboratory analysis and diagnostic reports")

        if st.button("Continue as Professional", use_container_width=True, type="primary"):
            st.session_state.user_type = "Professional"
            go_to_page('patient_info')

    with col2:
        st.markdown("#### Patient")
        st.caption("Enter measurements from your lab report to understand your results")

        if st.button("Continue as Patient", use_container_width=True):
            st.session_state.user_type = "Patient"
            go_to_page('patient_info')

    st.markdown("---")

    st.warning("""
    **Important Notice:** This system provides analytical insights based on cell measurements. 
    Results should be reviewed by qualified healthcare professionals. This is not a substitute 
    for professional medical diagnosis or treatment.
    """)

# ===========================
# PATIENT INFORMATION PAGE
# ===========================
elif st.session_state.page == 'patient_info':
    st.title("Patient Information")

    if st.session_state.user_type == "Professional":
        st.info("**Professional Mode** - Enter patient information from lab records")
    else:
        st.info("**Patient Mode** - Enter your information (optional for privacy)")

    col1, col2 = st.columns(2)

    with col1:
        patient_name = st.text_input("Patient Name / ID",
                                     value=st.session_state.patient_info.get('name', ''),
                                     placeholder="Optional - can be left blank")
        patient_age = st.number_input("Age",
                                      min_value=18, max_value=100,
                                      value=st.session_state.patient_info.get('age', 40))

    with col2:
        sample_id = st.text_input("Lab Report / Sample ID",
                                  value=st.session_state.patient_info.get('sample_id', ''),
                                  placeholder="e.g., LAB-2024-001")
        test_date = st.date_input("Test Date",
                                  value=st.session_state.patient_info.get('date', datetime.now()))

    lab_source = st.text_input("Laboratory / Medical Facility",
                               value=st.session_state.patient_info.get('lab', ''),
                               placeholder="Optional")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Home"):
            go_to_page('home')
    with col2:
        if st.button("Continue to Measurements →", type="primary"):
            st.session_state.patient_info = {
                'name': patient_name,
                'age': patient_age,
                'sample_id': sample_id,
                'date': test_date,
                'lab': lab_source
            }
            go_to_page('measurements')

# ===========================
# MEASUREMENTS INPUT PAGE
# ===========================
elif st.session_state.page == 'measurements':
    st.title("Cell Measurements Entry")

    if st.session_state.user_type == "Professional":
        st.info("Enter the 30 diagnostic measurements from the cytology report")
    else:
        st.info("Enter the measurements from your lab report. All 30 values are required.")

    st.info("""
    💡 **Tip:** These measurements come from digitized images of fine needle aspirate (FNA) of breast mass. 
    You'll find them in your cytology/pathology report under cell nucleus characteristics.
    """)

    measurements = {}

    # Mean Measurements
    with st.expander("**A. Mean Measurements** (Average values)", expanded=True):
        cols = st.columns(2)
        for i, feature in enumerate(feature_names[:10]):
            with cols[i % 2]:
                min_val, max_val, default_val = feature_ranges[feature]
                measurements[feature] = st.number_input(
                    feature_labels[feature],
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=st.session_state.measurements.get(feature, float(default_val)),
                    step=None,
                    format="%.6f",
                    key=f"input_{feature}",
                    help=f"Expected range: {min_val} - {max_val}"
                )

    # Standard Error Measurements
    with st.expander("**B. Standard Error Measurements** (Variability)", expanded=False):
        cols = st.columns(2)
        for i, feature in enumerate(feature_names[10:20]):
            with cols[i % 2]:
                min_val, max_val, default_val = feature_ranges[feature]
                measurements[feature] = st.number_input(
                    feature_labels[feature],
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=st.session_state.measurements.get(feature, float(default_val)),
                    step=None,
                    format="%.6f",
                    key=f"input_{feature}",
                    help=f"Expected range: {min_val} - {max_val}"
                )

    # Worst Measurements
    with st.expander("**C. Worst Measurements** (Largest values)", expanded=False):
        cols = st.columns(2)
        for i, feature in enumerate(feature_names[20:30]):
            with cols[i % 2]:
                min_val, max_val, default_val = feature_ranges[feature]
                measurements[feature] = st.number_input(
                    feature_labels[feature],
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=st.session_state.measurements.get(feature, float(default_val)),
                    step=None,
                    format="%.6f",
                    key=f"input_{feature}",
                    help=f"Expected range: {min_val} - {max_val}"
                )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.measurements = measurements
            go_to_page('patient_info')
    with col2:
        if st.button("Analyze Measurements", type="primary"):
            st.session_state.measurements = measurements

            with st.spinner("Analyzing cell characteristics..."):
                model, scaler = load_model_and_scaler()

                # Prepare and predict
                input_array = np.array([[measurements[f] for f in feature_names]])
                scaled_input = scaler.transform(input_array)
                prediction = model.predict(scaled_input)[0]

                # Get confidence
                if hasattr(model, 'decision_function'):
                    decision_score = model.decision_function(scaled_input)[0]
                    confidence = min(100, (abs(decision_score) / 3.0) * 100)
                else:
                    decision_score = 0.0
                    confidence = 85.0

                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.decision_score = decision_score

                go_to_page('results')

# ===========================
# RESULTS PAGE
# ===========================
elif st.session_state.page == 'results':
    st.title("Analysis Results")

    # Patient info header
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Patient/Sample", st.session_state.patient_info.get('name') or st.session_state.patient_info.get(
            'sample_id') or "Not specified")
    with col2:
        st.metric("Age", st.session_state.patient_info.get('age', 'N/A'))
    with col3:
        st.metric("Test Date", st.session_state.patient_info.get('date', datetime.now()).strftime('%Y-%m-%d'))

    st.markdown("---")

    # Main result
    if st.session_state.prediction == 0:
        result_text = "BENIGN"
        result_color = "white"
        bg_color = "#264F2F"
        icon = "✅"
        meaning = "Non-Cancerous"
    else:
        result_text = "MALIGNANT"
        result_color = "white"
        bg_color = "#971818"
        icon = "⚠️"
        meaning = "Cancerous"

    st.markdown(f"""
    <div style='background-color: {bg_color}; padding: 40px; border-radius: 12px; 
                border-left: 8px solid {result_color}; text-align: center; margin: 30px 0;'>
        <h1 style='color: {result_color}; margin: 0; font-size: 52px;'>{icon} {result_text}</h1>
        <p style='color: white; font-size: 22px; margin-top: 15px;'>{meaning}</p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence and metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Confidence Level", f"{st.session_state.confidence:.1f}%")
    with col2:
        st.metric("Decision Score", f"{st.session_state.decision_score:.4f}")
    with col3:
        confidence_label = "High" if st.session_state.confidence >= 70 else (
            "Moderate" if st.session_state.confidence >= 40 else "Lower")
        st.metric("Strength", confidence_label)

    st.markdown("---")

    # Interpretation
    st.subheader("What This Means")

    if st.session_state.prediction == 0:
        st.success(f"## {icon} {result_text}")
        st.markdown(f"**{meaning}**")

        st.markdown("""
        **Benign Classification**

        The analysis indicates that the cell measurements show characteristics consistent with 
        benign (non-cancerous) tissue. The cell nucleus features fall within patterns typically 
        associated with normal or non-malignant cells.

        **Next Steps:** While this is encouraging news, it's important to follow up with your 
        healthcare provider to discuss the complete results and any recommended monitoring or additional tests.
        """)
    else:
        st.error(f"## {icon} {result_text}")
        st.markdown(f"**{meaning}**")

        st.markdown("""
        **Malignant Classification**

        The analysis indicates that the cell measurements show characteristics consistent with 
        malignant (cancerous) tissue. The cell nucleus features display patterns that suggest 
        concerning abnormalities.

        **Important:** This result suggests the need for further medical evaluation. Please consult 
        with your healthcare provider promptly to discuss these findings and determine appropriate 
        next steps, which may include additional testing or specialist consultation.
        """)

    st.markdown("---")

    # Quick actions
    st.subheader("What would you like to do next?")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("View Recommendations", use_container_width=True, type="primary"):
            go_to_page('recommendations')

    with col2:
        if st.button("Talk to AI Assistant", use_container_width=True):
            go_to_page('chatbot')

    with col3:
        if st.button("New Analysis", use_container_width=True):
            st.session_state.prediction = None
            st.session_state.confidence = None
            st.session_state.measurements = {}
            go_to_page('patient_info')

# ===========================
# RECOMMENDATIONS PAGE
# ===========================
elif st.session_state.page == 'recommendations':
    st.title("Personalized Recommendations")

    if st.session_state.prediction is None:
        st.warning("Please complete an analysis first.")
        if st.button("Go to Measurements"):
            go_to_page('measurements')
        st.stop()

    st.info(f"Based on your **{'BENIGN' if st.session_state.prediction == 0 else 'MALIGNANT'}** classification result")

    if st.session_state.prediction == 0:  # Benign
        st.success("### Recommendations for Benign Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Medical Follow-Up

            - Schedule a follow-up appointment with your healthcare provider
            - Discuss the complete pathology report
            - Ask about recommended monitoring schedule
            - Keep records of all test results
            - Continue regular breast health screenings

            #### Monitoring & Prevention

            - Perform monthly breast self-examinations
            - Follow recommended mammography schedule
            - Report any changes in breast tissue promptly
            - Maintain awareness of family medical history
            - Stay informed about breast health
            """)

        with col2:
            st.markdown("""
            #### Lifestyle Recommendations

            - Maintain a healthy weight through balanced diet
            - Exercise regularly (at least 150 minutes/week)
            - Limit alcohol consumption
            - Avoid smoking
            - Eat plenty of fruits and vegetables
            - Manage stress through relaxation techniques

            #### Educational Resources

            - Learn proper self-examination techniques
            - Understand risk factors and prevention
            - Stay updated on screening guidelines
            - Join support groups if desired
            - Share information with family members
            """)

    else:  # Malignant
        st.error("###  Recommendations for Malignant Results")

        st.warning("""
        **Important:** These recommendations are general guidance. Your actual care plan should be 
        determined by your oncology team based on your specific situation.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Immediate Steps

            - Contact your healthcare provider promptly
            - Request referral to an oncologist/specialist
            - Gather all medical records and test results
            - Prepare questions for your medical team
            - Consider seeking a second opinion
            - Bring a family member/friend to appointments

            #### Further Testing

            - Additional biopsies may be recommended
            - Imaging tests (MRI, CT, PET scans)
            - Blood tests and tumor markers
            - Genetic testing if appropriate
            - Staging to determine extent
            """)

        with col2:
            st.markdown("""
            #### Support & Resources

            - Connect with cancer support groups
            - Consider counseling or therapy
            - Reach out to patient navigators
            - Explore financial assistance programs
            - Involve family in care planning
            - Look into clinical trials

            #### Questions to Ask Your Doctor

            - What specific type and stage is it?
            - What are my treatment options?
            - What are the expected outcomes?
            - What are potential side effects?
            - How will this affect my daily life?
            - What support services are available?
            """)

    st.markdown("---")

    st.info("**Need More Guidance?** Talk to our AI Assistant for personalized answers about these recommendations.")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Ask AI Assistant About Recommendations", type="primary", use_container_width=True):
        go_to_page('chatbot')

# ===========================
# AI CHATBOT PAGE WITH GEMINI
# ===========================
elif st.session_state.page == 'chatbot':
    st.title("AI Health Assistant")

    if not GEMINI_API_KEY:
        st.error("Gemini API key not configured.")
        st.stop()

    st.info("""
    I can help you understand:
    • Your breast cancer test results  
    • Medical terms in your report  
    • Possible next steps  
    • Emotional and lifestyle guidance  

    ⚠️ I do not replace a doctor.
    """)

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["message"])

    # Chat input (correct Streamlit way)
    user_input = st.chat_input("Ask me a question about your results or breast cancer...")

    if user_input:
        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "message": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

                    # Build context
                    context = SYSTEM_PROMPT

                    if st.session_state.prediction is not None:
                        result = (
                            "benign (non-cancerous)"
                            if st.session_state.prediction == 0
                            else "malignant (cancerous)"
                        )
                        context += f"\n\nUSER CONTEXT: The user's analysis result shows {result} characteristics."

                    full_prompt = f"{context}\n\nUSER QUESTION:\n{user_input}"

                    response = gemini_model.generate_content(full_prompt)

                    if response and response.text:
                        ai_response = response.text
                    else:
                        ai_response = (
                            "I’m sorry, I couldn’t generate a response right now. "
                            "Please try rephrasing your question."
                        )

                except Exception as e:
                    ai_response = f"⚠️ I encountered an error: {e}"

            st.markdown(ai_response)

        # Save assistant response
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": ai_response
        })

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ===========================
# ABOUT PAGE
# ===========================
elif st.session_state.page == 'about':
    st.title("About This System")

    st.markdown("### 🏥 Breast Cancer Diagnostic Assistant")
    st.markdown("""
    A professional system designed to assist laboratory assistants and patients in understanding 
    breast cancer diagnostic results through machine learning analysis.
    """)

    st.subheader("Machine Learning Model")
    st.markdown("""
    - **Algorithm:** Support Vector Machine (SVM)
    - **Preprocessing:** StandardScaler for feature normalization
    - **Training/Test Split:** 80% training, 20% testing
    - **Performance:** Approximately 96% accuracy on test data
    - **Validation:** Cross-validated for reliable performance
    """)

    st.subheader("How It Works")
    st.markdown("""
    1. **Input:** User enters 30 cell nucleus measurements from lab reports
    2. **Preprocessing:** Features are normalized using StandardScaler
    3. **Prediction:** SVM model classifies as benign or malignant
    4. **Confidence:** Distance from decision boundary indicates confidence
    5. **Output:** Result with confidence score and personalized recommendations
    """)

    st.subheader("Who Can Use This")
    st.markdown("""
    - **Lab Assistants:** Enter measurements from laboratory analysis
    - **Healthcare Professionals:** Review and interpret diagnostic results
    - **Patients:** Understand lab report findings with AI assistance
    - **Researchers:** Study machine learning applications in healthcare
    """)

    st.subheader("⚠️ Important Disclaimer")
    st.error("""
    **⚠️ Medical Disclaimer**

    This system provides analytical insights for educational and informational purposes only. 
    It is NOT a medical diagnostic tool and should NOT be used as the sole basis for medical decisions.

    **Always consult qualified healthcare professionals for:**
    - Proper diagnosis and medical evaluation
    - Treatment planning and decisions
    - Interpretation of lab results
    - Medical advice and care

    Results from this system should be reviewed by medical professionals and used only as a 
    supplementary decision support tool.
    """)

    st.subheader("Support & Resources")
    st.markdown("""
    - **National Cancer Institute:** cancer.gov
    - **Breast Cancer Support Groups:** breastcancer.org
    - **Emergency Medical Help:** local emergency number
    """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p><strong>Version 1.0</strong> | Developed for educational and research purposes</p>
        <p>Powered by Machine Learning and Google Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)