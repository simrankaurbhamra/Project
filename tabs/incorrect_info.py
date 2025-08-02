"""
Incorrect Information Risk Prediction Tab

This module handles the prediction and visualization of incorrect information risk scores.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_profile import create_auto_filled_form, is_profile_complete, are_predictions_available, get_prediction_value, create_prediction_chart

def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects"""
    try:
        with open('models/incorrect_info_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        
        with open('data/clean/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('data/clean/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return model, label_encoders, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found. Please train the models first!")
        return None, None, None

def create_input_form():
    """Create user input form for incorrect information risk prediction"""
    st.header("📚 Incorrect Information Risk Assessment")
    st.markdown("### Evaluate your risk of receiving or spreading misinformation from AI")
    
    if is_profile_complete():
        return create_auto_filled_form("incorrect_info")
    
    st.warning("⚠️ Complete your profile in the sidebar to auto-fill this form!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="info_gender")
        age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-55", "56+"], key="info_age")
        country = st.selectbox("Country", ["USA", "Canada", "UK", "Germany", "India", "Other"], key="info_country")
        profession = st.selectbox("Profession", 
                                ["Software Developer", "Student", "Marketing Manager", "Doctor", 
                                 "Teacher", "Engineer", "Designer", "Other"], key="info_profession")
        education_level = st.selectbox("Education Level", 
                                     ["High School", "Some College", "Bachelor Degree", 
                                      "Master Degree", "PhD"], key="info_education")
    
    with col2:
        st.subheader("AI Usage Patterns")
        tech_experience = st.selectbox("Tech Experience Level", ["Beginner", "Intermediate", "Advanced"], key="info_tech")
        ai_tool = st.selectbox("Primary AI Tool", ["ChatGPT", "Claude", "Copilot", "Gemini", "Other"], key="info_tool")
        purpose = st.selectbox("Purpose of AI Use", 
                              ["Work Tasks", "Learning", "Content Creation", "Research", "Entertainment"], key="info_purpose")
        hours_per_day = st.slider("Hours spent per day on AI", 0.0, 12.0, 2.0, 0.1, key="info_hours")
        frequency = st.selectbox("Weekly Usage Frequency", 
                                ["Daily", "5-6 times/week", "3-4 times/week", "1-2 times/week", "Rarely"], key="info_freq")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Usage Details")
        months_using = st.slider("Months using AI", 1, 60, 12, key="info_months")
        task_complexity = st.selectbox("Task Complexity Level", ["Simple", "Moderate", "Complex"], key="info_complexity")
        verification = st.selectbox("How often do you verify AI outputs?", 
                                  ["Always", "Often", "Sometimes", "Rarely", "Never"], key="info_verify")
        confidence = st.slider("Confidence in AI outputs (1-10)", 1, 10, 7, key="info_confidence")
    
    with col4:
        st.subheader("Information Habits")
        critical_thinking_rating = st.slider("Critical thinking self-rating (1-10)", 1, 10, 7, key="info_critical")
        privacy_awareness = st.slider("Privacy awareness level (1-10)", 1, 10, 7, key="info_awareness")
        shares_info = st.selectbox("Do you share personal info with AI?", 
                                 ["Never", "Rarely", "Sometimes", "Often", "Always"], key="info_shares")
        reads_news = st.selectbox("Do you read AI-related news?", 
                                ["Never", "Rarely", "Sometimes", "Often", "Always"], key="info_news")
    
    return {
        'gender': gender,
        'age_group': age_group,
        'country': country,
        'profession': profession,
        'education_level': education_level,
        'tech_experience_level': tech_experience,
        'ai_tool_primary': ai_tool,
        'purpose_of_ai_use': purpose,
        'hours_spent_per_day_on_ai': hours_per_day,
        'frequency_of_use_weekly': frequency,
        'months_using_ai': months_using,
        'task_complexity_level': task_complexity,
        'verification_frequency': verification,
        'confidence_in_ai_outputs': confidence,
        'critical_thinking_self_rating': critical_thinking_rating,
        'privacy_awareness_level': privacy_awareness,
        'shares_personal_info_with_ai': shares_info,
        'reads_ai_related_news': reads_news
    }

def preprocess_input(user_input, label_encoders, scaler):
    """Preprocess user input for model prediction"""
  
    df = pd.DataFrame([user_input])
    
   
    categorical_cols = ['gender', 'age_group', 'country', 'profession', 'education_level', 
                       'tech_experience_level', 'ai_tool_primary', 'purpose_of_ai_use', 
                       'frequency_of_use_weekly', 'task_complexity_level', 'verification_frequency',
                       'shares_personal_info_with_ai', 'reads_ai_related_news']
    
 
    for col in categorical_cols:
        if col in label_encoders:
            try:
                df[f'{col}_encoded'] = label_encoders[col].transform(df[col])
            except ValueError:
                # Handle unseen categories by using the most frequent category
                df[f'{col}_encoded'] = 0  # Default encoding
    

    feature_cols = (['hours_spent_per_day_on_ai', 'months_using_ai', 'confidence_in_ai_outputs',
                    'critical_thinking_self_rating', 'privacy_awareness_level'] + 
                   [f'{col}_encoded' for col in categorical_cols])
    
    X = df[feature_cols]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled

def interpret_incorrect_info_score(score):
    """Interpret the incorrect information risk score"""
    if score <= 2.0:
        return "🟢 Very Low Risk", "Excellent fact-checking habits and information verification skills."
    elif score <= 4.0:
        return "🟡 Low Risk", "Good information literacy with minor areas for improvement."
    elif score <= 6.0:
        return "🟠 Moderate Risk", "Some vulnerability to misinformation. Enhance verification practices."
    elif score <= 8.0:
        return "🔴 High Risk", "Significant risk of accepting or spreading incorrect information."
    else:
        return "🚨 Very High Risk", "Critical vulnerability to misinformation. Immediate action needed."

def show_info_verification_recommendations(risk_level, score, user_input):
    """Show personalized recommendations for information verification"""
    st.subheader("🔍 Information Verification Action Plan")
    
    if score <= 4.0:
        recommendations = [
            "✅ **Excellent Information Habits - Maintain Current Practices:**",
            "• Continue your rigorous fact-checking approach",
            "• Share your verification methods with others",
            "• Stay updated on new verification tools and techniques",
            "• Help others identify misinformation",
            "• Consider becoming a digital literacy advocate",
        ]
    elif score <= 6.0:
        recommendations = [
            "⚖️ **Moderate Risk - Strengthen Verification Practices:**",
            "• Always cross-reference AI information with authoritative sources",
            "• Use multiple verification methods for important information",
            "• Learn to identify common AI hallucination patterns",
            "• Develop a systematic fact-checking routine",
            "• Question AI responses that seem too convenient or surprising",
            "• Use fact-checking websites and tools regularly",
        ]
    else:
        recommendations = [
            "🚨 **High Risk - Immediate Verification Protocol Needed:**",
            "• NEVER trust AI information without independent verification",
            "• Always check at least 2-3 authoritative sources",
            "• Learn to identify reliable vs unreliable information sources",
            "• Take a digital literacy course or workshop",
            "• Use AI primarily for creative tasks, not factual information",
            "• Ask experts or colleagues to verify important AI-generated content",
            "• Be extremely cautious about sharing AI information publicly",
            "• Consider reducing AI usage for information-critical tasks",
        ]
    
    for rec in recommendations:
        if rec.startswith("•"):
            st.write(rec)
        else:
            st.markdown(f"**{rec}**")
    
    # Specific recommendations based on user profile
    st.subheader("🎯 Targeted Recommendations for Your Profile")
    
    specific_tips = []
    
    if user_input['verification_frequency'] in ['Rarely', 'Never']:
        specific_tips.append("⚠️ **Critical Gap**: You rarely verify AI outputs. This is your highest risk factor.")
    
    if user_input['confidence_in_ai_outputs'] >= 8:
        specific_tips.append("🚨 **Overconfidence Alert**: Very high confidence in AI can lead to accepting incorrect information.")
    
    if user_input['purpose_of_ai_use'] in ['Research', 'Work Tasks']:
        specific_tips.append("💼 **Professional Risk**: Using AI for research/work requires extra verification diligence.")
    
    if user_input['reads_ai_related_news'] in ['Rarely', 'Never']:
        specific_tips.append("📰 **Knowledge Gap**: Stay informed about AI limitations and misinformation risks.")
    
    if user_input['tech_experience_level'] == 'Beginner':
        specific_tips.append("🎓 **Learning Opportunity**: Invest in digital literacy training for safer AI usage.")
    
    for tip in specific_tips:
        st.warning(tip)

def show_verification_toolkit():
    """Show fact-checking and verification tools"""
    st.subheader("🛠️ Information Verification Toolkit")
    
    verification_methods = {
        "Cross-Reference Multiple Sources": "Always check 2-3 independent, authoritative sources",
        "Check Publication Date": "Ensure information is current and hasn't been superseded",
        "Verify Author Credentials": "Check if the source has expertise in the subject area",
        "Look for Citations": "Reliable information should reference credible sources",
        "Use Fact-Checking Sites": "Snopes, FactCheck.org, PolitiFact for controversial claims",
        "Check Official Sources": "Government, academic institutions, established organizations",
        "Reverse Image Search": "Verify images haven't been manipulated or taken out of context",
        "Check URL Legitimacy": "Be wary of suspicious domains or URL structures"
    }
    
    for method, description in verification_methods.items():
        with st.expander(f"✅ {method}"):
            st.write(description)
    
    st.subheader("🚩 Red Flags for AI-Generated Information")
    red_flags = [
        "Information that seems too perfect or convenient",
        "Claims without any sources or references",
        "Statistics that can't be verified elsewhere",
        "Information that contradicts established facts",
        "Overly confident statements about uncertain topics",
        "Historical 'facts' that seem questionable",
        "Scientific claims without peer review",
        "Current events that can't be found in news sources"
    ]
    
    for flag in red_flags:
        st.error(f"🚩 {flag}")

def run_incorrect_info_tab():
    """Main function to run the incorrect information risk prediction tab"""
    
    # Check if predictions are already available
    if are_predictions_available():
        prediction_value = get_prediction_value('incorrect_info')
        if prediction_value is not None:
            # Display automatic prediction results
            st.header("📚 Incorrect Information Risk Assessment")
            st.success("✅ Prediction automatically generated from your saved profile!")
            
            # Display results
            st.markdown("---")
            st.header("📊 Information Risk Assessment Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Risk gauge
                fig = create_prediction_chart(prediction_value, "Incorrect Information Risk Score", "incorrect_info")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk interpretation
                risk_level, interpretation = interpret_incorrect_info_score(prediction_value)
                st.metric("Information Risk Level", risk_level)
                st.write(f"**Score:** {prediction_value:.2f}/10")
                st.write(f"**Interpretation:** {interpretation}")
            
            # Get profile for additional insights
            from shared_profile import get_profile_data
            user_input = get_profile_data()
            
            # Information behavior analysis
            st.subheader("📈 Your Information Verification Profile")
            
            verification_profile = {
                "Verification Frequency": user_input['verification_frequency'],
                "AI Confidence Level": f"{user_input['confidence_in_ai_outputs']}/10",
                "Primary Use Case": user_input['purpose_of_ai_use'],
                "Tech Experience": user_input['tech_experience_level'],
                "Information Awareness": user_input['reads_ai_related_news']
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Verification:** {verification_profile['Verification Frequency']}")
                st.info(f"**AI Confidence:** {verification_profile['AI Confidence Level']}")
            with col2:
                st.info(f"**Primary Use:** {verification_profile['Primary Use Case']}")
                st.info(f"**Experience:** {verification_profile['Tech Experience']}")
            with col3:
                st.info(f"**Info Awareness:** {verification_profile['Information Awareness']}")
            
            # Show recommendations
            show_info_verification_recommendations(risk_level, prediction_value, user_input)
            
            # Show verification toolkit
            show_verification_toolkit()
            return
    
    model, label_encoders, scaler = load_model_and_preprocessors()
    
    if model is None:
        st.warning("⚠️ Please train the models first by running the training script!")
        return
    
    # Create input form
    user_input = create_input_form()
    
    
    if user_input is None:
        return
    
    # Prediction button
    if st.button("🔮 Assess Incorrect Information Risk", type="primary"):
        try:
            # Preprocess input
            X_processed = preprocess_input(user_input, label_encoders, scaler)
            
            # Make prediction
            prediction = model.predict(X_processed)[0]
            
            # Display results
            st.markdown("---")
            st.header("📊 Information Risk Assessment Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Risk gauge
                fig = create_prediction_chart(prediction, "Incorrect Information Risk Score", "incorrect_info")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk interpretation
                risk_level, interpretation = interpret_incorrect_info_score(prediction)
                st.metric("Information Risk Level", risk_level)
                st.write(f"**Score:** {prediction:.2f}/10")
                st.write(f"**Interpretation:** {interpretation}")
            
           
            st.subheader("📈 Your Information Verification Profile")
            
            verification_profile = {
                "Verification Frequency": user_input['verification_frequency'],
                "AI Confidence Level": f"{user_input['confidence_in_ai_outputs']}/10",
                "Primary Use Case": user_input['purpose_of_ai_use'],
                "Tech Experience": user_input['tech_experience_level'],
                "Information Awareness": user_input['reads_ai_related_news']
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Verification:** {verification_profile['Verification Frequency']}")
                st.info(f"**AI Confidence:** {verification_profile['AI Confidence Level']}")
            with col2:
                st.info(f"**Primary Use:** {verification_profile['Primary Use Case']}")
                st.info(f"**Experience:** {verification_profile['Tech Experience']}")
            with col3:
                st.info(f"**Info Awareness:** {verification_profile['Information Awareness']}")
            
            # Show recommendations
            show_info_verification_recommendations(risk_level, prediction, user_input)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that all required model files are present and properly trained.")
    
    # Show verification toolkit
    show_verification_toolkit()

if __name__ == "__main__":
    run_incorrect_info_tab()