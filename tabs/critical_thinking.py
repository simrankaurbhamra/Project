"""
Critical Thinking Risk Prediction Tab

This module handles the prediction and visualization of critical thinking risk scores.
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
       
        with open('models/critical_thinking_model.pkl', 'rb') as f:
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
    """Create user input form for critical thinking risk prediction"""
    st.header("🧠 Critical Thinking Risk Assessment")
    st.markdown("### Enter your AI usage information to assess critical thinking risk")
    
    
    if is_profile_complete():
        return create_auto_filled_form("critical_thinking")
    
    
    st.warning("⚠️ Complete your profile in the sidebar to auto-fill this form!")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-55", "56+"])
        country = st.selectbox("Country", ["USA", "Canada", "UK", "Germany", "India", "Other"])
        profession = st.selectbox("Profession", 
                                ["Software Developer", "Student", "Marketing Manager", "Doctor", 
                                 "Teacher", "Engineer", "Designer", "Other"])
        education_level = st.selectbox("Education Level", 
                                     ["High School", "Some College", "Bachelor Degree", 
                                      "Master Degree", "PhD"])
    
    with col2:
        st.subheader("AI Usage Patterns")
        tech_experience = st.selectbox("Tech Experience Level", ["Beginner", "Intermediate", "Advanced"])
        ai_tool = st.selectbox("Primary AI Tool", ["ChatGPT", "Claude", "Copilot", "Gemini", "Other"])
        purpose = st.selectbox("Purpose of AI Use", 
                              ["Work Tasks", "Learning", "Content Creation", "Research", "Entertainment"])
        hours_per_day = st.slider("Hours spent per day on AI", 0.0, 12.0, 2.0, 0.1)
        frequency = st.selectbox("Weekly Usage Frequency", 
                                ["Daily", "5-6 times/week", "3-4 times/week", "1-2 times/week", "Rarely"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Usage Details")
        months_using = st.slider("Months using AI", 1, 60, 12)
        task_complexity = st.selectbox("Task Complexity Level", ["Simple", "Moderate", "Complex"])
        verification = st.selectbox("How often do you verify AI outputs?", 
                                  ["Always", "Often", "Sometimes", "Rarely", "Never"])
        confidence = st.slider("Confidence in AI outputs (1-10)", 1, 10, 7)
    
    with col4:
        st.subheader("Self Assessment")
        critical_thinking_rating = st.slider("Critical thinking self-rating (1-10)", 1, 10, 7)
        privacy_awareness = st.slider("Privacy awareness level (1-10)", 1, 10, 7)
        shares_info = st.selectbox("Do you share personal info with AI?", 
                                 ["Never", "Rarely", "Sometimes", "Often", "Always"])
        reads_news = st.selectbox("Do you read AI-related news?", 
                                ["Never", "Rarely", "Sometimes", "Often", "Always"])
    
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
    
    # Select features in the same order as training
    feature_cols = (['hours_spent_per_day_on_ai', 'months_using_ai', 'confidence_in_ai_outputs',
                    'critical_thinking_self_rating', 'privacy_awareness_level'] + 
                   [f'{col}_encoded' for col in categorical_cols])
    
    X = df[feature_cols]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled

def interpret_risk_score(score):
    """Interpret the critical thinking risk score"""
    if score <= 2.0:
        return "🟢 Very Low Risk", "Your critical thinking skills appear to be well-maintained despite AI usage."
    elif score <= 4.0:
        return "🟡 Low Risk", "Minimal impact on critical thinking. Continue being mindful of AI dependency."
    elif score <= 6.0:
        return "🟠 Moderate Risk", "Some potential impact on critical thinking. Consider more verification practices."
    elif score <= 8.0:
        return "🔴 High Risk", "Significant risk to critical thinking skills. Recommended to reduce AI dependency."
    else:
        return "🚨 Very High Risk", "Critical thinking skills may be severely impacted. Immediate action recommended."

def show_recommendations(risk_level, score):
    """Show personalized recommendations based on risk level"""
    st.subheader("💡 Personalized Recommendations")
    
    if score <= 4.0:
        recommendations = [
            "✅ Continue your current balanced approach to AI usage",
            "📚 Keep reading diverse sources beyond AI-generated content",
            "🤔 Maintain your habit of questioning and verifying information",
            "🔄 Regularly challenge yourself with complex problems without AI assistance"
        ]
    elif score <= 6.0:
        recommendations = [
            "⚖️ Try to balance AI assistance with independent thinking",
            "✓ Always fact-check important AI-generated information",
            "🧪 Practice solving problems step-by-step without AI first",
            "📖 Engage with challenging content that requires deep analysis",
            "👥 Discuss AI outputs with colleagues or peers for validation"
        ]
    else:
        recommendations = [
            "🚨 Implement 'AI-free' periods in your daily routine",
            "📝 Practice writing and problem-solving without AI assistance",
            "🔍 Develop a systematic approach to verify all AI outputs",
            "🧠 Engage in activities that promote critical thinking (debates, puzzles, analysis)",
            "📚 Return to traditional research methods for important tasks",
            "⏰ Set strict time limits for AI tool usage",
            "🎯 Focus on understanding 'why' rather than just accepting AI answers"
        ]
    
    for rec in recommendations:
        st.write(rec)

def create_risk_visualization(score):
    """Create a risk visualization gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Critical Thinking Risk Score"},
        delta = {'reference': 5.0},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 2], 'color': "lightgreen"},
                {'range': [2, 4], 'color': "yellow"},
                {'range': [4, 6], 'color': "orange"},
                {'range': [6, 8], 'color': "red"},
                {'range': [8, 10], 'color': "darkred"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 7.0
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def run_critical_thinking_tab():
    """Main function to run the critical thinking risk prediction tab"""
    
  
    if are_predictions_available():
        prediction_value = get_prediction_value('critical_thinking')
        if prediction_value is not None:
            st.header("🧠 Critical Thinking Risk Assessment")
            st.success("✅ Prediction automatically generated from your saved profile!")
            
            
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                
                fig = create_prediction_chart(prediction_value, "Critical Thinking Risk Score", "critical_thinking")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
              
                risk_level, interpretation = interpret_risk_score(prediction_value)
                st.metric("Risk Level", risk_level)
                st.write(f"**Score:** {prediction_value:.2f}/10")
                st.write(f"**Interpretation:** {interpretation}")
            
            from shared_profile import get_profile_data
            user_input = get_profile_data()
            
           
            show_recommendations(risk_level, prediction_value)
            
            st.subheader("📈 Key Factors Analysis")
            st.info(f"""
            **Your Profile Summary:**
            - AI Usage: {user_input['hours_spent_per_day_on_ai']:.1f} hours/day for {user_input['months_using_ai']} months
            - Experience Level: {user_input['tech_experience_level']}
            - Verification Habit: {user_input['verification_frequency']}
            - Self-rated Critical Thinking: {user_input['critical_thinking_self_rating']}/10
            """)
            return

    model, label_encoders, scaler = load_model_and_preprocessors()
    
    if model is None:
        st.warning("⚠️ Please train the models first by running the training script!")
        return
    
    user_input = create_input_form()
    
    
    if st.button("🔮 Predict Critical Thinking Risk", type="primary"):
        try:
            
            X_processed = preprocess_input(user_input, label_encoders, scaler)
            
        
            prediction = model.predict(X_processed)[0]
            
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                
                fig = create_risk_visualization(prediction)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
              
                risk_level, interpretation = interpret_risk_score(prediction)
                st.metric("Risk Level", risk_level)
                st.write(f"**Score:** {prediction:.2f}/10")
                st.write(f"**Interpretation:** {interpretation}")
            
           
            show_recommendations(risk_level, prediction)
            
          
            st.subheader("📈 Key Factors Analysis")
            st.info(f"""
            **Your Profile Summary:**
            - AI Usage: {user_input['hours_spent_per_day_on_ai']:.1f} hours/day for {user_input['months_using_ai']} months
            - Experience Level: {user_input['tech_experience_level']}
            - Verification Habit: {user_input['verification_frequency']}
            - Self-rated Critical Thinking: {user_input['critical_thinking_self_rating']}/10
            """)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that all required model files are present and properly trained.")

if __name__ == "__main__":
    run_critical_thinking_tab()