"""
Privacy Concern Level Prediction Tab

This module handles the prediction and visualization of privacy concern levels.
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
        # Load model
        with open('models/privacy_concern_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessors
        with open('data/clean/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('data/clean/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return model, label_encoders, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found. Please train the models first!")
        return None, None, None

def create_input_form():
    """Create user input form for privacy concern prediction"""
    st.header("🔒 Privacy Concern Level Assessment")
    st.markdown("### Evaluate your privacy risk when using AI tools")
    
    
    if is_profile_complete():
        return create_auto_filled_form("privacy")
    
    st.warning("⚠️ Complete your profile in the sidebar to auto-fill this form!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="privacy_gender")
        age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-55", "56+"], key="privacy_age")
        country = st.selectbox("Country", ["USA", "Canada", "UK", "Germany", "India", "Other"], key="privacy_country")
        profession = st.selectbox("Profession", 
                                ["Software Developer", "Student", "Marketing Manager", "Doctor", 
                                 "Teacher", "Engineer", "Designer", "Other"], key="privacy_profession")
        education_level = st.selectbox("Education Level", 
                                     ["High School", "Some College", "Bachelor Degree", 
                                      "Master Degree", "PhD"], key="privacy_education")
    
    with col2:
        st.subheader("AI Usage Patterns")
        tech_experience = st.selectbox("Tech Experience Level", ["Beginner", "Intermediate", "Advanced"], key="privacy_tech")
        ai_tool = st.selectbox("Primary AI Tool", ["ChatGPT", "Claude", "Copilot", "Gemini", "Other"], key="privacy_tool")
        purpose = st.selectbox("Purpose of AI Use", 
                              ["Work Tasks", "Learning", "Content Creation", "Research", "Entertainment"], key="privacy_purpose")
        hours_per_day = st.slider("Hours spent per day on AI", 0.0, 12.0, 2.0, 0.1, key="privacy_hours")
        frequency = st.selectbox("Weekly Usage Frequency", 
                                ["Daily", "5-6 times/week", "3-4 times/week", "1-2 times/week", "Rarely"], key="privacy_freq")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Usage Details")
        months_using = st.slider("Months using AI", 1, 60, 12, key="privacy_months")
        task_complexity = st.selectbox("Task Complexity Level", ["Simple", "Moderate", "Complex"], key="privacy_complexity")
        verification = st.selectbox("How often do you verify AI outputs?", 
                                  ["Always", "Often", "Sometimes", "Rarely", "Never"], key="privacy_verify")
        confidence = st.slider("Confidence in AI outputs (1-10)", 1, 10, 7, key="privacy_confidence")
    
    with col4:
        st.subheader("Privacy Behavior")
        critical_thinking_rating = st.slider("Critical thinking self-rating (1-10)", 1, 10, 7, key="privacy_critical")
        privacy_awareness = st.slider("Privacy awareness level (1-10)", 1, 10, 7, key="privacy_awareness")
        shares_info = st.selectbox("Do you share personal info with AI?", 
                                 ["Never", "Rarely", "Sometimes", "Often", "Always"], key="privacy_shares")
        reads_news = st.selectbox("Do you read AI-related news?", 
                                ["Never", "Rarely", "Sometimes", "Often", "Always"], key="privacy_news")
    
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
    
    # Encode categorical variables
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

def interpret_privacy_score(score):
    """Interpret the privacy concern level score"""
    if score <= 2.0:
        return "🔴 Very Low Privacy Awareness", "You may be sharing too much personal data with AI systems."
    elif score <= 4.0:
        return "🟠 Low Privacy Awareness", "Consider being more cautious about personal information sharing."
    elif score <= 6.0:
        return "🟡 Moderate Privacy Awareness", "Good balance, but room for improvement in privacy practices."
    elif score <= 8.0:
        return "🟢 High Privacy Awareness", "Excellent privacy practices with AI tools."
    else:
        return "🟢 Very High Privacy Awareness", "Outstanding privacy consciousness and data protection habits."

def show_privacy_recommendations(awareness_level, score):
    """Show personalized privacy recommendations"""
    st.subheader("🛡️ Privacy Protection Recommendations")
    
    if score <= 4.0:
        recommendations = [
            "🚨 **Immediate Actions Needed:**",
            "• Stop sharing personal identifiable information (PII) with AI tools",
            "• Review and delete conversation history in AI platforms",
            "• Read privacy policies of AI tools you use regularly",
            "• Use pseudonyms or fictional scenarios instead of real personal data",
            "• Enable privacy settings and opt-out options where available",
            "• Consider using AI tools that process data locally",
            "• Be aware that AI conversations may be stored and analyzed"
        ]
    elif score <= 6.0:
        recommendations = [
            "⚖️ **Balanced Approach Recommended:**",
            "• Continue being selective about information sharing",
            "• Regularly review privacy settings in AI platforms",
            "• Use work-related or generic examples instead of personal ones",
            "• Stay updated on AI privacy policies and changes",
            "• Consider the sensitivity of information before sharing",
            "• Use privacy-focused AI alternatives when possible",
            "• Enable two-factor authentication on AI accounts"
        ]
    else:
        recommendations = [
            "✅ **Maintain Current Excellent Practices:**",
            "• Continue your privacy-conscious approach to AI usage",
            "• Share your knowledge with others about AI privacy",
            "• Stay informed about emerging privacy threats",
            "• Consider using only local or open-source AI tools for sensitive work",
            "• Regularly audit your AI tool usage and data sharing",
            "• Help others understand the importance of AI privacy",
            "• Consider privacy advocacy or education roles"
        ]
    
    for rec in recommendations:
        if rec.startswith("•"):
            st.write(rec)
        else:
            st.markdown(f"**{rec}**")

def create_privacy_visualization(score):
    """Create a privacy awareness gauge"""
   
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Privacy Concern Level"},
        delta = {'reference': 5.0},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 2], 'color': "darkred"},
                {'range': [2, 4], 'color': "red"},
                {'range': [4, 6], 'color': "orange"},
                {'range': [6, 8], 'color': "lightgreen"},
                {'range': [8, 10], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 6.0
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def show_privacy_tips():
    """Show general privacy tips"""
    st.subheader("🎯 General Privacy Best Practices with AI")
    
    tips = {
        "Data Minimization": "Only share the minimum information necessary to get helpful responses",
        "Anonymization": "Replace real names, locations, and identifiers with generic alternatives",
        "Context Awareness": "Consider who might have access to your AI conversations",
        "Regular Cleanup": "Periodically delete your conversation history and account data",
        "Tool Selection": "Choose AI tools with strong privacy policies and data protection",
        "Local Processing": "Prefer AI tools that process data on your device when possible",
        "Educational Use": "Use AI for learning rather than processing sensitive personal data",
        "Policy Updates": "Stay informed about changes in AI platform privacy policies"
    }
    
    for tip, description in tips.items():
        with st.expander(f"💡 {tip}"):
            st.write(description)


def run_privacy_tab():
    """Main function to run the privacy concern prediction tab"""
    
  
    if are_predictions_available():
        prediction_value = get_prediction_value('privacy_concern')
        if prediction_value is not None:
           
            st.header("🔒 Privacy Concern Level Assessment")
            st.success("✅ Prediction automatically generated from your saved profile!")
            
            # Display results
            st.markdown("---")
            st.header("📊 Privacy Assessment Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Privacy gauge
                fig = create_prediction_chart(prediction_value, "Privacy Concern Level", "privacy_concern")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Privacy interpretation
                awareness_level, interpretation = interpret_privacy_score(prediction_value)
                st.metric("Privacy Awareness Level", awareness_level)
                st.write(f"**Score:** {prediction_value:.2f}/10")
                st.write(f"**Interpretation:** {interpretation}")
            
            # Get profile for additional insights
            from shared_profile import get_profile_data
            user_input = get_profile_data()
            
            # Recommendations
            show_privacy_recommendations(awareness_level, prediction_value)
            
            # Privacy behavior analysis
            st.subheader("📊 Your Privacy Behavior Analysis")
            
            privacy_factors = {
                "Information Sharing": user_input['shares_personal_info_with_ai'],
                "Privacy Awareness": f"{user_input['privacy_awareness_level']}/10",
                "News Consumption": user_input['reads_ai_related_news'],
                "AI Tool Choice": user_input['ai_tool_primary'],
                "Usage Intensity": f"{user_input['hours_spent_per_day_on_ai']:.1f} hrs/day"
            }
            
            col1, col2 = st.columns(2)
            with col1:
                for factor, value in list(privacy_factors.items())[:3]:
                    st.info(f"**{factor}:** {value}")
            
            with col2:
                for factor, value in list(privacy_factors.items())[3:]:
                    st.info(f"**{factor}:** {value}")
            
            # Show general privacy tips
            show_privacy_tips()
            return
    
    
    model, label_encoders, scaler = load_model_and_preprocessors()
    
    if model is None:
        st.warning("⚠️ Please train the models first by running the training script!")
        return
    
    
    user_input = create_input_form()
    
    # Only proceed with prediction if we have user input
    if user_input is None:
        return
    
    # Prediction button
    if st.button("🔮 Assess Privacy Concern Level", type="primary"):
        try:
            # Preprocess input
            X_processed = preprocess_input(user_input, label_encoders, scaler)
            
            # Make prediction
            prediction = model.predict(X_processed)[0]
            
            # Display results
            st.markdown("---")
            st.header("📊 Privacy Assessment Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Privacy gauge
                fig = create_privacy_visualization(prediction)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Privacy interpretation
                awareness_level, interpretation = interpret_privacy_score(prediction)
                st.metric("Privacy Awareness Level", awareness_level)
                st.write(f"**Score:** {prediction:.2f}/10")
                st.write(f"**Interpretation:** {interpretation}")
            
            # Recommendations
            show_privacy_recommendations(awareness_level, prediction)
            
            # Privacy behavior analysis
            st.subheader("📊 Your Privacy Behavior Analysis")
            
            privacy_factors = {
                "Information Sharing": user_input['shares_personal_info_with_ai'],
                "Privacy Awareness": f"{user_input['privacy_awareness_level']}/10",
                "News Consumption": user_input['reads_ai_related_news'],
                "AI Tool Choice": user_input['ai_tool_primary'],
                "Usage Intensity": f"{user_input['hours_spent_per_day_on_ai']:.1f} hrs/day"
            }
            
            col1, col2 = st.columns(2)
            with col1:
                for factor, value in list(privacy_factors.items())[:3]:
                    st.info(f"**{factor}:** {value}")
            
            with col2:
                for factor, value in list(privacy_factors.items())[3:]:
                    st.info(f"**{factor}:** {value}")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that all required model files are present and properly trained.")
    
    # Show general privacy tips
    show_privacy_tips()

if __name__ == "__main__":
    run_privacy_tab()
