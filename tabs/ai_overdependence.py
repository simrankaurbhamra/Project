"""
AI Overdependence Risk Prediction Tab

This module handles the prediction and visualization of AI overdependence risk scores.
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
       
        with open('models/ai_overdependence_model.pkl', 'rb') as f:
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
    """Create user input form for AI overdependence risk prediction"""
    st.header("⚡ AI Overdependence Risk Assessment")
    st.markdown("### Evaluate your dependency level on AI tools and systems")
    
    if is_profile_complete():
        return create_auto_filled_form("ai_overdependence")
    st.warning("⚠️ Complete your profile in the sidebar to auto-fill this form!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="overdep_gender")
        age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-55", "56+"], key="overdep_age")
        country = st.selectbox("Country", ["USA", "Canada", "UK", "Germany", "India", "Other"], key="overdep_country")
        profession = st.selectbox("Profession", 
                                ["Software Developer", "Student", "Marketing Manager", "Doctor", 
                                 "Teacher", "Engineer", "Designer", "Other"], key="overdep_profession")
        education_level = st.selectbox("Education Level", 
                                     ["High School", "Some College", "Bachelor Degree", 
                                      "Master Degree", "PhD"], key="overdep_education")
    
    with col2:
        st.subheader("AI Usage Patterns")
        tech_experience = st.selectbox("Tech Experience Level", ["Beginner", "Intermediate", "Advanced"], key="overdep_tech")
        ai_tool = st.selectbox("Primary AI Tool", ["ChatGPT", "Claude", "Copilot", "Gemini", "Other"], key="overdep_tool")
        purpose = st.selectbox("Purpose of AI Use", 
                              ["Work Tasks", "Learning", "Content Creation", "Research", "Entertainment"], key="overdep_purpose")
        hours_per_day = st.slider("Hours spent per day on AI", 0.0, 12.0, 2.0, 0.1, key="overdep_hours")
        frequency = st.selectbox("Weekly Usage Frequency", 
                                ["Daily", "5-6 times/week", "3-4 times/week", "1-2 times/week", "Rarely"], key="overdep_freq")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Usage Details")
        months_using = st.slider("Months using AI", 1, 60, 12, key="overdep_months")
        task_complexity = st.selectbox("Task Complexity Level", ["Simple", "Moderate", "Complex"], key="overdep_complexity")
        verification = st.selectbox("How often do you verify AI outputs?", 
                                  ["Always", "Often", "Sometimes", "Rarely", "Never"], key="overdep_verify")
        confidence = st.slider("Confidence in AI outputs (1-10)", 1, 10, 7, key="overdep_confidence")
    
    with col4:
        st.subheader("Dependency Indicators")
        critical_thinking_rating = st.slider("Critical thinking self-rating (1-10)", 1, 10, 7, key="overdep_critical")
        privacy_awareness = st.slider("Privacy awareness level (1-10)", 1, 10, 7, key="overdep_awareness")
        shares_info = st.selectbox("Do you share personal info with AI?", 
                                 ["Never", "Rarely", "Sometimes", "Often", "Always"], key="overdep_shares")
        reads_news = st.selectbox("Do you read AI-related news?", 
                                ["Never", "Rarely", "Sometimes", "Often", "Always"], key="overdep_news")
    
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
                
                df[f'{col}_encoded'] = 0 
    
    feature_cols = (['hours_spent_per_day_on_ai', 'months_using_ai', 'confidence_in_ai_outputs',
                    'critical_thinking_self_rating', 'privacy_awareness_level'] + 
                   [f'{col}_encoded' for col in categorical_cols])
    
    X = df[feature_cols]
    
    
    X_scaled = scaler.transform(X)
    
    return X_scaled

def interpret_overdependence_score(score):
    """Interpret the AI overdependence risk score"""
    if score <= 2.0:
        return "🟢 Very Low Overdependence", "Healthy independence from AI tools. Great balance!"
    elif score <= 4.0:
        return "🟡 Low Overdependence", "Good balance between AI use and independent thinking."
    elif score <= 6.0:
        return "🟠 Moderate Overdependence", "Noticeable dependency patterns. Consider reducing reliance."
    elif score <= 8.0:
        return "🔴 High Overdependence", "Strong dependency on AI. Immediate steps needed to regain independence."
    else:
        return "🚨 Very High Overdependence", "Critical dependency levels. Professional guidance may be helpful."

def show_overdependence_recommendations(risk_level, score, user_input):
    """Show personalized recommendations for managing AI overdependence"""
    st.subheader("💡 Personalized Action Plan")
    
    if score <= 4.0:
        recommendations = [
            "✅ **Maintain Current Healthy Balance:**",
            "• Continue your current balanced approach to AI usage",
            "• Keep practicing tasks without AI assistance regularly",
            "• Share your balanced approach with others who may be struggling",
            "• Stay aware of usage patterns and set reasonable limits",
            "• Use AI as a tool to enhance, not replace, your capabilities",
        ]
    elif score <= 6.0:
        recommendations = [
            "⚖️ **Moderate Risk - Take Preventive Action:**",
            "• Set specific daily/weekly limits for AI tool usage",
            "• Designate 'AI-free' hours or days in your routine",
            "• Practice completing tasks the traditional way first, then use AI",
            "• Keep a usage log to track your AI dependency patterns",
            "• Find alternative tools or methods for some tasks",
            "• Join communities focused on balanced technology use",
        ]
    else:
        recommendations = [
            "🚨 **High Risk - Immediate Intervention Needed:**",
            "• Implement strict usage limits immediately (max 2-3 hours/day)",
            "• Start with AI-free periods: begin with 2-4 hours daily",
            "• Relearn fundamental skills without AI assistance",
            "• Seek support from colleagues, friends, or professionals",
            "• Use app blockers or time management tools",
            "• Consider a digital detox or AI-free weekend",
            "• Focus on building confidence in your natural abilities",
            "• Track daily progress in reducing dependency",
        ]
    
    for rec in recommendations:
        if rec.startswith("•"):
            st.write(rec)
        else:
            st.markdown(f"**{rec}**")
    
    st.subheader("🎯 Targeted Recommendations Based on Your Profile")
    
    specific_tips = []
    
    if user_input['hours_spent_per_day_on_ai'] > 5:
        specific_tips.append("⏰ **High Usage Alert**: Your daily usage is very high. Start by reducing usage by 30 minutes each week.")
    
    if user_input['verification_frequency'] in ['Rarely', 'Never']:
        specific_tips.append("🔍 **Verification Gap**: You rarely verify AI outputs. This increases dependency risk.")
    
    if user_input['confidence_in_ai_outputs'] >= 8:
        specific_tips.append("⚠️ **Overconfidence Warning**: Very high confidence in AI may indicate overdependence.")
    
    if user_input['task_complexity_level'] == 'Complex' and user_input['hours_spent_per_day_on_ai'] > 4:
        specific_tips.append("🧩 **Complex Task Dependency**: Consider tackling complex problems step-by-step without AI first.")
    
    for tip in specific_tips:
        st.warning(tip)

def show_healthy_usage_tips():
    """Show tips for healthy AI usage"""
    st.subheader("🌱 Building Healthy AI Usage Habits")
    
    healthy_habits = {
        "The 70-30 Rule": "Use AI for 30% of tasks, handle 70% independently to maintain skills",
        "Verification First": "Always fact-check and verify AI outputs, especially for important decisions",
        "Skill Maintenance": "Regularly practice core skills without AI assistance",
        "Gradual Reduction": "If overdependent, reduce usage by 15-20% weekly until balanced",
        "Alternative Methods": "Learn non-AI ways to accomplish the same tasks",
        "Mindful Usage": "Ask yourself 'Do I really need AI for this?' before using it",
        "Time Boundaries": "Set specific times for AI use and stick to them",
        "Quality over Quantity": "Focus on meaningful AI interactions rather than constant usage"
    }
    
    for habit, description in healthy_habits.items():
        with st.expander(f"💚 {habit}"):
            st.write(description)

def run_ai_overdependence_tab():
    """Main function to run the AI overdependence risk prediction tab"""
    
    if are_predictions_available():
        prediction_value = get_prediction_value('ai_overdependence')
        if prediction_value is not None:
       
            st.header("⚡ AI Overdependence Risk Assessment")
            st.success("✅ Prediction automatically generated from your saved profile!")
            
          
            st.markdown("---")
            st.header("📊 Overdependence Assessment Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
            
                fig = create_prediction_chart(prediction_value, "AI Overdependence Risk Score", "ai_overdependence")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                risk_level, interpretation = interpret_overdependence_score(prediction_value)
                st.metric("Overdependence Risk Level", risk_level)
                st.write(f"**Score:** {prediction_value:.2f}/10")
                st.write(f"**Interpretation:** {interpretation}")
            
            
            from shared_profile import get_profile_data
            user_input = get_profile_data()
            
            st.subheader("📈 Your AI Usage Pattern Analysis")
            usage_metrics = {
                "Daily Usage": f"{user_input['hours_spent_per_day_on_ai']:.1f} hours",
                "Experience": f"{user_input['months_using_ai']} months",
                "Frequency": user_input['frequency_of_use_weekly'],
                "Confidence Level": f"{user_input['confidence_in_ai_outputs']}/10",
                "Verification Rate": user_input['verification_frequency']
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Daily Usage:** {usage_metrics['Daily Usage']}")
                st.info(f"**Experience:** {usage_metrics['Experience']}")
            with col2:
                st.info(f"**Frequency:** {usage_metrics['Frequency']}")
                st.info(f"**Confidence:** {usage_metrics['Confidence Level']}")
            with col3:
                st.info(f"**Verification:** {usage_metrics['Verification Rate']}")
            
            
            show_overdependence_recommendations(risk_level, prediction_value, user_input)
            
           
            show_healthy_usage_tips()
            return
    
    model, label_encoders, scaler = load_model_and_preprocessors()
    
    if model is None:
        st.warning("⚠️ Please train the models first by running the training script!")
        return
    
    user_input = create_input_form()
    
    
    if user_input is None:
        return
    
   
    if st.button("🔮 Assess AI Overdependence Risk", type="primary"):
        try:
           
            X_processed = preprocess_input(user_input, label_encoders, scaler)
            
            # Make prediction
            prediction = model.predict(X_processed)[0]
            
            # Display results
            st.markdown("---")
            st.header("📊 Overdependence Assessment Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
               
                fig = create_prediction_chart(prediction, "AI Overdependence Risk Score", "ai_overdependence")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk interpretation
                risk_level, interpretation = interpret_overdependence_score(prediction)
                st.metric("Overdependence Risk Level", risk_level)
                st.write(f"**Score:** {prediction:.2f}/10")
                st.write(f"**Interpretation:** {interpretation}")
            
            # Usage pattern analysis
            st.subheader("📈 Your AI Usage Pattern Analysis")
            usage_metrics = {
                "Daily Usage": f"{user_input['hours_spent_per_day_on_ai']:.1f} hours",
                "Experience": f"{user_input['months_using_ai']} months",
                "Frequency": user_input['frequency_of_use_weekly'],
                "Confidence Level": f"{user_input['confidence_in_ai_outputs']}/10",
                "Verification Rate": user_input['verification_frequency']
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Daily Usage:** {usage_metrics['Daily Usage']}")
                st.info(f"**Experience:** {usage_metrics['Experience']}")
            with col2:
                st.info(f"**Frequency:** {usage_metrics['Frequency']}")
                st.info(f"**Confidence:** {usage_metrics['Confidence Level']}")
            with col3:
                st.info(f"**Verification:** {usage_metrics['Verification Rate']}")
            
            # Recommendations
            show_overdependence_recommendations(risk_level, prediction, user_input)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that all required model files are present and properly trained.")
    
   
    show_healthy_usage_tips()

if __name__ == "__main__":
    run_ai_overdependence_tab()