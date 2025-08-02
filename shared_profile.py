"""
Shared User Profile Management

This module handles the centralized user profile that auto-populates all prediction tabs.
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import os

def initialize_profile():
    """Initialize the user profile in session state if it doesn't exist"""
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            # Demographics
            'gender': None,
            'age_group': None,
            'country': None,
            'profession': None,
            'education_level': None,
            
            # AI Usage Patterns
            'tech_experience_level': None,
            'ai_tool_primary': None,
            'purpose_of_ai_use': None,
            'hours_spent_per_day_on_ai': 2.0,
            'frequency_of_use_weekly': None,
            
            # Usage Details
            'months_using_ai': 12,
            'task_complexity_level': None,
            'verification_frequency': None,
            'confidence_in_ai_outputs': 7,
            
            # Self Assessment
            'critical_thinking_self_rating': 7,
            'privacy_awareness_level': 7,
            'shares_personal_info_with_ai': None,
            'reads_ai_related_news': None,
            
            # Profile status
            'profile_completed': False
        }
    
    # Initialize predictions storage
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {
            'critical_thinking': None,
            'ai_overdependence': None,
            'incorrect_info': None,
            'privacy_concern': None,
            'predictions_generated': False
        }

def load_all_models():
    """Load all trained models and preprocessing objects"""
    try:
        models = {}
        
        # Load all models
        model_files = {
            'critical_thinking': 'models/critical_thinking_model.pkl',
            'ai_overdependence': 'models/ai_overdependence_model.pkl',
            'incorrect_info': 'models/incorrect_info_model.pkl',
            'privacy_concern': 'models/privacy_concern_model.pkl'
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
            else:
                return None, None, None
        
        # Load preprocessors
        with open('data/clean/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('data/clean/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return models, label_encoders, scaler
    except FileNotFoundError:
        return None, None, None

def preprocess_input(user_input, label_encoders, scaler):
    """Preprocess user input for model prediction"""
    # Create a dataframe with user input
    df = pd.DataFrame([user_input])
    
    # Categorical columns to encode
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
    
    # Select features in the same order as training
    feature_cols = (['hours_spent_per_day_on_ai', 'months_using_ai', 'confidence_in_ai_outputs',
                    'critical_thinking_self_rating', 'privacy_awareness_level'] + 
                   [f'{col}_encoded' for col in categorical_cols])
    
    X = df[feature_cols]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled

def generate_all_predictions():
    """Generate predictions for all risk categories automatically"""
    if not is_profile_complete():
        return False
    
    # Load models
    models, label_encoders, scaler = load_all_models()
    
    if models is None:
        st.error("❌ Models not found. Please train the models first!")
        return False
    
    try:
        # Get user profile data
        profile = get_profile_data()
        
        # Preprocess input
        X_processed = preprocess_input(profile, label_encoders, scaler)
        
        # Generate predictions for all models
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(X_processed)[0]
            predictions[model_name] = float(prediction)
        
        # Store predictions in session state
        st.session_state.predictions.update(predictions)
        st.session_state.predictions['predictions_generated'] = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error generating predictions: {str(e)}")
        return False

def create_prediction_chart(prediction_value, chart_title, chart_type="critical_thinking"):
    """Create a plotly chart for predictions"""
    
    # Define color schemes and interpretations for different chart types
    chart_configs = {
        'critical_thinking': {
            'colors': [
                {'range': [0, 2], 'color': "lightgreen"},
                {'range': [2, 4], 'color': "yellow"},
                {'range': [4, 6], 'color': "orange"},
                {'range': [6, 8], 'color': "red"},
                {'range': [8, 10], 'color': "darkred"}
            ],
            'bar_color': "darkblue",
            'threshold': 7.0
        },
        'ai_overdependence': {
            'colors': [
                {'range': [0, 2], 'color': "lightgreen"},
                {'range': [2, 4], 'color': "yellow"},
                {'range': [4, 6], 'color': "orange"},
                {'range': [6, 8], 'color': "red"},
                {'range': [8, 10], 'color': "darkred"}
            ],
            'bar_color': "darkred",
            'threshold': 6.0
        },
        'incorrect_info': {
            'colors': [
                {'range': [0, 2], 'color': "lightgreen"},
                {'range': [2, 4], 'color': "yellow"},
                {'range': [4, 6], 'color': "orange"},
                {'range': [6, 8], 'color': "red"},
                {'range': [8, 10], 'color': "darkred"}
            ],
            'bar_color': "darkred",
            'threshold': 6.0
        },
        'privacy_concern': {
            'colors': [
                {'range': [0, 2], 'color': "darkred"},
                {'range': [2, 4], 'color': "red"},
                {'range': [4, 6], 'color': "orange"},
                {'range': [6, 8], 'color': "lightgreen"},
                {'range': [8, 10], 'color': "green"}
            ],
            'bar_color': "darkgreen",
            'threshold': 6.0
        }
    }
    
    config = chart_configs.get(chart_type, chart_configs['critical_thinking'])
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': chart_title},
        delta = {'reference': 5.0},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': config['bar_color']},
            'steps': config['colors'],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': config['threshold']
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def create_profile_form():
    """Create the user profile form in the sidebar"""
    
    st.sidebar.markdown("## 👤 User Profile")
    
    # Check if profile is completed
    profile = st.session_state.user_profile
    profile_completed = profile.get('profile_completed', False)
    
    if profile_completed:
        st.sidebar.success("✅ Profile Complete!")
        
        # Show profile summary
        with st.sidebar.expander("📋 View Profile Summary", expanded=False):
            st.write(f"**Name:** {profile.get('gender', 'Not set')}, {profile.get('age_group', 'Not set')}")
            st.write(f"**Location:** {profile.get('country', 'Not set')}")
            st.write(f"**Profession:** {profile.get('profession', 'Not set')}")
            st.write(f"**Education:** {profile.get('education_level', 'Not set')}")
            st.write(f"**AI Tool:** {profile.get('ai_tool_primary', 'Not set')}")
            st.write(f"**Daily Usage:** {profile.get('hours_spent_per_day_on_ai', 0):.1f} hours")
        
        # Edit profile button
        if st.sidebar.button("✏️ Edit Profile"):
            st.session_state.user_profile['profile_completed'] = False
            st.rerun()
    
    else:
        st.sidebar.info("📝 Complete your profile to auto-fill prediction forms")
        
        # Profile form
        with st.sidebar.form("user_profile_form"):
            st.markdown("### Demographics")
            
            gender = st.selectbox("Gender", 
                                ["", "Male", "Female", "Other"], 
                                index=0)
            
            age_group = st.selectbox("Age Group", 
                                   ["", "18-25", "26-35", "36-45", "46-55", "56+"], 
                                   index=0)
            
            country = st.selectbox("Country", 
                                 ["", "USA", "Canada", "UK", "Germany", "India", "Other"], 
                                 index=0)
            
            profession = st.selectbox("Profession", 
                                    ["", "Software Developer", "Student", "Marketing Manager", 
                                     "Doctor", "Teacher", "Engineer", "Designer", "Other"], 
                                    index=0)
            
            education_level = st.selectbox("Education Level", 
                                         ["", "High School", "Some College", "Bachelor Degree", 
                                          "Master Degree", "PhD"], 
                                         index=0)
            
            st.markdown("### AI Usage Patterns")
            
            tech_experience = st.selectbox("Tech Experience Level", 
                                         ["", "Beginner", "Intermediate", "Advanced"], 
                                         index=0)
            
            ai_tool = st.selectbox("Primary AI Tool", 
                                 ["", "ChatGPT", "Claude", "Copilot", "Gemini", "Other"], 
                                 index=0)
            
            purpose = st.selectbox("Purpose of AI Use", 
                                 ["", "Work Tasks", "Learning", "Content Creation", 
                                  "Research", "Entertainment"], 
                                 index=0)
            
            hours_per_day = st.slider("Hours spent per day on AI", 0.0, 12.0, 2.0, 0.1)
            
            frequency = st.selectbox("Weekly Usage Frequency", 
                                   ["", "Daily", "5-6 times/week", "3-4 times/week", 
                                    "1-2 times/week", "Rarely"], 
                                   index=0)
            
            st.markdown("### Usage Details")
            
            months_using = st.slider("Months using AI", 1, 60, 12)
            
            task_complexity = st.selectbox("Task Complexity Level", 
                                         ["", "Simple", "Moderate", "Complex"], 
                                         index=0)
            
            verification = st.selectbox("How often do you verify AI outputs?", 
                                      ["", "Always", "Often", "Sometimes", "Rarely", "Never"], 
                                      index=0)
            
            confidence = st.slider("Confidence in AI outputs (1-10)", 1, 10, 7)
            
            st.markdown("### Self Assessment")
            
            critical_thinking_rating = st.slider("Critical thinking self-rating (1-10)", 1, 10, 7)
            
            privacy_awareness = st.slider("Privacy awareness level (1-10)", 1, 10, 7)
            
            shares_info = st.selectbox("Do you share personal info with AI?", 
                                     ["", "Never", "Rarely", "Sometimes", "Often", "Always"], 
                                     index=0)
            
            reads_news = st.selectbox("Do you read AI-related news?", 
                                    ["", "Never", "Rarely", "Sometimes", "Often", "Always"], 
                                    index=0)
            
            # Form submit button
            submitted = st.form_submit_button("💾 Save Profile", type="primary")
            
            if submitted:
                # Validate required fields
                required_fields = [gender, age_group, country, profession, education_level,
                                 tech_experience, ai_tool, purpose, frequency, task_complexity,
                                 verification, shares_info, reads_news]
                
                if all(field for field in required_fields):
                    # Save to session state
                    st.session_state.user_profile.update({
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
                        'reads_ai_related_news': reads_news,
                        'profile_completed': True
                    })
                    
                    st.success("✅ Profile saved successfully!")
                    
                    # Automatically generate predictions for all tabs
                    with st.spinner("🔮 Generating predictions for all risk categories..."):
                        if generate_all_predictions():
                            st.success("🎉 All predictions generated successfully! Check all tabs for your results.")
                        else:
                            st.warning("⚠️ Could not generate predictions. Please ensure models are trained.")
                    
                    st.rerun()
                else:
                    st.error("❌ Please fill in all required fields!")

def get_profile_data():
    """Get the current user profile data"""
    return st.session_state.get('user_profile', {})

def is_profile_complete():
    """Check if the user profile is complete"""
    profile = get_profile_data()
    return profile.get('profile_completed', False)

def get_predictions():
    """Get all generated predictions"""
    return st.session_state.get('predictions', {})

def are_predictions_available():
    """Check if predictions have been generated"""
    predictions = get_predictions()
    return predictions.get('predictions_generated', False)

def get_prediction_value(prediction_type):
    """Get a specific prediction value"""
    predictions = get_predictions()
    return predictions.get(prediction_type, None)

def create_auto_filled_form(tab_name=""):
    """Create a form that's auto-filled with profile data"""
    profile = get_profile_data()
    
    if not is_profile_complete():
        st.warning("⚠️ Complete your profile in the sidebar to auto-fill this form!")
        return None
    
    st.info("✅ Form auto-filled from your saved profile. You can modify any field below.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", 
                            ["Male", "Female", "Other"], 
                            index=["Male", "Female", "Other"].index(profile.get('gender', 'Male')),
                            key=f"{tab_name}_gender")
        
        age_group = st.selectbox("Age Group", 
                               ["18-25", "26-35", "36-45", "46-55", "56+"],
                               index=["18-25", "26-35", "36-45", "46-55", "56+"].index(profile.get('age_group', '18-25')),
                               key=f"{tab_name}_age")
        
        country = st.selectbox("Country", 
                             ["USA", "Canada", "UK", "Germany", "India", "Other"],
                             index=["USA", "Canada", "UK", "Germany", "India", "Other"].index(profile.get('country', 'USA')),
                             key=f"{tab_name}_country")
        
        profession = st.selectbox("Profession", 
                                ["Software Developer", "Student", "Marketing Manager", "Doctor", 
                                 "Teacher", "Engineer", "Designer", "Other"],
                                index=["Software Developer", "Student", "Marketing Manager", "Doctor", 
                                      "Teacher", "Engineer", "Designer", "Other"].index(profile.get('profession', 'Software Developer')),
                                key=f"{tab_name}_profession")
        
        education_level = st.selectbox("Education Level", 
                                     ["High School", "Some College", "Bachelor Degree", 
                                      "Master Degree", "PhD"],
                                     index=["High School", "Some College", "Bachelor Degree", 
                                           "Master Degree", "PhD"].index(profile.get('education_level', 'Bachelor Degree')),
                                     key=f"{tab_name}_education")
    
    with col2:
        st.subheader("AI Usage Patterns")
        tech_experience = st.selectbox("Tech Experience Level", 
                                     ["Beginner", "Intermediate", "Advanced"],
                                     index=["Beginner", "Intermediate", "Advanced"].index(profile.get('tech_experience_level', 'Intermediate')),
                                     key=f"{tab_name}_tech")
        
        ai_tool = st.selectbox("Primary AI Tool", 
                             ["ChatGPT", "Claude", "Copilot", "Gemini", "Other"],
                             index=["ChatGPT", "Claude", "Copilot", "Gemini", "Other"].index(profile.get('ai_tool_primary', 'ChatGPT')),
                             key=f"{tab_name}_tool")
        
        purpose = st.selectbox("Purpose of AI Use", 
                             ["Work Tasks", "Learning", "Content Creation", "Research", "Entertainment"],
                             index=["Work Tasks", "Learning", "Content Creation", "Research", "Entertainment"].index(profile.get('purpose_of_ai_use', 'Work Tasks')),
                             key=f"{tab_name}_purpose")
        
        hours_per_day = st.slider("Hours spent per day on AI", 0.0, 12.0, 
                                 profile.get('hours_spent_per_day_on_ai', 2.0), 0.1, 
                                 key=f"{tab_name}_hours")
        
        frequency = st.selectbox("Weekly Usage Frequency", 
                               ["Daily", "5-6 times/week", "3-4 times/week", "1-2 times/week", "Rarely"],
                               index=["Daily", "5-6 times/week", "3-4 times/week", "1-2 times/week", "Rarely"].index(profile.get('frequency_of_use_weekly', 'Daily')),
                               key=f"{tab_name}_freq")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Usage Details")
        months_using = st.slider("Months using AI", 1, 60, 
                                profile.get('months_using_ai', 12), 
                                key=f"{tab_name}_months")
        
        task_complexity = st.selectbox("Task Complexity Level", 
                                     ["Simple", "Moderate", "Complex"],
                                     index=["Simple", "Moderate", "Complex"].index(profile.get('task_complexity_level', 'Moderate')),
                                     key=f"{tab_name}_complexity")
        
        verification = st.selectbox("How often do you verify AI outputs?", 
                                  ["Always", "Often", "Sometimes", "Rarely", "Never"],
                                  index=["Always", "Often", "Sometimes", "Rarely", "Never"].index(profile.get('verification_frequency', 'Sometimes')),
                                  key=f"{tab_name}_verify")
        
        confidence = st.slider("Confidence in AI outputs (1-10)", 1, 10, 
                              profile.get('confidence_in_ai_outputs', 7), 
                              key=f"{tab_name}_confidence")
    
    with col4:
        st.subheader("Self Assessment")
        critical_thinking_rating = st.slider("Critical thinking self-rating (1-10)", 1, 10, 
                                           profile.get('critical_thinking_self_rating', 7), 
                                           key=f"{tab_name}_critical")
        
        privacy_awareness = st.slider("Privacy awareness level (1-10)", 1, 10, 
                                    profile.get('privacy_awareness_level', 7), 
                                    key=f"{tab_name}_awareness")
        
        shares_info = st.selectbox("Do you share personal info with AI?", 
                                 ["Never", "Rarely", "Sometimes", "Often", "Always"],
                                 index=["Never", "Rarely", "Sometimes", "Often", "Always"].index(profile.get('shares_personal_info_with_ai', 'Rarely')),
                                 key=f"{tab_name}_shares")
        
        reads_news = st.selectbox("Do you read AI-related news?", 
                                ["Never", "Rarely", "Sometimes", "Often", "Always"],
                                index=["Never", "Rarely", "Sometimes", "Often", "Always"].index(profile.get('reads_ai_related_news', 'Sometimes')),
                                key=f"{tab_name}_news")
    
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

def reset_profile():
    """Reset the user profile"""
    if 'user_profile' in st.session_state:
        del st.session_state.user_profile
    initialize_profile()