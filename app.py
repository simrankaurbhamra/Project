"""
AI Risk Prediction Dashboard

A comprehensive Streamlit application for predicting various AI usage risks:
1. Critical Thinking Risk Score
2. AI Overdependence Risk
3. Incorrect Information Risk Score  
4. Privacy Concern Level

Author: AI Risk Assessment Team
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'tabs'))

try:
    from tabs.critical_thinking import run_critical_thinking_tab
    from tabs.ai_overdependence import run_ai_overdependence_tab
    from tabs.incorrect_info import run_incorrect_info_tab
    from tabs.privacy import run_privacy_tab
    from tabs.graphs import run_graphs_tab
    from shared_profile import initialize_profile, create_profile_form
except ImportError as e:
    st.error(f"Error importing tab modules: {e}")
    st.stop()

def main():
    """Main application function"""
    
    initialize_profile()
    
 
    st.set_page_config(
        page_title="AI Risk Prediction Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .tab-content {
        padding: 2rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🤖 AI Risk Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Assess your AI usage risks and get personalized recommendations</p>', unsafe_allow_html=True)
    
  
    with st.sidebar:
      
        create_profile_form()
        
        st.markdown("---")
        st.markdown("## 📊 About This Dashboard")
        st.info("""
        This dashboard helps you assess various risks associated with AI usage:
        
        **📈 Data Visualizations**: Interactive charts and graphs showing dataset insights
        
        **🧠 Critical Thinking Risk**: Measures potential impact on analytical skills
        
        **⚡ AI Overdependence**: Evaluates dependency levels on AI tools
        
        **📚 Incorrect Information Risk**: Assesses vulnerability to misinformation
        
        **🔒 Privacy Concern Level**: Analyzes privacy awareness and data protection habits
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        1. **Complete your profile above** (one-time setup)
        2. **Start with Data Visualizations** to explore the dataset
        3. Select a risk assessment tab - forms will auto-fill!
        4. Modify any fields if needed, then predict
        5. Review your results and recommendations
        6. Take action based on the guidance provided
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Important Notes")
        st.warning("""
        - All assessments are based on machine learning models
        - Results are for guidance purposes only
        - Your data is processed locally and not stored
        - Consider consulting professionals for serious concerns
        """)
    
    
    models_exist = all([
        os.path.exists('models/critical_thinking_model.pkl'),
        os.path.exists('models/ai_overdependence_model.pkl'),
        os.path.exists('models/incorrect_info_model.pkl'),
        os.path.exists('models/privacy_concern_model.pkl')
    ])
    
    if not models_exist:
        st.error("⚠️ **Models not found!** Please train the models first.")
        st.markdown("""
        ### 🔧 Setup Instructions:
        
        1. **Run Data Cleaning**: Execute the `data_cleaning.ipynb` notebook
        2. **Train Models**: Run the training script:
        ```bash
        cd training
        python train_models.py
        ```
        3. **Restart this app**: Refresh the page after training is complete
        """)
        
        st.subheader("📁 Model Files Status")
        model_files = [
            'models/critical_thinking_model.pkl',
            'models/ai_overdependence_model.pkl', 
            'models/incorrect_info_model.pkl',
            'models/privacy_concern_model.pkl'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                st.success(f"✅ {model_file}")
            else:
                st.error(f"❌ {model_file}")
        
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Visualizations",
        "🧠 Critical Thinking",
        "⚡ AI Overdependence", 
       "📚 Incorrect Information",
        "🔒 Privacy Concern"
    ])
    
 
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        try:
            run_graphs_tab()
        except Exception as e:
            st.error(f"Error in Graphs tab: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        try:
            run_critical_thinking_tab()
        except Exception as e:
            st.error(f"Error in Critical Thinking tab: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
 
    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        try:
            run_ai_overdependence_tab()
        except Exception as e:
            st.error(f"Error in AI Overdependence tab: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
 
    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        try:
            run_incorrect_info_tab()
        except Exception as e:
            st.error(f"Error in Incorrect Information tab: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        try:
            run_privacy_tab()
        except Exception as e:
            st.error(f"Error in Privacy tab: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            🤖 AI Risk Prediction Dashboard v1.0<br>
            Built with Streamlit & Scikit-learn<br>
            <em>Use responsibly and consult professionals for serious concerns</em>
        </div>
        """, unsafe_allow_html=True)

   

if __name__ == "__main__":
    main()