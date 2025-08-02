"""
Graphs and Data Visualization Tab

This module creates comprehensive visualizations for the AI risk prediction dataset.
Includes interactive charts using Plotly and static charts using Seaborn.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_data():
    """Load the raw dataset for visualization"""
    try:
       
        if os.path.exists('data/clean/ai_cleaned.csv'):
            df_clean = pd.read_csv('data/clean/ai_cleaned.csv')
            df_raw = pd.read_csv('data/raw/ai.csv')
            return df_raw, df_clean
        else:
           
            df_raw = pd.read_csv('data/raw/ai.csv')
            return df_raw, None
    except FileNotFoundError:
        st.error("❌ Dataset not found. Please ensure the data files are in the correct location.")
        return None, None

def create_country_ai_tool_analysis(df):
    """Create visualizations for countries and AI tools usage"""
    st.header("🌍 Countries & AI Tools Analysis")
    
    
    country_counts = df['country'].value_counts()
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=country_counts.index,
        values=country_counts.values,
        hole=.4,
        textinfo='label+percent+value',
        textposition='outside',
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig_donut.update_layout(
        title="User Distribution by Country",
        height=500,
        showlegend=True,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_donut, use_container_width=True)


def create_country_specific_charts(df, selected_country):
    """Create country-specific visualizations in full width"""
    if selected_country == "All Countries":
        country_data = df
        st.subheader(f"📊 Analysis for All Countries")
    else:
        country_data = df[df['country'] == selected_country]
        st.subheader(f"📊 Analysis for {selected_country}")
        
    if len(country_data) == 0:
        st.warning(f"No data available for {selected_country}")
        return
    
  
    st.markdown("#### 📈 AI Usage Frequency by Profession")
    prof_freq = country_data.groupby(['profession', 'frequency_of_use_weekly']).size().reset_index(name='count')
    
    fig_prof_freq = px.bar(
        prof_freq,
        x='profession',
        y='count',
        color='frequency_of_use_weekly',
        title=f"AI Usage Frequency by Profession",
        barmode='stack',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_prof_freq.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_prof_freq, use_container_width=True)
    
   
    st.markdown("#### ⏰ Hours Spent on AI per Day by Age Group")
    # Calculate average hours per age group
    avg_hours_by_age = country_data.groupby('age_group')['hours_spent_per_day_on_ai'].mean().reset_index()
    
    fig_hours_age = px.bar(
        avg_hours_by_age,
        x='age_group',
        y='hours_spent_per_day_on_ai',
        title=f"Average Daily AI Usage Hours by Age Group",
        color='age_group',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_hours_age.update_layout(height=400)
    st.plotly_chart(fig_hours_age, use_container_width=True)
    
    # Stacked Bar: Inaccuracy by AI vs Frequency of Use
    st.markdown("#### 📊 AI Inaccuracy Risk vs Usage Frequency")
    # Create risk categories based on incorrect_info_risk_score
    country_data_copy = country_data.copy()
    country_data_copy['risk_category'] = pd.cut(
        country_data_copy['incorrect_info_risk_score'], 
        bins=[0, 3, 6, 10], 
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    risk_freq = country_data_copy.groupby(['frequency_of_use_weekly', 'risk_category']).size().reset_index(name='count')
    
    fig_risk_freq = px.bar(
        risk_freq,
        x='frequency_of_use_weekly',
        y='count',
        color='risk_category',
        title=f"Information Inaccuracy Risk by Usage Frequency",
        barmode='stack',
        color_discrete_sequence=['green', 'orange', 'red']
    )
    fig_risk_freq.update_layout(height=400)
    st.plotly_chart(fig_risk_freq, use_container_width=True)
    
    # Donut of purpose of AI use in selected country
    st.markdown("#### 🎯 Purpose of AI Use")
    purpose_counts = country_data['purpose_of_ai_use'].value_counts()
    
    fig_purpose_donut = go.Figure(data=[go.Pie(
        labels=purpose_counts.index,
        values=purpose_counts.values,
        hole=.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig_purpose_donut.update_layout(
        title=f"Purpose of AI Use Distribution",
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig_purpose_donut, use_container_width=True)

    # Purpose -> AI tools mainly use (sunburst)
    st.markdown("#### 🌅 Purpose → AI Tools Usage (Sunburst)")
    purpose_tool = country_data.groupby(['purpose_of_ai_use', 'ai_tool_primary']).size().reset_index(name='count')
    
    fig_sunburst = px.sunburst(
    purpose_tool,
    path=['purpose_of_ai_use', 'ai_tool_primary'],
    values='count',
    color='purpose_of_ai_use',
    color_discrete_sequence=px.colors.qualitative.Alphabet
)
    
    fig_sunburst.update_layout(
    title=f"Purpose → AI Tools Usage Pattern",
    height=500,
    margin=dict(t=50, l=0, r=0, b=0)
)
    
    st.plotly_chart(fig_sunburst, use_container_width=True)

def create_risk_correlation_matrix(df):
    """Create risk correlation matrix"""
    st.header("🔗 Risk Correlation Matrix")
    
    risk_cols = ['critical_thinking_risk_score', 'ai_overdependence_risk', 'incorrect_info_risk_score', 'privacy_concern_level']
    risk_corr = df[risk_cols].corr()
    
    fig_corr = px.imshow(
        risk_corr,
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu",
        aspect="auto",
        text_auto=True
    )
    
    fig_corr.update_layout(
        title="Risk Scores Correlation Matrix",
        height=500,
        width=600
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

def create_risk_distributions(df):
    """Create risk score distributions"""
    st.header("📊 Risk Score Distributions")
    
    risk_cols = ['critical_thinking_risk_score', 'ai_overdependence_risk', 'incorrect_info_risk_score', 'privacy_concern_level']
    risk_names = ['Critical Thinking Risk', 'AI Overdependence Risk', 'Incorrect Info Risk', 'Privacy Concern Level']
    
    # Create subplots for all risk scores
    fig_risks = make_subplots(
        rows=2, cols=2,
        subplot_titles=risk_names,
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    colors = ['red', 'orange', 'blue', 'green']
    
    for i, (col, color, name) in enumerate(zip(risk_cols, colors, risk_names)):
        row = (i // 2) + 1
        column = (i % 2) + 1
        
        fig_risks.add_trace(
            go.Histogram(x=df[col], name=name, marker_color=color, nbinsx=20, showlegend=False),
            row=row, col=column
        )
    
    fig_risks.update_layout(
        height=600,
        title_text="Distribution of All Risk Scores",
        showlegend=False
    )
    
    st.plotly_chart(fig_risks, use_container_width=True)

def run_graphs_tab():
    """Main function to run the graphs and visualization tab"""
    
    # Load data
    df_raw, df_clean = load_data()
    
    if df_raw is None:
        st.error("❌ Unable to load data. Please check if the dataset files exist.")
        return
   
    st.markdown("## 📈 AI Risk Prediction - Data Visualizations")
    st.markdown("---")
    
   
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(df_raw))
    with col2:
        st.metric("Countries", df_raw['country'].nunique())
    with col3:
        st.metric("AI Tools", df_raw['ai_tool_primary'].nunique())
    with col4:
        st.metric("Professions", df_raw['profession'].nunique())
    
    st.markdown("---")
    
 
    st.header("🌍 User Distribution by Countries")
    country_counts = df_raw['country'].value_counts()
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=country_counts.index,
        values=country_counts.values,
        hole=.4,
        textinfo='label+percent+value',
        textposition='outside',
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig_donut.update_layout(
        title="User Distribution by Country",
        height=500,
        showlegend=True,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown("---")
    
   
    st.header("🌎 Country-Specific Analysis")
    
    # Country dropdown
    countries = ["All Countries"] + sorted(df_raw['country'].unique().tolist())
    selected_country = st.selectbox("Select a country to analyze:", countries)
    
   
    create_country_specific_charts(df_raw, selected_country)
    st.markdown("---")
    
    create_risk_correlation_matrix(df_raw)
    st.markdown("---")
    
    # 4. Risk Score Distributions
    create_risk_distributions(df_raw)

if __name__ == "__main__":
    run_graphs_tab()