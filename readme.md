# AI Risk Prediction System - README

## 🎯 Project Overview

This is a comprehensive **AI Risk Prediction System** built with **Streamlit** and **Classical Machine Learning** algorithms. The system analyzes AI usage patterns and predicts four key risk scores:

1. **Critical Thinking Risk Score** - Impact on analytical skills
2. **AI Overdependence Risk** - Dependency levels on AI tools  
3. **Incorrect Information Risk Score** - Vulnerability to misinformation
4. **Privacy Concern Level** - Privacy awareness and data protection habits

## 📁 Project Structure

```
ai/
├── app.py                    # Main Streamlit application
├── data_cleaning.ipynb       # Data preprocessing notebook
├── requirements.txt          # Python dependencies
|
├── README.md                 # This file
├── data/
│   ├── raw/ai.csv           # Original dataset (341 records)
│   └── clean/               # Preprocessed data & encoders
├── training/
│   └── train_models.py      # ML model training script
├── models/                  # Trained model files (.pkl)
└── tabs/
    ├── graphs.py            # Data visualizations tab
    ├── critical_thinking.py # Critical thinking predictions
    ├── ai_overdependence.py # Overdependence predictions
    ├── incorrect_info.py    # Misinformation risk predictions
    └── privacy.py           # Privacy concern predictions
```

## 🚀 Getting Started

### Manual Setup (Recommended)

**Step 1: Navigate to AI Directory**
```bash
cd ai
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Data Cleaning**
```bash
 data_cleaning.ipynb
```
- Run all cells to clean and preprocess the data
- This will create cleaned data in `data/clean/`

**Step 4: Train ML Models**
```bash
cd training
python train_models.py
```
- Trains 4 different ML models using classical algorithms
- Saves trained models to `models/` directory

**Step 5: Launch Streamlit App**
```bash
cd ..
streamlit run app.py
```

### Automated Setup

```bash
cd ai
./setup.sh
```

## 📊 Features

### 🎨 Data Visualizations Tab
- **Country & AI Tools Analysis**: Donut charts, sunburst charts, heatmaps
- **Demographic Analysis**: Age/gender distributions, profession analysis
- **AI Usage Patterns**: Scatter plots, time analysis, purpose breakdown
- **Risk Analysis**: Distribution plots, correlation matrices, 3D visualizations
- **Behavior Analysis**: Verification patterns, privacy behavior
- **Advanced Analytics**: Treemaps, violin plots, radar charts

### 🤖 Risk Prediction Tabs
Each prediction tab includes:
- Interactive input forms
- Real-time risk assessment
- Personalized recommendations
- Risk visualization gauges
- Actionable guidance

## 🔬 Machine Learning Models

The system uses **Classical ML** algorithms including:
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Ridge Regression**
- **Linear Regression**
- **Support Vector Regression**

Models are trained with:
- **Hyperparameter tuning** using GridSearchCV
- **Cross-validation** for robust evaluation
- **Feature scaling** and categorical encoding
- **Performance comparison** across all algorithms

## 📈 Key Visualizations

- **Donut Charts**: Country distribution, education levels
- **Sunburst Charts**: Country vs AI tool usage hierarchy
- **Heatmaps**: Tool preferences by demographics
- **3D Scatter Plots**: Multi-dimensional risk analysis
- **Treemaps**: Profession and tool combinations
- **Violin Plots**: Risk distribution by age groups
- **Radar Charts**: User self-assessment patterns
- **Correlation Matrices**: Risk factor relationships

## 🎯 Usage Guidelines

1. **Explore Data First**: Start with the "Data Visualizations" tab
2. **Honest Input**: Provide accurate information for better predictions
3. **Review Recommendations**: Follow the personalized guidance
4. **Regular Assessment**: Re-evaluate your risk levels periodically
5. **Take Action**: Implement suggested improvements

## ⚠️ Important Notes

- All processing is done locally - no data is stored externally
- Results are for guidance purposes only
- Consult professionals for serious concerns
- Models are trained on a specific dataset and may not generalize to all users

## 🛠️ Technical Details

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Python with Scikit-learn
- **Visualizations**: Plotly (interactive) + Seaborn (static)
- **Data Processing**: Pandas + NumPy
- **Model Storage**: Pickle serialization

## 📝 Dataset Information

- **Records**: 341 users
- **Features**: 22 columns including demographics, usage patterns, and risk scores
- **Target Variables**: 4 risk prediction scores
- **Data Quality**: Cleaned and preprocessed with feature engineering

## 🔄 Model Performance

Each model is evaluated using:
- **R² Score** (coefficient of determination)
- **Mean Squared Error** (MSE)
- **Mean Absolute Error** (MAE)
- **Cross-validation scores**

Best performing models are automatically selected and saved.

## 🎉 Ready to Use!

Your AI Risk Prediction System is now complete and ready for deployment. Follow the manual setup steps to get started!