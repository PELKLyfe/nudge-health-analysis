# Nudge Health Analysis

A comprehensive health data analysis platform that helps healthcare providers analyze patient data for risk stratification and clinical decision support.

## Features

- **Comorbidity Analysis**: Analyze patterns in ICD-10 codes to identify comorbidities and risk factors
- **Biomarker Analysis**: Process and visualize biomarker data to identify clinical trends
- **Combined Analysis**: Integrate ICD and biomarker data for comprehensive patient risk assessment
- **Risk Stratification**: Automated risk scoring across multiple clinical domains
- **AI-Powered Recommendations**: Generate clinical insights based on patient data
- **PDF Reports**: Create professional patient reports with comprehensive clinical recommendations
- **FHIR Integration**: Connect to FHIR servers to access patient data
- **Interactive Visualizations**: Network visualizations to understand complex relationships between conditions

## Technology Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas & NumPy**: Data processing
- **NetworkX**: Network analysis for comorbidity relationships
- **Matplotlib & Plotly**: Data visualization
- **OpenAI API**: AI-powered clinical recommendations
- **FHIR Client**: Healthcare interoperability
- **FPDF**: PDF report generation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nudge-health-analysis.git
cd nudge-health-analysis

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run Nudgehealthanalysis.py
```

## Usage

1. Upload patient data in Excel format with the required columns (Patient ID, Gender, Age, Zip Code, and diagnosis codes)
2. Upload biomarker data (optional)
3. View the analysis results and AI-generated clinical recommendations
4. Generate a PDF report for the patient

## Data Format

The application expects Excel files with specific column structures:
- **ICD Data**: Must include columns for Patient ID, Gender, Age, Zip Code, and at least one diagnosis column
- **Biomarker Data**: Must include columns for Patient ID, Biomarker, and Value

## License

[Specify your license here]

## Contact

[Your contact information] 