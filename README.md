# Nudge Health Analysis

A comprehensive clinical data analysis platform that integrates network analysis, risk scoring, and AI-powered recommendations.

## Features

- Integrated network analysis and risk score calculations
- Clinical domain analysis and visualization
- Patient-level risk assessment
- Population health insights
- AI-powered clinical recommendations
- PDF report generation

## Deployment

This app is deployed on Streamlit Cloud. To run locally:

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run Nudgehealthanalysis_new.py
   ```

## Data Requirements

The app accepts two types of input files:
1. ICD-10 diagnosis data (Excel format)
2. Biomarker data (Excel format)

Example files are provided in the repository.

## Environment Variables

For AI recommendations, set your OpenAI API key in the Streamlit secrets or as an environment variable:
```
OPENAI_API_KEY=your_api_key_here
```

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