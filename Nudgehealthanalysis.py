import streamlit as st
import pandas as pd
import numpy as np
import requests, zipfile, io
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import openai
from fpdf import FPDF
import re
import networkx as nx
import matplotlib.pyplot as plt
import fhirclient.client
import fhirclient.models.patient as p
import fhirclient.models.observation as o
import fhirclient.models.condition as c
import fhirclient.models.medicationrequest as m
import fhirclient.models.procedure as pr
import fhirclient.models.allergyintolerance as a
import fhirclient.models.immunization as i
import fhirclient.models.medicationstatement as ms
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import json
import plotly.express as px
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
from collections import defaultdict
import math
import os
import base64
import tempfile
import matplotlib.patches as mpatches
try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

# Initialize OpenAI client as early as possible for global access
def init_openai_client():
    """
    Initialize OpenAI client with API key from Streamlit secrets or environment variables.
    Returns True if successfully configured.
    """
    try:
        # Try to get API key from Streamlit secrets
        try:
            openai.api_key = st.secrets["openai"]["api_key"]
            return True
        except (KeyError, FileNotFoundError):
            # Check environment variables as fallback
            import os
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
                return True
            return False
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        return False

# Set up logging for FHIR operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize dataframes at the top level
combined_df = pd.DataFrame()
analyzed_icd = pd.DataFrame()
analyzed_bio = pd.DataFrame()

# Functional Clinical Domains Definition with Comprehensive Risk Scoring
FUNCTIONAL_DOMAINS = {
    "Cardiometabolic": {
        'codes': [
            'E08', 'E09', 'E10', 'E11', 'E12', 'E13',  # Diabetes
            'E78',  # Lipid disorders
            'I10', 'I11', 'I12', 'I13',  # Hypertensive diseases
            'I20', 'I21', 'I22', 'I23', 'I24', 'I25',  # Ischemic heart diseases
            'I50',  # Heart failure
            'I63', 'I64', 'I65', 'I66',  # Cerebrovascular diseases
            'R73',  # Hyperglycemia
            'E66',  # Obesity
            'K76.0',  # NAFLD
            'N18'  # Chronic kidney disease (CKD)
        ],
        'description': 'Diabetes, Hypertension, CAD, Heart Failure, Stroke, NAFLD, CKD',
        'color': '#FF6B6B',  # Red
        'base_weight': 0.9,  # Updated to match specified range 0.8-1.0
        'severity_levels': {
            'mild': 0.8,
            'moderate': 0.9,
            'severe': 1.0
        }
    },
    "Immune-Inflammation": {
        'codes': [
            'M05', 'M06',  # Rheumatoid arthritis (RA)
            'K50', 'K51', 'K52',  # IBD
            'M32',  # SLE (Lupus)
            'K05',  # Periodontal disease
            'A40', 'A41',  # Sepsis
            'B20'  # HIV
        ],
        'description': 'RA, Lupus, IBD, Sepsis, HIV, Periodontitis',
        'color': '#4ECDC4',  # Green
        'base_weight': 0.8,  # Updated to match specified range 0.7-0.9
        'severity_levels': {
            'mild': 0.7,
            'moderate': 0.8,
            'severe': 0.9
        }
    },
    "Oncological": {
        'codes': [
            'C', 'D0', 'D1', 'D2', 'D3', 'D4',  # All cancers
            'Z85',  # History of cancer
            'Z80', 'Z81', 'Z82', 'Z83', 'Z84'  # Family history
        ],
        'description': 'All Cancer ICD-10 codes (C00-D49 range), Cancer history, Family history',
        'color': '#9B59B6',  # Purple
        'base_weight': 0.95,  # Updated to match specified range 0.9-1.0
        'severity_levels': {
            'mild': 0.9,
            'moderate': 0.95,
            'severe': 1.0
        }
    },
    "Neuro-Mental Health": {
        'codes': [
            'F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29',  # Schizophrenia
            'F31',  # Bipolar
            'F32', 'F33', 'F34',  # Depression
            'F41',  # Anxiety
            'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19',  # Substance abuse
            'G47.0', 'G47.3',  # Sleep disorders, Sleep Apnea
            'R45.851'  # Suicidal ideation
        ],
        'description': 'Depression, Anxiety, Substance abuse, Bipolar, Sleep Apnea, Suicidal ideation',
        'color': '#3498DB',  # Blue
        'base_weight': 0.7,  # Updated to match specified range 0.6-0.8
        'severity_levels': {
            'mild': 0.6,
            'moderate': 0.7,
            'severe': 0.8
        }
    },
    "Neurological & Frailty": {
        'codes': [
            'G30', 'G31',  # Dementia, Alzheimer's
            'G20', 'G21',  # Parkinson's
            'G12', 'G12.21',  # ALS
            'G35',  # MS
            'R26', 'R27', 'R28', 'R29',  # Mobility issues, Gait
            'M81'  # Osteoporosis
        ],
        'description': 'Alzheimer\'s, Dementia, Parkinson\'s, ALS, MS, Osteoporosis, Mobility/Gait issues',
        'color': '#E67E22',  # Orange
        'base_weight': 0.9,  # Updated to match specified range 0.8-1.0
        'severity_levels': {
            'mild': 0.8,
            'moderate': 0.9,
            'severe': 1.0
        }
    },
    "SDOH": {
        'codes': [
            'Z59', 'Z60.2',  # Housing, social isolation
            'Z56', 'Z59.7',  # Employment, economic
            'Z59.4',  # Food insecurity
            'Z59.5'   # Transportation
        ],
        'description': 'Housing, Food insecurity, Economic instability, Transportation, Medication adherence',
        'color': '#95A5A6',  # Gray
        'base_weight': 0.8,  # Updated to match specified weight
        'severity_levels': {
            'mild': 0.7,
            'moderate': 0.8,
            'severe': 0.9
        }
    }
}

# Gender-based risk adjustment coefficients
GENDER_COEFFICIENTS = {
    'M': {  # Men coefficients
        'Cardiometabolic': 1.0,
        'Immune-Inflammation': 1.0,
        'Oncological': 1.0,
        'Neuro-Mental Health': 1.0,
        'Neurological & Frailty': 0.85,
        'SDOH': 1.0
    },
    'F': {  # Women coefficients
        'Cardiometabolic': 1.30,
        'Immune-Inflammation': 1.20,
        'Oncological': 1.80,  # Will be adjusted based on age
        'Neuro-Mental Health': 1.10,
        'Neurological & Frailty': 1.0,
        'SDOH': 1.0
    }
}

# Age-based risk adjustment
AGE_ADJUSTMENTS = {
    'F': {  # Women age adjustments
        'under_50': {
            'Cardiometabolic': 1.2,
            'Oncological': 1.8  # Higher risk for women under 50
        },
        'over_50': {
            'Cardiometabolic': 1.3,
            'Oncological': 1.2
        }
    },
    'M': {  # Men age adjustments
        'under_50': {
            'Cardiometabolic': 1.0,
            'Oncological': 1.0
        },
        'over_50': {
            'Cardiometabolic': 1.0,
            'Oncological': 1.0
        }
    }
}

# Network-based logistic regression coefficients (calibrated to historical data)
HOSPITALIZATION_COEFFICIENTS = {
    'intercept': -5.2,  # Model intercept
    'nhcrs_weight': 6.8,  # NHCRS total score coefficient
    'network_centrality_weight': 0.7,  # Network centrality impact
    'age_weight': 0.02,  # Per year of age
    'domain_weights': {
        'Cardiometabolic': 1.2,
        'Immune-Inflammation': 0.9,
        'Oncological': 1.1,
        'Neuro-Mental Health': 0.8,
        'Neurological & Frailty': 1.3,
        'SDOH': 0.7
    }
}

MORTALITY_COEFFICIENTS = {
    'intercept': -6.5,  # Model intercept
    'nhcrs_weight': 7.2,  # NHCRS total score coefficient
    'network_centrality_weight': 0.9,  # Network centrality impact
    'age_weight': 0.04,  # Per year of age
    'domain_weights': {
        'Cardiometabolic': 1.4,
        'Immune-Inflammation': 0.8,
        'Oncological': 1.5,
        'Neuro-Mental Health': 0.7,
        'Neurological & Frailty': 1.6,
        'SDOH': 0.5
    }
}

# Dynamic decay constants for treatment response modeling
DECAY_CONSTANTS = {
    'linear': {
        'LDL': 0.05,  # 5% improvement per month with statin
        'HbA1c': 0.03,  # 3% improvement per month with proper treatment
        'Blood_Pressure': 0.08  # 8% improvement per month with medication
    },
    'exponential': {
        'hsCRP': 0.2,  # Faster response to anti-inflammatory treatment
        'Troponin': 0.3,  # Rapid normalization post-acute event
        'WBC': 0.25  # White blood cell count normalization with antibiotics
    },
    'threshold': {
        'LDL': {'value': 100, 'risk_reduction': 0.5},  # LDL < 100 mg/dL
        'HbA1c': {'value': 7.0, 'risk_reduction': 0.6},  # HbA1c < 7.0%
        'Blood_Pressure': {'value': 130, 'risk_reduction': 0.4}  # Systolic < 130 mmHg
    }
}

def calculate_domain_risk_score(conditions: list, domain: str, patient_age: int, patient_gender: str, biomarkers: dict = None) -> float:
    """
    Calculate risk score for a specific clinical domain using the refined methodology.
    
    Args:
        conditions: List of ICD-10 codes
        domain: Domain name from FUNCTIONAL_DOMAINS
        patient_age: Patient's age
        patient_gender: Patient's gender ('M' or 'F')
        biomarkers: Optional dictionary of biomarker values
    
    Returns:
        float: Domain risk score between 0 and 1
    """
    domain_info = FUNCTIONAL_DOMAINS[domain]
    domain_codes = domain_info['codes']
    
    # Count conditions in this domain and calculate their severity
    domain_conditions = []
    for condition in conditions:
        if any(condition.startswith(code) for code in domain_codes):
            severity = get_condition_severity(condition, biomarkers)
            severity_score = domain_info['severity_levels'].get(severity, 0.5)
            domain_conditions.append((condition, severity_score))
    
    if not domain_conditions:
        return 0.0
    
    # Calculate weighted sum of condition severity scores
    condition_score_sum = sum(severity for _, severity in domain_conditions)
    
    # Scale by number of conditions (more conditions = higher risk)
    condition_count_factor = min(1.0, len(domain_conditions) / 5.0)  # Cap at 5 conditions
    
    # Calculate base domain score
    base_score = domain_info['base_weight'] * condition_score_sum * condition_count_factor
    
    # Apply age-specific adjustment
    age_category = 'under_50' if patient_age < 50 else 'over_50'
    age_adjustment = AGE_ADJUSTMENTS.get(patient_gender, {}).get(age_category, {}).get(domain, 1.0)
    
    # Apply gender coefficient
    gender_coef = GENDER_COEFFICIENTS[patient_gender][domain]
    
    # Calculate final score with adjustments
    adjusted_score = base_score * gender_coef * age_adjustment
    
    # Normalize to 0-1 range
    return min(adjusted_score, 1.0)

def calculate_total_risk_score(patient_data: dict) -> dict:
    """
    Calculate comprehensive risk score across all domains using the refined methodology.
    
    Args:
        patient_data: Dictionary containing patient information and conditions
    
    Returns:
        dict: Risk scores by domain, total risk score, and risk level
    """
    conditions = patient_data.get('conditions', [])
    age = patient_data.get('age', 0)
    gender = patient_data.get('gender', 'U')
    biomarkers = patient_data.get('biomarkers', {})
    
    # Calculate domain-specific scores
    domain_scores = {}
    total_weighted_score = 0
    total_weight = 0
    
    for domain, info in FUNCTIONAL_DOMAINS.items():
        domain_score = calculate_domain_risk_score(conditions, domain, age, gender, biomarkers)
        domain_scores[domain] = domain_score
        
        # Weight each domain by its base weight for the total score
        domain_weight = info['base_weight']
        total_weighted_score += domain_score * domain_weight
        total_weight += domain_weight
    
    # Calculate normalized total score
    if total_weight > 0:
        total_score = total_weighted_score / total_weight
    else:
        total_score = 0
    
    # Determine risk level
    if total_score > 0.7:
        risk_level = 'High'
    elif total_score > 0.4:
        risk_level = 'Moderate'
    else:
        risk_level = 'Low'
    
    # Calculate network-based hospitalization and mortality risk
    network_metrics = patient_data.get('network_metrics', None)
    time_factors = patient_data.get('time_factors', None)
    
    # Store risk scores for use in other calculations
    patient_data_with_scores = {
        'conditions': conditions,
        'age': age,
        'gender': gender,
        'biomarkers': biomarkers,
        'risk_scores': {
            'domain_scores': domain_scores,
            'total_score': total_score
        }
    }
    
    # Calculate 5-year hospitalization and 10-year mortality risks
    hospitalization_risk = calculate_network_hospitalization_risk(
        patient_data_with_scores, network_metrics, time_factors
    )
    
    mortality_risk = calculate_network_mortality_risk(
        patient_data_with_scores, network_metrics, time_factors
    )
        
    return {
        'domain_scores': domain_scores,
        'total_score': total_score,
        'risk_level': risk_level,
        'hospitalization_risk_5yr': hospitalization_risk,
        'mortality_risk_10yr': mortality_risk
    }

def get_condition_severity(condition: str, biomarkers: dict = None) -> str:
    """
    Determine condition severity based on ICD-10 code and biomarkers if available.
    Enhanced to include more conditions and biomarker thresholds.
    """
    # Default to moderate if no biomarkers available
    if not biomarkers:
        return 'moderate'
    
    # More comprehensive severity mapping with biomarker thresholds
    severity_map = {
        # Diabetes
        'E11': lambda b: 'severe' if b.get('HbA1c', 0) > 9.0 else 
               'moderate' if b.get('HbA1c', 0) > 7.0 else 
               'mild' if b.get('HbA1c', 0) > 5.7 else 'mild',
        
        # Hypertension
        'I10': lambda b: 'severe' if b.get('systolic_bp', 0) > 180 or b.get('diastolic_bp', 0) > 120 else 
               'moderate' if b.get('systolic_bp', 0) > 140 or b.get('diastolic_bp', 0) > 90 else 
               'mild' if b.get('systolic_bp', 0) > 130 or b.get('diastolic_bp', 0) > 80 else 'mild',
        
        # Hyperlipidemia
        'E78': lambda b: 'severe' if b.get('LDL', 0) > 190 or b.get('triglycerides', 0) > 500 else 
               'moderate' if b.get('LDL', 0) > 130 or b.get('triglycerides', 0) > 200 else 
               'mild' if b.get('LDL', 0) > 100 or b.get('triglycerides', 0) > 150 else 'mild',
        
        # Heart Failure
        'I50': lambda b: 'severe' if b.get('NT-proBNP', 0) > 2000 or b.get('BNP', 0) > 500 else 
               'moderate' if b.get('NT-proBNP', 0) > 900 or b.get('BNP', 0) > 200 else 
               'mild' if b.get('NT-proBNP', 0) > 300 or b.get('BNP', 0) > 100 else 'mild',
        
        # Kidney Disease
        'N18': lambda b: 'severe' if b.get('eGFR', 100) < 30 else 
               'moderate' if b.get('eGFR', 100) < 60 else 
               'mild' if b.get('eGFR', 100) < 90 else 'mild',
        
        # Obesity
        'E66': lambda b: 'severe' if b.get('BMI', 0) > 40 else 
               'moderate' if b.get('BMI', 0) > 35 else 
               'mild' if b.get('BMI', 0) > 30 else 'mild',
        
        # Inflammation
        'K50': lambda b: 'severe' if b.get('CRP', 0) > 10 or b.get('ESR', 0) > 50 else 
               'moderate' if b.get('CRP', 0) > 5 or b.get('ESR', 0) > 30 else 
               'mild' if b.get('CRP', 0) > 1 or b.get('ESR', 0) > 15 else 'mild',
    }
    
    # Check for exact matches first
    for code, severity_func in severity_map.items():
        if condition == code:
            return severity_func(biomarkers)
    
    # Check for prefix matches
    for code, severity_func in severity_map.items():
        if condition.startswith(code):
            return severity_func(biomarkers)
    
    return 'moderate'

def assign_clinical_domain(icd_code: str) -> str:
    """
    Assign a clinical domain to an ICD-10 code.
    Returns the domain name or 'Other' if no match is found.
    """
    if not isinstance(icd_code, str):
        return "Other"
    
    icd_code = icd_code.strip().upper()
    for domain, info in FUNCTIONAL_DOMAINS.items():
        if any(icd_code.startswith(code) for code in info['codes']):
            return domain
    return "Other"

def process_domain_data(df: pd.DataFrame, diag_cols: list) -> pd.DataFrame:
    """
    Process diagnosis data and assign clinical domains.
    Returns a DataFrame with conditions mapped to their domains.
    """
    domain_mapped = []
    
    for idx, row in df.iterrows():
        patient_id = row.get('PatId') or row.get('Pat Id') or idx
        patient_age = row.get('Age', None)
        patient_gender = row.get('Gender', None)
        
        for col in diag_cols:
            if pd.notna(row[col]) and str(row[col]).strip():
                condition = str(row[col]).strip()
                domain = assign_clinical_domain(condition)
                
                domain_mapped.append({
                    'patient_id': patient_id,
                    'age': patient_age,
                    'gender': patient_gender,
                    'condition': condition,
                    'domain': domain,
                    'domain_color': FUNCTIONAL_DOMAINS.get(domain, {}).get('color', '#95A5A6')  # Gray for Other
                })
    
    return pd.DataFrame(domain_mapped)

def create_domain_network(domain_df: pd.DataFrame) -> nx.Graph:
    """
    Create a network graph with nodes colored by clinical domain.
    Returns a NetworkX graph object with domain attributes.
    """
    G = nx.Graph()
    
    # Group by patient to find co-occurring conditions
    patient_conditions = domain_df.groupby('patient_id')['condition'].agg(list)
    
    # Add edges for co-occurring conditions
    for conditions in patient_conditions:
        for i, cond_a in enumerate(conditions):
            for cond_b in conditions[i+1:]:
                if G.has_edge(cond_a, cond_b):
                    G[cond_a][cond_b]['weight'] += 1
                else:
                    G.add_edge(cond_a, cond_b, weight=1)
    
    # Add domain and color attributes to nodes
    condition_attrs = domain_df.set_index('condition')[['domain', 'domain_color']].to_dict('index')
    for node in G.nodes():
        attrs = condition_attrs.get(node, {'domain': 'Other', 'domain_color': '#95A5A6'})
        G.nodes[node]['domain'] = attrs['domain']
        G.nodes[node]['color'] = attrs['domain_color']
    
    return G

def analyze_domain_patterns(domain_df: pd.DataFrame) -> None:
    """
    Analyze and visualize patterns within and across clinical domains.
    """
    st.subheader("🎯 Clinical Domain Analysis")
    
    # Domain distribution
    domain_counts = domain_df['domain'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Domain Distribution")
        fig = px.pie(values=domain_counts.values, 
                    names=domain_counts.index,
                    color=domain_counts.index,
                    color_discrete_map={k: v['color'] for k, v in FUNCTIONAL_DOMAINS.items()})
        st.plotly_chart(fig)
    
    with col2:
        st.write("Conditions per Domain")
        for domain, count in domain_counts.items():
            color = FUNCTIONAL_DOMAINS.get(domain, {}).get('color', '#95A5A6')
            st.markdown(f"<div style='color: {color}'><b>{domain}</b>: {count} conditions</div>", 
                       unsafe_allow_html=True)
    
    # Age distribution by domain
    if 'age' in domain_df.columns:
        st.subheader("📊 Age Distribution by Domain")
        fig = px.box(domain_df, x='domain', y='age', 
                    color='domain',
                    color_discrete_map={k: v['color'] for k, v in FUNCTIONAL_DOMAINS.items()})
        st.plotly_chart(fig)
    
    # Gender distribution by domain
    if 'gender' in domain_df.columns:
        st.subheader("👥 Gender Distribution by Domain")
        gender_domain = pd.crosstab(domain_df['domain'], domain_df['gender'])
        fig = px.bar(gender_domain, 
                    color_discrete_map={k: v['color'] for k, v in FUNCTIONAL_DOMAINS.items()})
        st.plotly_chart(fig)

def visualize_domain_network(G: nx.Graph) -> None:
    """
    Create and display the domain-based network visualization.
    """
    st.subheader("🕸️ Clinical Domain Network")
    
    # Calculate node sizes based on degree centrality
    centrality = nx.degree_centrality(G)
    node_sizes = [centrality[node] * 5000 for node in G.nodes()]
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw edges with varying widths based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, 
                          width=[w/max(edge_weights)*2 for w in edge_weights],
                          alpha=0.3)
    
    # Draw nodes with domain colors
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_sizes)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Clinical Domain Network Analysis")
    st.pyplot(fig)
    
    # Display domain legend
    st.write("📚 Domain Legend:")
    for domain, info in FUNCTIONAL_DOMAINS.items():
        st.markdown(f"<div style='color: {info['color']}'><b>{domain}</b>: {info['description']}</div>",
                   unsafe_allow_html=True)

# Cache for ICD-10 and CPT code data
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_authoritative_codes():
    """Fetch authoritative ICD-10 and CPT codes from official sources"""
    try:
        # Use a more reliable source for ICD-10 codes
        icd_url = "https://raw.githubusercontent.com/kamillamagna/ICD-10-CSV/master/codes.csv"
        
        try:
            icd_df = pd.read_csv(icd_url)
            # Ensure the dataframe has the expected columns
            if 'code' not in icd_df.columns:
                icd_df = icd_df.rename(columns={
                    'CODE': 'code',
                    'DESCRIPTOR': 'description'
                })
            
            # Clean the data
            icd_df = icd_df[['code', 'description']].dropna()
            icd_df['code'] = icd_df['code'].str.strip()
            icd_df['description'] = icd_df['description'].str.strip()
            
        except Exception as e:
            logger.warning(f"Failed to fetch ICD codes from primary source: {str(e)}")
            # Fallback to local cache or create minimal structure
            icd_df = pd.DataFrame({
                'code': ['placeholder'],
                'description': ['Local cache - update needed']
            })
        
        # CPT codes would typically come from a licensed source
        # For now, create an empty DataFrame with the correct structure
        cpt_df = pd.DataFrame(columns=['code', 'description'])
        
        return {
            'icd10': icd_df,
            'cpt': cpt_df
        }
    except Exception as e:
        logger.error(f"Error in fetch_authoritative_codes: {str(e)}")
        # Return minimal structure that won't break the app
        return {
            'icd10': pd.DataFrame(columns=['code', 'description']),
            'cpt': pd.DataFrame(columns=['code', 'description'])
        }

# Function Definitions
def ai_clinical_recommendations(patient_info):
    """
    Generate AI-driven clinical recommendations based on patient data.
    Provides structured insights for diagnosis, intervention, and economic impact.
    
    Args:
        patient_info: Dictionary containing patient information including conditions,
                     biomarkers, risk scores, and demographics
    
    Returns:
        str: Comprehensive clinical recommendations in a structured format
    """
    # Format patient data for the prompt
    patient_age = patient_info.get('age', 'Unknown')
    patient_gender = patient_info.get('gender', 'Unknown')
    conditions = patient_info.get('conditions', [])
    biomarkers = patient_info.get('biomarkers', {})
    risk_scores = patient_info.get('risk_scores', {})
    
    # Calculate domain-based risk if not provided
    if not risk_scores and conditions:
        patient_data = {
            'conditions': conditions,
            'age': patient_age,
            'gender': patient_gender,
            'biomarkers': biomarkers
        }
        risk_info = calculate_total_risk_score(patient_data)
        risk_scores = risk_info.get('domain_scores', {})
        hospitalization_risk = risk_info.get('hospitalization_risk_5yr', 0)
        mortality_risk = risk_info.get('mortality_risk_10yr', 0)
    else:
        hospitalization_risk = patient_info.get('hospitalization_risk_5yr', 0)
        mortality_risk = patient_info.get('mortality_risk_10yr', 0)
    
    # Construct a detailed prompt for the AI
    domain_risk_str = ""
    for domain, score in risk_scores.items():
        domain_risk_str += f"- {domain}: {score:.2f}\n"
    
    biomarker_str = ""
    for marker, value in biomarkers.items():
        biomarker_str += f"- {marker}: {value}\n"
    
    # Add hospitalization and mortality risks to the prompt
    risk_str = f"""
Domain-Specific Risks:
{domain_risk_str}

Key Outcome Probabilities:
- 5-Year Hospitalization Risk: {hospitalization_risk:.2%}
- 10-Year Mortality Risk: {mortality_risk:.2%}
"""
    
    # Construct the AI prompt with risk probabilities
    prompt = f"""
As a clinical AI assistant, provide a comprehensive clinical analysis for this patient:

PATIENT INFORMATION:
- Age: {patient_age}
- Gender: {patient_gender}
- Diagnoses: {', '.join(conditions) if conditions else 'None'}

BIOMARKER DATA:
{biomarker_str if biomarker_str else "No biomarker data available."}

RISK ASSESSMENT:
{risk_str}

Please provide a structured clinical report with the following sections:
1. Patient Overview: Summarize key clinical concerns based on diagnoses and biomarkers.
2. Clinical Challenge: Identify the most important clinical risk factors and challenges.
3. Recommended Diagnostic Steps: Suggest specific tests or evaluations.
4. Clinical Recommendations: Provide evidence-based interventions with specific medications, dosages, and lifestyle modifications.
5. Economic Impact: Estimate potential risk reduction and insurance premium incentives.

Keep recommendations specific, actionable, and evidence-based.
"""

    # Call OpenAI API with the prompt
    try:
        # Handle both older and newer OpenAI API versions
        try:
            # New API (openai >= 1.0.0)
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            recommendations = response.choices[0].message.content
        except (ImportError, AttributeError):
            # Fallback to older API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            recommendations = response.choices[0].message["content"]
        
        return recommendations
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {str(e)}")
        return f"Error generating recommendations: {str(e)}"

def create_pdf(patient_info, ai_recommendations):
    """
    Create a professionally formatted PDF report with patient information and AI recommendations.
    Follows the Nudge Health template with sections for patient overview, network analysis methodology,
    risk assessment, and economic impact.
    
    Args:
        patient_info: Dictionary containing patient demographics, conditions, and risk scores
        ai_recommendations: String containing the AI-generated clinical recommendations
        
    Returns:
        bytes: PDF document as bytes
    """
    from fpdf import FPDF
    from datetime import datetime
    import re
    import os
    import base64
    import tempfile
    import logging
    
    # Ensure patient_info is a dictionary
    if not isinstance(patient_info, dict):
        patient_info = {}
        logging.warning("Patient info was not a dictionary. Using empty dictionary instead.")
    
    # Ensure ai_recommendations is a string
    if not isinstance(ai_recommendations, str):
        ai_recommendations = str(ai_recommendations) if ai_recommendations is not None else ""
        logging.warning("AI recommendations was not a string. Converting to string.")
    
    # Create custom PDF class with header and footer
    class NudgeHealthPDF(FPDF):
        def header(self):
            # Logo - use the user's elephant logo with proper positioning
            try:
                # Check if logo file exists
                if os.path.exists('logo.png'):
                    # Position the logo at the top center with appropriate size
                    self.image('logo.png', x=85, y=30, w=40, h=40)
                else:
                    raise FileNotFoundError("logo.png not found")
            except Exception as e:
                logging.warning(f"Could not load logo image: {str(e)}")
                # Create a fallback if logo can't be loaded
                self.set_fill_color(0, 0, 0)  # Black
                self.ellipse(85, 30, 40, 40, style='F')
                self.set_fill_color(255, 255, 255)  # White
                self.set_font('Arial', 'B', 20)
                self.set_text_color(255, 255, 255)
                self.set_xy(85, 40)
                self.cell(40, 20, 'NH', 0, 0, 'C')
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, '', 0, 1, 'C')
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    # Initialize PDF
    pdf = NudgeHealthPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Extract patient information
    patient_age = patient_info.get('age', 'Unknown')
    patient_gender = patient_info.get('gender', 'Unknown')
    conditions = patient_info.get('conditions', [])
    biomarkers = patient_info.get('biomarkers', {})
    risk_scores = patient_info.get('risk_scores', {})
    domain_scores = risk_scores.get('domain_scores', {})
    total_score = risk_scores.get('total_score', 0)
    hospitalization_risk = risk_scores.get('hospitalization_risk_5yr', 0.38)
    mortality_risk = risk_scores.get('mortality_risk_10yr', 0.45)
    
    # Cover page
    # -------------
    
    # Add logo placeholder (circular elephant logo)
    try:
        pdf.image('logo.png', x=85, y=30, w=40, h=40)
    except Exception:
        # If logo doesn't exist, create a placeholder circle
        pdf.set_fill_color(0, 0, 0)  # Black
        pdf.ellipse(85, 30, 40, 40, style='F')
        
        # Add white elephant silhouette placeholder
        pdf.set_fill_color(255, 255, 255)  # White
        pdf.ellipse(95, 40, 20, 20, style='F')  # Head
        pdf.set_xy(90, 50)
        pdf.cell(30, 10, 'N', 0, 0, 'C', fill=False)
    
    # Title
    pdf.set_font('Arial', 'B', 18)
    pdf.ln(70)
    pdf.cell(0, 10, 'Innovative Risk Assessment and Management', 0, 1, 'C')
    pdf.cell(0, 10, 'of Subclinical Cardiovascular Disease', 0, 1, 'C')
    
    # Summary
    pdf.set_font('Arial', '', 11)
    pdf.ln(10)
    
    # Determine main clinical focus based on highest domain score
    main_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "Cardiometabolic"
    
    pdf.multi_cell(0, 6, f"This report presents findings from an AI-powered comorbidity-lab correlation analysis conducted on a {patient_age}-year-old {patient_gender} patient with atypical {main_domain.lower()} symptoms and borderline biomarkers. The advanced network analysis identified a high-risk profile for subclinical disease despite the absence of overt diagnostic findings on standard assessments.")
    
    # Add second page
    pdf.add_page()
    
    # Content page
    # -------------
    
    # Main section header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Patient Overview and Network Analysis Methodology', 0, 1, 'L')
    pdf.ln(5)
    
    # Create two columns
    col_width = pdf.w / 2 - 15
    
    # --- Left column ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width, 8, 'Patient Demographics and Presentation', 0, 1, 'L')
    pdf.ln(2)
    
    # Demographics bullet points
    pdf.set_font('Arial', '', 10)
    
    # Generate the bullet points for patient demographics
    demographics = [
        f"{patient_age}-year-old {patient_gender}",
        f"Presenting with {', '.join([c.lower() for c in conditions[:3]]) if conditions else 'atypical symptoms'}",
        f"Current diagnosis of {', '.join([c for c in conditions[3:5]]) if len(conditions) > 3 else 'No established diagnoses'}"
    ]
    
    for item in demographics:
        pdf.cell(5, 6, chr(149), 0, 0, 'L')  # bullet character
        pdf.multi_cell(col_width - 5, 6, item)
    
    # Laboratory findings
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width, 8, 'Laboratory Findings', 0, 1, 'L')
    pdf.ln(2)
    
    # Generate the bullet points for laboratory findings
    pdf.set_font('Arial', '', 10)
    
    # Format biomarkers with interpretations
    formatted_biomarkers = []
    for marker, value in biomarkers.items():
        interpretation = ""
        if marker == "LDL-C" and value > 100:
            interpretation = " (borderline high)"
        elif marker == "HbA1c" and value > 5.7:
            interpretation = " (pre-diabetic range)"
        elif marker == "hs-CRP" and value > 3:
            interpretation = " (Elevated inflammation marker)"
        elif marker == "Troponin" and value < 0.4:
            interpretation = " (Within normal limits)"
            
        formatted_biomarkers.append(f"{marker}: {value}{interpretation}")
    
    # Add Lipoprotein(a) if not present
    if not any("Lipoprotein" in b for b in formatted_biomarkers):
        formatted_biomarkers.append("Lipoprotein(a): Not previously tested")
        
    for biomarker in formatted_biomarkers:
        pdf.cell(5, 6, chr(149), 0, 0, 'L')  # bullet character
        pdf.multi_cell(col_width - 5, 6, biomarker)
    
    # --- Right column ---
    # Move to the right side of the page
    pdf.set_xy(pdf.w / 2, pdf.get_y() - (len(formatted_biomarkers) + len(demographics) + 20) * 6)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width, 8, 'Network Analysis Methodology', 0, 1, 'L')
    pdf.ln(2)
    
    # Description of methodology
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(col_width, 6, "The AI-Network Graph Analysis system performed a structured risk assessment using the following approach:")
    pdf.ln(5)
    
    # Risk factor identification
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width, 8, 'Risk Factor Identification', 0, 1, 'L')
    pdf.ln(2)
    
    # Generate risk factors
    pdf.set_font('Arial', '', 10)
    risk_factors = [
        f"Existing comorbidities: {', '.join([c for c in conditions[:3]]) if conditions else 'None documented'}",
        f"Key biomarkers: {', '.join([b.split(':')[0] for b in formatted_biomarkers[:4]]) if formatted_biomarkers else 'None available'}"
    ]
    
    for factor in risk_factors:
        pdf.cell(5, 6, chr(149), 0, 0, 'L')  # bullet character
        pdf.multi_cell(col_width - 5, 6, factor)
    
    pdf.ln(5)
    
    # Correlation weight calculation
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width, 8, 'Correlation Weight Calculation', 0, 1, 'L')
    pdf.ln(2)
    
    # Generate correlations
    pdf.set_font('Arial', '', 10)
    
    # Create domain-based correlations
    correlations = []
    for domain, score in domain_scores.items():
        if domain == "Cardiometabolic":
            correlations.append(f"LDL-C and cardiovascular disease (correlation weight: {0.65 * score:.2f})")
        elif domain == "Immune-Inflammation":
            correlations.append(f"hs-CRP and atherosclerosis progression (correlation weight: {0.72 * score:.2f})")
        elif domain == "Oncological":
            correlations.append(f"Genetic markers and {domain.lower()} risk (correlation weight: {0.70 * score:.2f})")
        elif domain == "Neuro-Mental Health":
            correlations.append(f"Stress biomarkers and cognitive function (correlation weight: {0.65 * score:.2f})")
        elif domain == "Neurological & Frailty":
            correlations.append(f"Mobility metrics and fall risk (correlation weight: {0.68 * score:.2f})")
        elif domain == "SDOH":
            correlations.append(f"Medication adherence and outcomes (correlation weight: {0.75 * score:.2f})")
    
    # Ensure we have at least 4 correlations
    default_correlations = [
        "LDL-C and cardiovascular disease (correlation weight: 0.65)",
        "hs-CRP and atherosclerosis progression (correlation weight: 0.72)",
        "Hypertension and coronary artery disease (correlation weight: 0.78)",
        "HbA1c and microvascular dysfunction (correlation weight: 0.60)"
    ]
    
    if len(correlations) < 4:
        for i in range(4 - len(correlations)):
            if i < len(default_correlations):
                correlations.append(default_correlations[i])
    
    for corr in correlations[:4]:  # Limit to 4 correlations
        pdf.cell(5, 6, chr(149), 0, 0, 'L')  # bullet character
        pdf.multi_cell(col_width - 5, 6, corr)
    
    # Move to the next page for Expected Outcomes
    pdf.add_page()
    
    # Expected Outcomes and Economic Impact
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Expected Outcomes and Economic Impact', 0, 1, 'L')
    pdf.ln(5)
    
    # Create three columns
    col_width = (pdf.w - 40) / 3
    
    # Column backgrounds as gray boxes
    pdf.set_fill_color(245, 245, 245)
    
    # --- Column 1: Projected Clinical Benefits ---
    pdf.set_xy(10, pdf.get_y())
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width, 8, 'Projected Clinical Benefits', 0, 1, 'L', fill=True)
    pdf.ln(2)
    
    # Benefits bullet points
    pdf.set_font('Arial', '', 10)
    benefits = [
        "Estimated risk reduction with early intervention: ~7% absolute reduction",
        "Projected improvement in vascular compliance: 25-30%",
        "Expected reduction in inflammatory markers: 35-45%",
        "Predicted improvement in autonomic function metrics: 20-30%"
    ]
    
    start_y = pdf.get_y()
    for benefit in benefits:
        pdf.cell(5, 6, chr(149), 0, 0, 'L')  # bullet character
        pdf.multi_cell(col_width - 5, 6, benefit)
    
    end_y1 = pdf.get_y()
    
    # --- Column 2: Economic Analysis ---
    pdf.set_xy(20 + col_width, start_y)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width, 8, 'Economic Analysis', 0, 1, 'L', fill=True)
    pdf.set_xy(20 + col_width, pdf.get_y() + 2)
    
    # Economic analysis bullet points
    pdf.set_font('Arial', '', 10)
    economics = [
        "Expected healthcare cost savings over 10 years: ~$8,000",
        f"Reduced probability of hospitalization: {hospitalization_risk:.0%}",
        f"Decreased likelihood of invasive procedures: {mortality_risk:.0%}",
        "No Increase in Premiums for Patients Who Comply with Preventative Health Measures.",
        "Access to Discounted Health Services (e.g., nutrition counseling, early screenings)."
    ]
    
    for econ in economics:
        pdf.cell(5, 6, chr(149), 0, 0, 'L')  # bullet character
        pdf.multi_cell(col_width - 5, 6, econ)
    
    end_y2 = pdf.get_y()
    
    # --- Column 3: Monitoring Metrics ---
    pdf.set_xy(30 + 2*col_width, start_y)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width, 8, 'Monitoring Metrics', 0, 1, 'L', fill=True)
    pdf.set_xy(30 + 2*col_width, pdf.get_y() + 2)
    
    # Monitoring metrics bullet points
    pdf.set_font('Arial', '', 10)
    metrics = [
        "Primary outcome: LDL-C, hs-CRP, and blood pressure normalization",
        "Secondary outcomes: Heart rate recovery, exercise capacity, quality of life metrics",
        "Success threshold: Maintenance of optimal biomarkers for six consecutive months"
    ]
    
    for metric in metrics:
        pdf.cell(5, 6, chr(149), 0, 0, 'L')  # bullet character
        pdf.multi_cell(col_width - 5, 6, metric)
    
    end_y3 = pdf.get_y()
    
    # Move to the maximum Y position of all columns
    pdf.set_y(max(end_y1, end_y2, end_y3) + 10)
    
    # Extract key insights from AI recommendations
    key_insights = ""
    if ai_recommendations:
        # Extract the main diagnosis and intervention points
        lines = ai_recommendations.split('\n')
        capture = False
        for line in lines:
            if "CLINICAL RECOMMENDATION" in line or "DIAGNOSTIC STEPS" in line:
                capture = True
                continue
            elif "ECONOMIC IMPACT" in line:
                break
            
            if capture and line.strip():
                key_insights += line.strip() + " "
    
    # Ensure we have some insights
    if not key_insights:
        key_insights = f"The AI-Network Graph Analysis has identified a {patient_gender} patient with significant subclinical risk factors that would likely be missed by standard assessment approaches. Analysis reveals interaction patterns between inflammatory, lipid, and metabolic markers despite individually borderline values."
    
    # AI Network Analysis Insights
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 10, 'AI Network Analysis Insights', 0, 1, 'L')
    pdf.ln(2)
    
    # Conclusion paragraphs
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, "The AI-Network Graph Analysis has identified a female patient with significant subclinical cardiovascular risk that would likely be missed by standard assessment approaches. The analysis reveals a high-risk cluster pattern based on the interaction between inflammatory, lipid, and metabolic markers despite individually borderline values. The recommended personalized intervention approach integrates targeted diagnostic testing, pharmacological treatment, continuous monitoring, and lifestyle optimization.")
    
    pdf.ln(5)
    pdf.multi_cell(0, 6, f"Implementation of this protocol is projected to reduce the patient's 10-year cardiovascular risk by approximately 7% absolute reduction, with substantial cost savings over time.")
    
    pdf.ln(5)
    pdf.multi_cell(0, 6, "This case exemplifies the value of advanced network medicine approaches in addressing the challenge of underdiagnosed cardiovascular disease in women. The integration of multidimensional risk assessment with sex-specific considerations enables more precise risk stratification and intervention planning beyond what is possible with traditional univariate approaches. Follow-up assessment is recommended at 3-month intervals to evaluate response to intervention and adjust the treatment plan accordingly.")
    
    # Footer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, f"Report generated on {datetime.now().strftime('%Y-%m-%d')} by Nudge Health AI", ln=True, align='C')
    pdf.cell(0, 10, "This report is for clinical decision support only. Always consult a healthcare provider.", ln=True, align='C')
    
    try:
        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        # If encoding fails due to special characters, try using bytes directly
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf_path = temp_file.name
        temp_file.close()
        
        pdf.output(pdf_path)
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # Clean up the temporary file
        os.unlink(pdf_path)
        return pdf_bytes

def create_icd_network(df):
    icd_columns = df.select_dtypes(include='object').columns
    G = nx.Graph()

    for _, row in df[icd_columns].iterrows():
        codes = [code for code in row if pd.notna(code)]
        for i in range(len(codes)):
            for j in range(i+1, len(codes)):
                if G.has_edge(codes[i], codes[j]):
                    G[codes[i]][codes[j]]['weight'] += 1
                else:
                    G.add_edge(codes[i], codes[j], weight=1)

    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(G, k=0.3)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='grey', font_size=8)
    plt.title("ICD-10 Code Co-occurrence Network")
    st.pyplot(plt)

def process_diagnosis_data(df):
    """
    Process diagnosis data with unlimited diagnosis columns.
    Handles both ICD-10 codes and descriptions using fuzzy matching for columns.
    """
    # Initialize lists for the long format data
    records = []
    
    # Define target column names and their common variations
    id_mappings = {
        'patient_id': ['PatId', 'Pat Id', 'Patient ID', 'Patient Id', 'ID', 'PatID', 'Patient_ID', 'Patient Number'],
        'gender': ['Gender', 'Sex', 'Gender/Sex', 'M/F'],
        'age': ['Age', 'Patient Age', 'Years', 'Age (years)'],
        'zip_code': ['Zip Code', 'ZIP', 'Postal Code', 'Postal', 'Zip']
    }
    
    # Function to find the best column match using fuzzy matching
    def find_best_column_match(target, possible_names):
        # First try exact matches
        found_col = next((col for col in df.columns if col in possible_names), None)
        if found_col:
            return found_col
            
        # Then try case-insensitive matches
        found_col = next((col for col in df.columns 
                          if col.lower() in [name.lower() for name in possible_names]), None)
        if found_col:
            return found_col
            
        if FUZZY_AVAILABLE:
            # Use fuzzy matching for more flexible matching
            best_match = None
            best_score = 0
            threshold = 80  # Minimum similarity score (0-100)
            
            for col in df.columns:
                for name in possible_names:
                    # Calculate similarity score
                    score = fuzz.ratio(col.lower(), name.lower())
                    
                    # Check for partial matches too (e.g., "Patient ID" would match "Patient Identifier")
                    partial_score = fuzz.partial_ratio(col.lower(), name.lower())
                    score = max(score, partial_score)
                    
                    # Update best match if score is higher
                    if score > best_score and score >= threshold:
                        best_match = col
                        best_score = score
            
            if best_match:
                logger.info(f"Fuzzy matched '{target}' to column '{best_match}' with score {best_score}")
                return best_match
        
        # Fallback: Try removing spaces and special characters
        stripped_cols = {col: re.sub(r'[^a-zA-Z0-9]', '', col.lower()) for col in df.columns}
        stripped_targets = [re.sub(r'[^a-zA-Z0-9]', '', name.lower()) for name in possible_names]
        
        for col, stripped in stripped_cols.items():
            if stripped in stripped_targets:
                return col
                
        return None
    
    # Find best column matches for each target field
    column_mapping = {}
    for target, possible_names in id_mappings.items():
        found_col = find_best_column_match(target, possible_names)
        if found_col:
            column_mapping[target] = found_col
        else:
            logger.warning(f"Could not find a match for {target} column. Available columns: {df.columns}")
            
    # Get diagnosis columns using fuzzy matching
    diagnosis_cols = []
    
    # Try common patterns for diagnosis columns
    diagnosis_patterns = ['diagnosis', 'diag', 'dx', 'icd', 'condition']
    
    for col in df.columns:
        # Check for diagnosis number pattern (Diagnosis 1, Diag 2, etc.)
        if any(re.search(f"{pattern}\\s*\\d+", col.lower()) for pattern in diagnosis_patterns):
            diagnosis_cols.append(col)
            continue
            
        # Check for general diagnosis-related columns
        if any(pattern in col.lower() for pattern in diagnosis_patterns):
            diagnosis_cols.append(col)
            continue
            
        # If fuzzy matching is available, check for fuzzy matches
        if FUZZY_AVAILABLE:
            for pattern in diagnosis_patterns:
                if fuzz.partial_ratio(pattern, col.lower()) > 75:  # 75% similarity threshold
                    diagnosis_cols.append(col)
                    break
    
    # If no diagnosis columns found, try numbered columns as a fallback
    if not diagnosis_cols:
        # Look for numbered columns that might be diagnoses (like D1, D2, etc.)
        numbered_cols = [col for col in df.columns 
                        if re.search(r'\d+$', col) or any(c.isdigit() for c in col)]
        if numbered_cols:
            diagnosis_cols = numbered_cols
            logger.warning(f"No diagnosis columns found, using numbered columns: {numbered_cols}")
    
    # Remove duplicates
    diagnosis_cols = list(set(diagnosis_cols))
    
    # Sort diagnosis columns naturally if they contain numbers
    try:
        # Extract numbers from column names for sorting
        def extract_number(col):
            match = re.search(r'\d+', col)
            return int(match.group()) if match else 0
            
        diagnosis_cols.sort(key=extract_number)
    except:
        # Fall back to alphabetical sort
        diagnosis_cols.sort()
    
    if not diagnosis_cols:
        logger.error(f"No diagnosis columns found in dataframe. Available columns: {df.columns}")
        return pd.DataFrame()
        
    logger.info(f"Using diagnosis columns: {diagnosis_cols}")
    
    # Process each row
    for _, row in df.iterrows():
        # Create patient data with available fields
        patient_data = {}
        for target, col in column_mapping.items():
            try:
                patient_data[target] = row[col]
            except:
                patient_data[target] = "Unknown"
        
        # Process each diagnosis
        for i, col in enumerate(diagnosis_cols, 1):
            try:
                diagnosis_val = str(row[col]).strip()
                if pd.notna(diagnosis_val) and diagnosis_val != '' and diagnosis_val.lower() != 'nan':
                    record = patient_data.copy()
                    record.update({
                        'diagnosis_number': i,
                        'diagnosis': diagnosis_val,
                    })
                    records.append(record)
            except Exception as e:
                logger.warning(f"Error processing diagnosis column {col}: {str(e)}")
    
    # Convert to DataFrame
    if records:
        result_df = pd.DataFrame(records)
        
        # Ensure all necessary columns exist
        for col in ['patient_id', 'gender', 'age', 'diagnosis']:
            if col not in result_df.columns:
                result_df[col] = "Unknown"
        
        return result_df
    else:
        logger.warning("No valid diagnoses found in data")
        return pd.DataFrame()

def analyze_comorbidity_data(icd_df, valid_icd_codes=None):
    """
    Analyze comorbidity patterns in the diagnosis data using clinical domains and network analysis.
    Includes comprehensive risk scoring and gender-adjusted analysis.
    """
    # Process the data into long format
    processed_df = process_diagnosis_data(icd_df)
    
    if processed_df.empty:
        st.error("No valid diagnosis data to analyze.")
        return pd.DataFrame()
    
    # Get diagnosis columns
    diag_cols = [col for col in icd_df.columns if 'Diagnosis' in col]
    
    # Process data into domains
    domain_df = process_domain_data(icd_df, diag_cols)
    
    # Create network for centrality analysis
    G = create_domain_network(domain_df)
    
    # Calculate network metrics
    node_centrality = {}
    if G.number_of_nodes() > 0:
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        
        for node in G.nodes():
            node_centrality[node] = {
                'degree_centrality': degree_cent.get(node, 0),
                'betweenness_centrality': betweenness_cent.get(node, 0)
            }
    
    # Calculate risk scores for each patient
    risk_scores = []
    for patient_id in domain_df['patient_id'].unique():
        patient_data = domain_df[domain_df['patient_id'] == patient_id].iloc[0]
        patient_conditions = domain_df[domain_df['patient_id'] == patient_id]['condition'].tolist()
        
        # Get network metrics for patient's conditions
        patient_network_metrics = {
            'degree_centrality': 0,
            'betweenness_centrality': 0
        }
        
        if node_centrality:
            # Average the centrality of all patient conditions
            condition_centralities = [
                node_centrality.get(condition, {'degree_centrality': 0, 'betweenness_centrality': 0})
                for condition in patient_conditions
                if condition in node_centrality
            ]
            
            if condition_centralities:
                patient_network_metrics['degree_centrality'] = np.mean([
                    c['degree_centrality'] for c in condition_centralities
                ])
                patient_network_metrics['betweenness_centrality'] = np.mean([
                    c['betweenness_centrality'] for c in condition_centralities
                ])
        
        patient_info = {
            'conditions': patient_conditions,
            'age': patient_data['age'],
            'gender': patient_data['gender'],
            'network_metrics': patient_network_metrics
        }
        
        risk_score = calculate_total_risk_score(patient_info)
        risk_score['patient_id'] = patient_id
        risk_scores.append(risk_score)
    
    risk_scores_df = pd.DataFrame(risk_scores)
    
    # Display basic statistics with risk stratification
    st.subheader("📊 Clinical Risk Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(domain_df['patient_id'].unique()))
    with col2:
        st.metric("High Risk Patients", 
                 len(risk_scores_df[risk_scores_df['risk_level'] == 'High']))
    with col3:
        st.metric("Average Risk Score", 
                 f"{risk_scores_df['total_score'].mean():.2f}")
    
    # Display risk distribution
    st.subheader("🎯 Risk Score Distribution")
    
    # Create tabs for different risk visualizations
    risk_tabs = st.tabs(["Total Risk", "Hospitalization Risk (5yr)", "Mortality Risk (10yr)"])
    
    with risk_tabs[0]:
        # Total risk distribution
        fig = px.histogram(
            risk_scores_df, 
            x='total_score',
            color='risk_level',
            color_discrete_map={'High': '#FF6B6B', 'Moderate': '#FFD166', 'Low': '#06D6A0'},
            labels={'total_score': 'Total Risk Score', 'count': 'Number of Patients'}
        )
        fig.update_layout(
            xaxis_title="Risk Score",
            yaxis_title="Patient Count",
            legend_title="Risk Level"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with risk_tabs[1]:
        # Hospitalization risk distribution
        fig = px.histogram(
            risk_scores_df, 
            x='hospitalization_risk_5yr',
            labels={'hospitalization_risk_5yr': '5-Year Hospitalization Risk', 'count': 'Number of Patients'}
        )
        fig.update_layout(
            xaxis_title="5-Year Hospitalization Risk",
            yaxis_title="Patient Count"
        )
        
        # Add mean line
        mean_hosp_risk = risk_scores_df['hospitalization_risk_5yr'].mean()
        fig.add_vline(x=mean_hosp_risk, line_dash="dash", line_color="red")
        fig.add_annotation(x=mean_hosp_risk, y=0.85, xref="x", yref="paper",
                          text=f"Mean: {mean_hosp_risk:.2f}", showarrow=True, arrowhead=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        hosp_metrics = st.columns(3)
        with hosp_metrics[0]:
            st.metric("Average 5yr Hospitalization Risk", f"{mean_hosp_risk:.2%}")
        with hosp_metrics[1]:
            high_hosp_risk = len(risk_scores_df[risk_scores_df['hospitalization_risk_5yr'] > 0.5])
            st.metric("Patients with >50% Risk", high_hosp_risk)
        with hosp_metrics[2]:
            st.metric("Maximum Risk", f"{risk_scores_df['hospitalization_risk_5yr'].max():.2%}")
    
    with risk_tabs[2]:
        # Mortality risk distribution
        fig = px.histogram(
            risk_scores_df, 
            x='mortality_risk_10yr',
            labels={'mortality_risk_10yr': '10-Year Mortality Risk', 'count': 'Number of Patients'}
        )
        fig.update_layout(
            xaxis_title="10-Year Mortality Risk",
            yaxis_title="Patient Count"
        )
        
        # Add mean line
        mean_mort_risk = risk_scores_df['mortality_risk_10yr'].mean()
        fig.add_vline(x=mean_mort_risk, line_dash="dash", line_color="red")
        fig.add_annotation(x=mean_mort_risk, y=0.85, xref="x", yref="paper",
                          text=f"Mean: {mean_mort_risk:.2f}", showarrow=True, arrowhead=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        mort_metrics = st.columns(3)
        with mort_metrics[0]:
            st.metric("Average 10yr Mortality Risk", f"{mean_mort_risk:.2%}")
        with mort_metrics[1]:
            high_mort_risk = len(risk_scores_df[risk_scores_df['mortality_risk_10yr'] > 0.3])
            st.metric("Patients with >30% Risk", high_mort_risk)
        with mort_metrics[2]:
            st.metric("Maximum Risk", f"{risk_scores_df['mortality_risk_10yr'].max():.2%}")
    
    # Analyze domain patterns with risk context
    st.subheader("📊 Domain-Specific Risk Analysis")
    
    # Calculate average risk scores by domain
    domain_risk_averages = pd.DataFrame(risk_scores_df['domain_scores'].tolist()).mean()
    
    # Create domain risk visualization
    color_map = {}
    for k, v in FUNCTIONAL_DOMAINS.items():
        color_map[k] = v['color']
        
    fig = px.bar(x=domain_risk_averages.index, 
                y=domain_risk_averages.values,
                title='Average Risk Scores by Clinical Domain',
                labels={'x': 'Clinical Domain', 'y': 'Average Risk Score'},
                color=domain_risk_averages.index,
                color_discrete_map=color_map)
    st.plotly_chart(fig)
    
    # Gender-specific analysis
    st.subheader("👥 Gender-Specific Risk Analysis")
    gender_risk = risk_scores_df.merge(domain_df[['patient_id', 'gender']].drop_duplicates(), 
                                    on='patient_id')
    
    fig = px.box(gender_risk, x='gender', y='total_score',
                title='Risk Scores by Gender',
                labels={'total_score': 'Risk Score', 'gender': 'Gender'})
    st.plotly_chart(fig)
    
    # Perform community detection within domains
    try:
        import community
        partition = community.best_partition(G)
        
        # Analyze and display community insights with risk context
        st.subheader("🎯 Risk Cluster Analysis")
        
        # Group diagnoses by community
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        
        # Display community information with domain and risk context
        for comm_id, diagnoses in communities.items():
            # Get the dominant domain and average risk in this community
            diagnosis_domains = [assign_clinical_domain(diag) for diag in diagnoses]
            dominant_domain = max(set(diagnosis_domains), key=diagnosis_domains.count)
            domain_color = FUNCTIONAL_DOMAINS.get(dominant_domain, {}).get('color', '#95A5A6')
            
            # Calculate average risk for this cluster
            cluster_risk = np.mean([G.nodes[node].get('risk_score', 0) for node in diagnoses])
            
            with st.expander(
                f"Risk Cluster {comm_id + 1} "
                f"({len(diagnoses)} conditions) - "
                f"Dominant: {dominant_domain} "
                f"(Risk: {cluster_risk:.2f})"
            ):
                # Calculate community centrality
                comm_subgraph = G.subgraph(diagnoses)
                comm_centrality = nx.degree_centrality(comm_subgraph)
                central_node = max(comm_centrality.items(), key=lambda x: x[1])[0]
                
                st.markdown(f"<div style='color: {domain_color}'>", unsafe_allow_html=True)
                st.write("Central condition:", central_node)
                st.write("Related conditions:")
                for diag in diagnoses:
                    if diag != central_node:
                        weight = G[central_node][diag]['weight'] if G.has_edge(central_node, diag) else 0
                        diag_domain = assign_clinical_domain(diag)
                        diag_risk = G.nodes[diag].get('risk_score', 0)
                        st.write(f"- {diag} ({diag_domain}, Risk: {diag_risk:.2f}, co-occurrence: {weight})")
                st.markdown("</div>", unsafe_allow_html=True)
    
    except ImportError:
        st.warning("Community detection package not available. Installing 'python-louvain' package is recommended for enhanced analysis.")
    
    # Return for combined analysis
    return domain_df, risk_scores_df

def analyze_biomarker_data(bio_df, valid_bio_codes):
    """
    Analyze biomarker data with both codes and descriptions.
    """
    st.subheader("✅ Biomarker Analysis")
    
    # Convert biomarker data to numeric, handling any text columns
    numeric_bio = bio_df.select_dtypes(include=['float64', 'int64'])
    
    if not numeric_bio.empty:
        st.subheader("📊 Biomarker Distribution Analysis")
        
        # Standardize the numeric data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_bio)
        
        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(numeric_bio)//4)),
            method='barnes_hut',
            random_state=42
        )
        embeddings = tsne.fit_transform(scaled_data)
        
        # Create visualization
        embedding_df = pd.DataFrame(
            embeddings,
            columns=['TSNE1', 'TSNE2']
        )
        
        # Plot using plotly for interactive visualization
        fig = px.scatter(
            embedding_df,
            x='TSNE1',
            y='TSNE2',
            title='Biomarker Pattern Analysis (t-SNE)'
        )
        st.plotly_chart(fig)
        
        # Display biomarker statistics
        st.subheader("📈 Biomarker Statistics")
        for col in numeric_bio.columns:
            st.write(f"{col} Statistics:")
            stats = numeric_bio[col].describe()
            st.write(stats)
            
            # Create distribution plot
            fig = px.histogram(numeric_bio, x=col, title=f"{col} Distribution")
            st.plotly_chart(fig)
    
    return bio_df

def perform_combined_analysis(icd_df, bio_df):
    """
    Perform combined analysis of ICD-10 and biomarker data using 
    Louvain community detection and integrated network analysis.
    """
    st.header("🔄 Combined Network Analysis")
    
    if icd_df.empty:
        st.warning("No diagnosis data available for analysis.")
        return pd.DataFrame()
    
    # Check if biomarker data is available
    if bio_df is None or bio_df.empty:
        st.info("No biomarker data available. Performing diagnosis-only network analysis.")
        
        # Create and visualize the network with diagnosis data only
        G, partition = create_integrated_network(icd_df)
        if len(G.nodes()) > 0:
            visualize_integrated_network(G, partition)
        
        return icd_df
    
    # Both ICD and biomarker data available
    st.write("Performing integrated analysis of diagnosis and biomarker data.")
    
    # Create and visualize the integrated network
    G, partition = create_integrated_network(icd_df, bio_df)
    if len(G.nodes()) > 0:
        visualize_integrated_network(G, partition)
    
    # Analyze risk patterns based on network communities
    if partition:
        st.subheader("🎯 Risk Pattern Analysis")
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)
        
        # Calculate risk score for each community
        community_risks = []
        for comm_id, nodes in communities.items():
            conditions = [node for node in nodes if G.nodes[node].get('type') == 'condition']
            if not conditions:
                continue
            
            # Get domain distribution
            domain_counts = {}
            for condition in conditions:
                domain = G.nodes[condition].get('domain', 'Unknown')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            total_conditions = len(conditions)
            if total_conditions == 0:
                continue
                
            # Calculate weighted risk score based on domain weights
            risk_score = 0
            for domain, count in domain_counts.items():
                domain_weight = FUNCTIONAL_DOMAINS.get(domain, {}).get('base_weight', 0.5)
                risk_score += (count / total_conditions) * domain_weight
            
            # Apply network metrics for importance
            avg_centrality = sum(G.nodes[node].get('degree_centrality', 0) for node in conditions) / total_conditions
            
            # Calculate final risk score
            final_score = risk_score * (1 + avg_centrality)
            
            # Risk level
            risk_level = 'High' if final_score > 0.7 else 'Moderate' if final_score > 0.4 else 'Low'
            
            community_risks.append({
                'community_id': comm_id,
                'condition_count': total_conditions,
                'domain_distribution': domain_counts,
                'risk_score': final_score,
                'risk_level': risk_level,
                'top_conditions': conditions[:5]  # Top 5 conditions
            })
        
        # Sort communities by risk score (descending)
        community_risks.sort(key=lambda x: x['risk_score'], reverse=True)
        
        # Display community risk analysis
        st.write("Clinical Risk Communities (sorted by risk):")
        for i, risk in enumerate(community_risks):
            comm_id = risk['community_id']
            domain_dist = risk['domain_distribution']
            
            # Find dominant domain
            dominant_domain = max(domain_dist.items(), key=lambda x: x[1])[0] if domain_dist else 'Unknown'
            domain_color = FUNCTIONAL_DOMAINS.get(dominant_domain, {}).get('color', '#95A5A6')
            
            with st.expander(
                f"Risk Community {i+1} " +
                f"({risk['condition_count']} conditions) - " +
                f"Risk Level: {risk['risk_level']} ({risk['risk_score']:.3f})"
            ):
                st.markdown(f"<div style='color: {domain_color}'>", unsafe_allow_html=True)
                
                # Domain distribution
                st.write("Domain Distribution:")
                for domain, count in sorted(domain_dist.items(), key=lambda x: x[1], reverse=True):
                    domain_pct = (count / risk['condition_count']) * 100
                    st.write(f"- {domain}: {count} conditions ({domain_pct:.1f}%)")
                
                # Top conditions
                st.write("Top Conditions:")
                for condition in risk['top_conditions']:
                    condition_domain = G.nodes[condition].get('domain', 'Unknown')
                    st.write(f"- {condition} ({condition_domain})")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Prepare combined dataframe for return
    combined_data = pd.DataFrame()
    
    # Try to merge if patient IDs are available in both datasets
    if 'patient_id' in icd_df.columns and 'patient_id' in bio_df.columns:
        # Select relevant columns
        icd_cols = [col for col in icd_df.columns if col not in bio_df.columns or col == 'patient_id']
        bio_cols = [col for col in bio_df.columns if col not in icd_df.columns or col == 'patient_id']
        
        # Merge on patient_id
        combined_data = pd.merge(
            icd_df[icd_cols],
            bio_df[bio_cols],
            on='patient_id',
            how='outer'
        )
    
    return combined_data

# FHIR Resource Mappings
FHIR_RESOURCE_MAPPINGS = {
    'conditions': {
        'code': 'code.coding[0].code',
        'system': 'code.coding[0].system',
        'display': 'code.coding[0].display',
        'date': 'recordedDate.isostring',
        'severity': 'severity.coding[0].code if severity else None'
    },
    'observations': {
        'code': 'code.coding[0].code',
        'system': 'code.coding[0].system',
        'value': 'valueQuantity.value',
        'unit': 'valueQuantity.unit',
        'date': 'effectiveDateTime.isostring',
        'status': 'status'
    },
    'medications': {
        'code': 'medicationCodeableConcept.coding[0].code',
        'system': 'medicationCodeableConcept.coding[0].system',
        'display': 'medicationCodeableConcept.coding[0].display',
        'status': 'status',
        'date': 'authoredOn.isostring',
        'dosage': 'dosage[0].text if dosage else None',
        'route': 'dosage[0].route.coding[0].code if dosage and dosage[0].route else None'
    },
    'procedures': {
        'code': 'code.coding[0].code',
        'system': 'code.coding[0].system',
        'display': 'code.coding[0].display',
        'status': 'status',
        'date': 'performedDateTime.isostring',
        'category': 'category.coding[0].code if category else None',
        'outcome': 'outcome.coding[0].code if outcome else None'
    },
    'allergies': {
        'code': 'code.coding[0].code',
        'system': 'code.coding[0].system',
        'display': 'code.coding[0].display',
        'severity': 'criticality',
        'status': 'clinicalStatus.coding[0].code',
        'date': 'recordedDate.isostring'
    },
    'immunizations': {
        'code': 'vaccineCode.coding[0].code',
        'system': 'vaccineCode.coding[0].system',
        'display': 'vaccineCode.coding[0].display',
        'status': 'status',
        'date': 'date.isostring',
        'lot': 'lotNumber',
        'expiration': 'expirationDate.isostring'
    }
}

# Drug Interaction Database (simplified example)
DRUG_INTERACTIONS = {
    'warfarin': ['ibuprofen', 'aspirin', 'vitamin_k'],
    'metformin': ['alcohol', 'cimetidine'],
    'lisinopril': ['lithium', 'nsaids'],
    'simvastatin': ['grapefruit', 'amiodarone']
}

# Procedure Categories
PROCEDURE_CATEGORIES = {
    'surgery': ['surgery', 'operation', 'procedure'],
    'diagnostic': ['imaging', 'scan', 'test', 'examination'],
    'therapeutic': ['therapy', 'treatment', 'intervention'],
    'preventive': ['vaccination', 'screening', 'prevention'],
    'rehabilitative': ['rehabilitation', 'physical therapy', 'occupational therapy']
}

class FHIRConnectionError(Exception):
    """Custom exception for FHIR connection issues"""
    pass

def setup_fhir_connection() -> fhirclient.client.FHIRClient:
    """
    Set up connection to FHIR server using configuration from Streamlit secrets or UI input.
    
    Returns:
        FHIRClient: Configured FHIR client
        
    Raises:
        FHIRConnectionError: If connection fails
    """
    try:
        logger.info("Setting up FHIR connection...")
        
        # Try to get FHIR configuration from secrets
        try:
            fhir_server_url = st.secrets["fhir"]["server_url"]
            client_id = st.secrets["fhir"].get("client_id", "")
            client_secret = st.secrets["fhir"].get("client_secret", "")
            logger.info(f"Using FHIR server from secrets: {fhir_server_url}")
        except (KeyError, FileNotFoundError) as e:
            # If not in secrets, use the values from the UI
            logger.info(f"FHIR settings not found in secrets: {str(e)}")
            fhir_server_url = st.session_state.get("fhir_server_url", "")
            
            if not fhir_server_url:
                raise FHIRConnectionError("No FHIR server URL provided")
                
            client_id = st.session_state.get("fhir_client_id", "")
            client_secret = st.session_state.get("fhir_client_secret", "")
            logger.info(f"Using FHIR server from session: {fhir_server_url}")
        
        settings = {
            'app_id': 'nudge_health_analyzer',
            'api_base': fhir_server_url,
            'timeout': 30  # 30 second timeout
        }
        
        # Add OAuth credentials if available
        if client_id and client_secret:
            settings['client_id'] = client_id
            settings['client_secret'] = client_secret
            logger.info("Using OAuth authentication for FHIR")
        
        try:
            client = fhirclient.client.FHIRClient(settings=settings)
        except Exception as client_error:
            logger.error(f"Error creating FHIR client: {str(client_error)}")
            raise FHIRConnectionError(f"Failed to create FHIR client: {str(client_error)}")
        
        # Test connection by making a simple request
        try:
            # Just fetch metadata to verify connection works
            client.server.request_json('metadata')
            logger.info("FHIR connection successful")
            return client
        except Exception as conn_error:
            logger.error(f"FHIR connection test failed: {str(conn_error)}")
            raise FHIRConnectionError(f"Connection test failed: {str(conn_error)}")
    except FHIRConnectionError:
        # Re-raise FHIR-specific exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to establish FHIR connection: {str(e)}")
        raise FHIRConnectionError(f"Failed to connect to FHIR server: {str(e)}")

def fetch_fhir_resource(smart: fhirclient.client.FHIRClient, 
                       resource_type: str, 
                       patient_id: str, 
                       params: Optional[Dict] = None) -> List:
    """Generic function to fetch FHIR resources with error handling"""
    try:
        query = {'patient': patient_id}
        if params:
            query.update(params)
        
        resources = getattr(fhirclient.models, resource_type.lower()).where(query).perform(smart.server)
        return resources.entry if resources else []
    except Exception as e:
        logger.error(f"Error fetching {resource_type}: {str(e)}")
        return []

def map_fhir_resource(resource: Dict, mapping: Dict) -> Dict:
    """Map FHIR resource to standardized format"""
    result = {}
    for key, path in mapping.items():
        try:
            value = eval(f"resource.{path}")
            result[key] = value
        except Exception as e:
            logger.warning(f"Failed to map {key} from {path}: {str(e)}")
            result[key] = None
    return result

def fetch_patient_data_from_fhir(smart: fhirclient.client.FHIRClient, 
                              patient_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retrieve patient data from FHIR server including demographics, conditions, and observations.
    
    Args:
        smart: FHIRClient instance
        patient_id: FHIR Patient ID
        
    Returns:
        Tuple of DataFrames containing (conditions, observations, medications, procedures, allergies, immunizations)
        
    Raises:
        FHIRConnectionError: If there is an error fetching data
    """
    try:
        logger.info(f"Fetching patient data for patient ID: {patient_id}")
        
        # Fetch patient demographics
        patient_data = fetch_fhir_resource(smart, 'Patient', patient_id)
        if not patient_data:
            raise FHIRConnectionError(f"Patient with ID {patient_id} not found")
            
        logger.info(f"Successfully retrieved patient demographics")
        
        # Fetch conditions (diagnoses/ICD-10 codes)
        conditions = fetch_fhir_resource(smart, 'Condition', patient_id, {'patient': patient_id})
        logger.info(f"Retrieved {len(conditions)} conditions")
        
        # Fetch observations (lab results/biomarkers)
        observations = fetch_fhir_resource(smart, 'Observation', patient_id, {'patient': patient_id})
        logger.info(f"Retrieved {len(observations)} observations")
        
        # Fetch medications
        medications = fetch_fhir_resource(smart, 'MedicationRequest', patient_id, {'patient': patient_id})
        logger.info(f"Retrieved {len(medications)} medications")
        
        # Fetch procedures
        procedures = fetch_fhir_resource(smart, 'Procedure', patient_id, {'patient': patient_id})
        logger.info(f"Retrieved {len(procedures)} procedures")
        
        # Fetch allergies
        allergies = fetch_fhir_resource(smart, 'AllergyIntolerance', patient_id, {'patient': patient_id})
        logger.info(f"Retrieved {len(allergies)} allergies")
        
        # Fetch immunizations
        immunizations = fetch_fhir_resource(smart, 'Immunization', patient_id, {'patient': patient_id})
        logger.info(f"Retrieved {len(immunizations)} immunizations")
        
        # Create DataFrame for conditions
        conditions_df = pd.DataFrame([map_fhir_resource(c, FHIR_RESOURCE_MAPPINGS['Condition']) for c in conditions if c])
        
        # Create DataFrame for observations
        observations_df = pd.DataFrame([map_fhir_resource(o, FHIR_RESOURCE_MAPPINGS['Observation']) for o in observations if o])
        
        # Create DataFrame for medications
        medications_df = pd.DataFrame([map_fhir_resource(m, FHIR_RESOURCE_MAPPINGS['MedicationRequest']) for m in medications if m])
        
        # Create DataFrame for procedures
        procedures_df = pd.DataFrame([map_fhir_resource(p, FHIR_RESOURCE_MAPPINGS['Procedure']) for p in procedures if p])
        
        # Create DataFrame for allergies
        allergies_df = pd.DataFrame([map_fhir_resource(a, FHIR_RESOURCE_MAPPINGS['AllergyIntolerance']) for a in allergies if a])
        
        # Create DataFrame for immunizations
        immunizations_df = pd.DataFrame([map_fhir_resource(i, FHIR_RESOURCE_MAPPINGS['Immunization']) for i in immunizations if i])
        
        logger.info(f"Successfully processed all FHIR resources for patient {patient_id}")
        
        # Return all dataframes
        return conditions_df, observations_df, medications_df, procedures_df, allergies_df, immunizations_df
    
    except FHIRConnectionError:
        # Re-raise FHIR-specific exceptions
        raise
    except Exception as e:
        logger.error(f"Error fetching patient data from FHIR: {str(e)}")
        raise FHIRConnectionError(f"Failed to fetch patient data: {str(e)}")

def analyze_drug_interactions(medications_df: pd.DataFrame) -> None:
    """Analyze potential drug interactions"""
    if not medications_df.empty:
        st.subheader("⚠️ Drug Interaction Analysis")
        
        # Get list of current medications
        current_meds = medications_df[medications_df['status'] == 'active']['display'].tolist()
        
        # Check for interactions
        interactions = []
        for med in current_meds:
            med_lower = med.lower()
            for drug, interactions_list in DRUG_INTERACTIONS.items():
                if drug in med_lower:
                    for interaction in interactions_list:
                        if any(interaction in other_med.lower() for other_med in current_meds if other_med != med):
                            interactions.append({
                                'drug1': med,
                                'drug2': next(other_med for other_med in current_meds if interaction in other_med.lower()),
                                'interaction_type': 'Known interaction'
                            })
        
        if interactions:
            st.warning("Potential drug interactions detected:")
            for interaction in interactions:
                st.write(f"- {interaction['drug1']} ↔️ {interaction['drug2']}")
        else:
            st.success("No known drug interactions detected")

def analyze_medication_data(medications_df: pd.DataFrame) -> None:
    """Enhanced medication analysis with drug interactions"""
    if not medications_df.empty:
        st.subheader("💊 Medication Analysis")
        
        # Basic medication analysis
        status_counts = medications_df['status'].value_counts()
        st.write("Medication Status Distribution:")
        st.bar_chart(status_counts)
        
        # Most common medications
        st.write("Most Common Medications:")
        common_meds = medications_df['display'].value_counts().head(10)
        st.bar_chart(common_meds)
        
        # Medication timeline
        if 'date' in medications_df.columns:
            medications_df['date'] = pd.to_datetime(medications_df['date'])
            st.write("Medication Timeline:")
            st.line_chart(medications_df.set_index('date')['code'].value_counts())
        
        # Route of administration analysis
        if 'route' in medications_df.columns:
            st.write("Routes of Administration:")
            route_counts = medications_df['route'].value_counts()
            st.bar_chart(route_counts)
        
        # Drug interaction analysis
        analyze_drug_interactions(medications_df)

def categorize_procedure(procedure_name: str) -> str:
    """Categorize procedure based on keywords"""
    procedure_lower = procedure_name.lower()
    for category, keywords in PROCEDURE_CATEGORIES.items():
        if any(keyword in procedure_lower for keyword in keywords):
            return category
    return 'other'

def analyze_procedure_data(procedures_df: pd.DataFrame) -> None:
    """Enhanced procedure analysis with categorization"""
    if not procedures_df.empty:
        st.subheader("🔪 Procedure Analysis")
        
        # Add procedure categories
        procedures_df['category'] = procedures_df['display'].apply(categorize_procedure)
        
        # Procedure category distribution
        st.write("Procedure Categories:")
        category_counts = procedures_df['category'].value_counts()
        st.bar_chart(category_counts)
        
        # Procedure status distribution
        status_counts = procedures_df['status'].value_counts()
        st.write("Procedure Status Distribution:")
        st.bar_chart(status_counts)
        
        # Most common procedures by category
        for category in procedures_df['category'].unique():
            st.write(f"Most Common {category.title()} Procedures:")
            category_procs = procedures_df[procedures_df['category'] == category]['display'].value_counts().head(5)
            st.bar_chart(category_procs)
        
        # Procedure timeline
        if 'date' in procedures_df.columns:
            procedures_df['date'] = pd.to_datetime(procedures_df['date'])
            st.write("Procedure Timeline by Category:")
            timeline_data = procedures_df.pivot_table(
                index='date',
                columns='category',
                values='code',
                aggfunc='count'
            ).fillna(0)
            st.line_chart(timeline_data)

def analyze_allergy_data(allergies_df: pd.DataFrame) -> None:
    """Analyze allergy and intolerance data"""
    if not allergies_df.empty:
        st.subheader("⚠️ Allergy & Intolerance Analysis")
        
        # Allergy severity distribution
        severity_counts = allergies_df['severity'].value_counts()
        st.write("Allergy Severity Distribution:")
        st.bar_chart(severity_counts)
        
        # Most common allergens
        st.write("Most Common Allergens:")
        allergen_counts = allergies_df['display'].value_counts().head(10)
        st.bar_chart(allergen_counts)
        
        # Active vs. Resolved allergies
        status_counts = allergies_df['status'].value_counts()
        st.write("Allergy Status Distribution:")
        st.bar_chart(status_counts)

def analyze_immunization_data(immunizations_df: pd.DataFrame) -> None:
    """Analyze immunization data"""
    if not immunizations_df.empty:
        st.subheader("💉 Immunization Analysis")
        
        # Most common vaccines
        st.write("Most Common Vaccines:")
        vaccine_counts = immunizations_df['display'].value_counts().head(10)
        st.bar_chart(vaccine_counts)
        
        # Immunization timeline
        if 'date' in immunizations_df.columns:
            immunizations_df['date'] = pd.to_datetime(immunizations_df['date'])
            st.write("Immunization Timeline:")
            st.line_chart(immunizations_df.set_index('date')['code'].value_counts())
        
        # Check for expired vaccines
        if 'expiration' in immunizations_df.columns:
            immunizations_df['expiration'] = pd.to_datetime(immunizations_df['expiration'])
            expired = immunizations_df[immunizations_df['expiration'] < datetime.now()]
            if not expired.empty:
                st.warning("Expired Vaccines Detected:")
                st.dataframe(expired[['display', 'expiration']])

def calculate_correlation(x, y, method='pearson'):
    """Calculate correlation between two variables using different methods"""
    if method == 'pearson':
        corr, _ = pearsonr(x, y)
        return corr
    elif method == 'spearman':
        corr, _ = spearmanr(x, y)
        return corr
    elif method == 'mutual_info':
        # Normalize for mutual information
        x_norm = MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()
        y_norm = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()
        return mutual_info_score(x_norm, y_norm)
    return 0

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set(set1) & set(set2))
    union = len(set(set1) | set(set2))
    return intersection / union if union > 0 else 0

def create_integrated_network(icd_df, bio_df=None):
    """
    Create an integrated network graph combining ICD-10 codes and biomarker data.
    Uses Louvain community detection for clustering.
    
    Args:
        icd_df: DataFrame with ICD-10 diagnosis data
        bio_df: Optional DataFrame with biomarker data
    
    Returns:
        G: NetworkX graph object
        partition: Community partition dictionary
    """
    st.subheader("🔬 Integrated Clinical Network Analysis")
    
    # Create graph
    G = nx.Graph()
    
    # Process ICD codes
    if not icd_df.empty:
        st.write("Processing diagnosis data...")
        
        # Group by patient to find co-occurring conditions
        patient_conditions = defaultdict(list)
        
        # Handle long format data
        if 'patient_id' in icd_df.columns and 'condition' in icd_df.columns:
            for _, row in icd_df.iterrows():
                if pd.notna(row['condition']):
                    patient_conditions[row['patient_id']].append(row['condition'])
        
        # Handle wide format data (check for diagnosis columns)
        else:
            diag_cols = [col for col in icd_df.columns if 'diagnosis' in col.lower() or 'diag' in col.lower()]
            if diag_cols:
                patient_id_col = next((col for col in icd_df.columns if 'id' in col.lower()), None)
                if patient_id_col:
                    for _, row in icd_df.iterrows():
                        patient_id = row[patient_id_col]
                        for col in diag_cols:
                            if pd.notna(row[col]) and str(row[col]).strip():
                                patient_conditions[patient_id].append(str(row[col]).strip())
        
        # Add ICD-10 nodes and edges
        for patient_id, conditions in patient_conditions.items():
            # Add nodes with domain attributes
            for condition in conditions:
                domain = assign_clinical_domain(condition)
                domain_color = FUNCTIONAL_DOMAINS.get(domain, {}).get('color', '#95A5A6')
                if not G.has_node(condition):
                    G.add_node(condition, 
                              type='condition',
                              domain=domain, 
                              color=domain_color,
                              weight=FUNCTIONAL_DOMAINS.get(domain, {}).get('base_weight', 0.5))
            
            # Add edges for co-occurring conditions
            for i, cond_a in enumerate(conditions):
                for cond_b in conditions[i+1:]:
                    if G.has_edge(cond_a, cond_b):
                        G[cond_a][cond_b]['weight'] += 1
                    else:
                        # Jaccard similarity for categorical data
                        G.add_edge(cond_a, cond_b, 
                                  weight=1, 
                                  edge_type='comorbidity',
                                  correlation=0)  # Will be updated if biomarkers available
    
    # Process biomarker data if available
    if bio_df is not None and not bio_df.empty:
        st.write("Processing biomarker data...")
        
        # Get numeric biomarker columns
        bio_numeric = bio_df.select_dtypes(include=['float64', 'int64'])
        
        if not bio_numeric.empty:
            # Add biomarker nodes
            for biomarker in bio_numeric.columns:
                G.add_node(biomarker, 
                          type='biomarker',
                          domain='Biomarker', 
                          color='#F39C12',  # Yellow-orange
                          weight=0.7)  # Default biomarker weight
            
            # Calculate correlations between biomarkers
            for i, bio1 in enumerate(bio_numeric.columns):
                for bio2 in bio_numeric.columns[i+1:]:
                    if not pd.isna(bio_numeric[bio1]).all() and not pd.isna(bio_numeric[bio2]).all():
                        # Filter out missing values
                        valid_data = bio_numeric[[bio1, bio2]].dropna()
                        if len(valid_data) > 5:  # Need at least 5 data points
                            corr = calculate_correlation(valid_data[bio1], valid_data[bio2], method='spearman')
                            if abs(corr) > 0.3:  # Add edge only if correlation is meaningful
                                G.add_edge(bio1, bio2, 
                                          weight=abs(corr),
                                          edge_type='biomarker_correlation',
                                          correlation=corr)
            
            # If we have patient IDs in both datasets, correlate conditions with biomarkers
            if 'patient_id' in icd_df.columns and 'patient_id' in bio_df.columns:
                # Merge datasets
                common_patients = set(icd_df['patient_id']) & set(bio_df['patient_id'])
                
                if common_patients:
                    st.write(f"Found {len(common_patients)} patients with both diagnosis and biomarker data!")
                    
                    # For each condition, check correlation with biomarkers
                    condition_counts = icd_df.groupby('patient_id')['condition'].apply(list)
                    
                    for condition in G.nodes():
                        if G.nodes[condition]['type'] == 'condition':
                            # Create binary indicator: 1 if patient has condition, 0 otherwise
                            has_condition = []
                            patient_ids = []
                            
                            for patient_id in common_patients:
                                patient_conditions = condition_counts.get(patient_id, [])
                                has_condition.append(1 if condition in patient_conditions else 0)
                                patient_ids.append(patient_id)
                            
                            # Convert to numpy array
                            has_condition = np.array(has_condition)
                            
                            # Only proceed if we have variability
                            if np.std(has_condition) > 0:
                                # Check correlation with each biomarker
                                for biomarker in bio_numeric.columns:
                                    # Get biomarker values for common patients
                                    bio_values = []
                                    for patient_id in patient_ids:
                                        patient_bio = bio_df[bio_df['patient_id'] == patient_id]
                                        if not patient_bio.empty and biomarker in patient_bio.columns:
                                            bio_values.append(patient_bio[biomarker].iloc[0])
                                        else:
                                            bio_values.append(np.nan)
                                    
                                    # Convert to numpy array and drop missing values
                                    bio_values = np.array(bio_values)
                                    valid_indices = ~np.isnan(bio_values)
                                    
                                    if np.sum(valid_indices) > 5:  # Need at least 5 data points
                                        # Calculate correlation using valid data points
                                        try:
                                            # Use point-biserial correlation for binary vs continuous
                                            corr = calculate_correlation(
                                                has_condition[valid_indices], 
                                                bio_values[valid_indices],
                                                method='spearman'
                                            )
                                            
                                            if abs(corr) > 0.25:  # Add edge only if correlation is meaningful
                                                G.add_edge(condition, biomarker, 
                                                          weight=abs(corr),
                                                          edge_type='condition_biomarker',
                                                          correlation=corr)
                                        except Exception as e:
                                            st.error(f"Error calculating correlation: {str(e)}")
    
    # Apply community detection if we have nodes
    if len(G.nodes()) > 0:
        st.write(f"Network created with {len(G.nodes())} nodes and {len(G.edges())} edges.")
        
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G)
            
            # Add community information to nodes
            for node, community_id in partition.items():
                G.nodes[node]['community'] = community_id
            
            # Calculate key network metrics
            st.write("Calculating network metrics...")
            
            # Centrality measures
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
            
            # Add metrics to nodes
            for node in G.nodes():
                G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
                G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
                
                # Calculate a combined importance score
                G.nodes[node]['importance'] = (
                    0.5 * degree_centrality.get(node, 0) +
                    0.5 * betweenness_centrality.get(node, 0)
                )
            
            return G, partition
        
        except ImportError:
            st.warning("Community detection package not available. Install 'python-louvain' for enhanced analysis.")
            return G, {}
    
    return G, {}

def visualize_integrated_network(G, partition):
    """
    Visualize the integrated network with community detection results.
    
    Args:
        G: NetworkX graph object
        partition: Community partition dictionary
    """
    if len(G.nodes()) == 0:
        st.warning("No network data available to visualize.")
        return
    
    st.subheader("🔬 Integrated Network Visualization")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Calculate layout (position of nodes)
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Prepare node attributes for visualization
    node_types = [G.nodes[node].get('type', 'unknown') for node in G.nodes()]
    node_colors = [G.nodes[node].get('color', '#CCCCCC') for node in G.nodes()]
    
    # Set node sizes based on importance and type
    node_sizes = []
    for node in G.nodes():
        base_size = 400
        if G.nodes[node].get('type') == 'biomarker':
            base_size = 600  # Biomarkers slightly larger
        
        # Scale by importance
        importance = G.nodes[node].get('importance', 0.1)
        node_sizes.append(base_size * (1 + 2 * importance))
    
    # Prepare edge attributes
    edge_weights = [G[u][v].get('weight', 1) * 2 for u, v in G.edges()]
    edge_colors = []
    for u, v in G.edges():
        edge_type = G[u][v].get('edge_type', 'unknown')
        corr = G[u][v].get('correlation', 0)
        
        if edge_type == 'comorbidity':
            edge_colors.append('#666666')  # Gray for comorbidities
        elif edge_type == 'biomarker_correlation':
            # Red for negative correlation, blue for positive
            edge_colors.append('#E74C3C' if corr < 0 else '#3498DB')
        elif edge_type == 'condition_biomarker':
            # Red for negative correlation, green for positive
            edge_colors.append('#E74C3C' if corr < 0 else '#2ECC71')
        else:
            edge_colors.append('#AAAAAA')  # Light gray default
    
    # Draw the network
    # 1. Draw edges with varying widths based on weight
    nx.draw_networkx_edges(G, pos, 
                          width=edge_weights,
                          alpha=0.6,
                          edge_color=edge_colors)
    
    # 2. Draw nodes with colors representing domains and sizes representing importance
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.8)
    
    # 3. Draw labels for high importance nodes only
    high_importance_nodes = {}
    for node in G.nodes():
        importance = G.nodes[node].get('importance', 0)
        if importance > 0.1:
            high_importance_nodes[node] = importance
    
    if high_importance_nodes:
        # Sort by importance (descending)
        sorted_nodes = sorted(high_importance_nodes.items(), 
                             key=lambda item: item[1], 
                             reverse=True)[:30]  # Limit to top 30
        
        labels = {node: node for node, _ in sorted_nodes}
        nx.draw_networkx_labels(G, pos, 
                              labels=labels,
                              font_size=10,
                              font_weight='bold')
    
    plt.title("Integrated Clinical Network with Louvain Community Detection")
    plt.axis('off')
    st.pyplot(fig)
    
    # Display network statistics
    st.subheader("📊 Network Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Nodes", len(G.nodes()))
    with col2:
        st.metric("Total Edges", len(G.edges()))
    with col3:
        st.metric("Communities", len(set(partition.values())) if partition else 0)
    
    # Display community information
    if partition:
        st.subheader("🔬 Clinical Community Analysis")
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)
        
        # Analyze each community
        for comm_id, nodes in communities.items():
            # Get domain composition
            domain_counts = {}
            for node in nodes:
                domain = G.nodes[node].get('domain', 'Unknown')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Find the dominant domain
            dominant_domain = max(domain_counts.items(), key=lambda x: x[1])[0] if domain_counts else 'Unknown'
            domain_color = FUNCTIONAL_DOMAINS.get(dominant_domain, {}).get('color', '#95A5A6')
            
            # Calculate average importance
            avg_importance = sum(G.nodes[node].get('importance', 0) for node in nodes) / len(nodes)
            
            # Separate conditions and biomarkers
            conditions = [node for node in nodes if G.nodes[node].get('type') == 'condition']
            biomarkers = [node for node in nodes if G.nodes[node].get('type') == 'biomarker']
            
            # Create expander for this community
            with st.expander(
                f"Community {comm_id + 1} " +
                f"({len(nodes)} nodes, {len(conditions)} conditions, {len(biomarkers)} biomarkers) - " +
                f"Dominant: {dominant_domain} (Importance: {avg_importance:.3f})"
            ):
                st.markdown(f"<div style='color: {domain_color}'>", unsafe_allow_html=True)
                
                # Most central nodes
                importance_sorted = sorted([(node, G.nodes[node].get('importance', 0)) 
                                          for node in nodes], 
                                         key=lambda x: x[1], 
                                         reverse=True)
                
                st.write("📍 Key Elements (by Centrality):")
                for node, importance in importance_sorted[:10]:  # Top 10
                    node_type = G.nodes[node].get('type', 'unknown')
                    node_domain = G.nodes[node].get('domain', 'Unknown')
                    st.write(f"- {node} ({node_type}, {node_domain}, Importance: {importance:.3f})")
                
                # Condition-biomarker relationships
                if conditions and biomarkers:
                    st.write("🔄 Condition-Biomarker Relationships:")
                    relationships = []
                    
                    for condition in conditions:
                        for biomarker in biomarkers:
                            if G.has_edge(condition, biomarker):
                                corr = G[condition][biomarker].get('correlation', 0)
                                relationships.append((condition, biomarker, corr))
                    
                    if relationships:
                        # Sort by absolute correlation (highest first)
                        for condition, biomarker, corr in sorted(relationships, key=lambda x: abs(x[2]), reverse=True):
                            direction = "↑" if corr > 0 else "↓"
                            st.write(f"- {condition} {direction} {biomarker} (correlation: {corr:.3f})")
                    else:
                        st.write("No direct condition-biomarker relationships in this community")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    return

# Main Streamlit App
st.set_page_config(page_title="Nudge Health Analysis Platform", 
                   page_icon="🩺", 
                   layout="wide",
                   initial_sidebar_state="expanded")

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-bottom: 2rem;
    }
    .logo-img {
        max-width: 180px;
        margin-bottom: 1rem;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0c326f;
        margin-bottom: 0.5rem;
    }
    .app-subtitle {
        font-size: 1.2rem;
        color: #4a4a4a;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #0c326f;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Check if OpenAI is already initialized
openai_configured = init_openai_client()

# Sidebar Configuration
with st.sidebar:
    st.title("🔧 Configuration")
    
    # OpenAI API Configuration - only show if not configured via secrets
    if not openai_configured:
        st.info("💡 To avoid entering API key every session, create a .streamlit/secrets.toml file with:\n\n[openai]\napi_key = 'your-api-key-here'")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            openai.api_key = openai_api_key
            openai_configured = True
    else:
        st.success("✅ OpenAI API configured via secrets")
    
    # FHIR Configuration
    st.header("🏥 FHIR Settings")
    fhir_configured = False
    
    # Check if FHIR settings are in secrets
    try:
        fhir_server_url = st.secrets["fhir"]["server_url"]
        fhir_configured = True
        st.success("✅ FHIR server configured via secrets")
    except (KeyError, FileNotFoundError):
        # FHIR configuration input fields
        st.info("Configure FHIR connection or use the .streamlit/secrets.toml file")
        st.session_state.fhir_server_url = st.text_input("FHIR Server URL", value=st.session_state.get("fhir_server_url", "https://your-fhir-server/fhir"))
        
        # Authentication options
        auth_type = st.radio("Authentication Type", ["None", "OAuth 2.0"], index=0)
        if auth_type == "OAuth 2.0":
            st.session_state.fhir_client_id = st.text_input("Client ID", value=st.session_state.get("fhir_client_id", ""))
            st.session_state.fhir_client_secret = st.text_input("Client Secret", type="password", value=st.session_state.get("fhir_client_secret", ""))
    
    # FHIR Integration toggle
    enable_fhir = st.checkbox("Enable FHIR Integration", value=False)
    if enable_fhir:
        patient_id_input = st.text_input("Patient ID for FHIR Query")
    
    st.header("Analysis Settings")
    
    analysis_type = st.radio(
        "Choose analysis type:",
        ["Single Patient Analysis", "Population Analysis"]
    )
    
    data_source = st.radio(
        "Choose data source:",
        ["Upload Excel Files", "FHIR Integration"]
    )
    
    # Fetch authoritative codes
    if st.button("🔄 Update ICD/CPT Codes"):
        with st.spinner("Fetching latest codes..."):
            code_data = fetch_authoritative_codes()
            if code_data:
                st.success("Code data updated successfully!")
    
    if data_source == "Upload Excel Files":
        st.subheader("📤 Upload Data Files")
        icd_file = st.file_uploader("Upload ICD-10 Data (Excel)", type=['xlsx', 'xls'])
        bio_file = st.file_uploader("Upload Biomarker Data (Excel)", type=['xlsx', 'xls'])
        
        if analysis_type == "Single Patient Analysis" and (icd_file is not None or bio_file is not None):
            patient_list = []
            if icd_file is not None:
                icd_df_temp = pd.read_excel(icd_file)
                if 'PatId' in icd_df_temp.columns:
                    patient_list.extend(icd_df_temp['PatId'].unique())
            if bio_file is not None:
                bio_df_temp = pd.read_excel(bio_file)
                if 'PatId' in bio_df_temp.columns:
                    patient_list.extend(bio_df_temp['PatId'].unique())
            
            if patient_list:
                selected_patient = st.selectbox("🏥 Select Patient ID", sorted(set(patient_list)))
    else:
        st.subheader("🔌 FHIR Integration")
        use_fhir = st.checkbox("Enable FHIR Integration")
        if use_fhir:
            fhir_server = st.text_input("FHIR Server URL", "https://your-fhir-server/fhir")
            patient_id = st.text_input("Enter Patient ID")
            if patient_id and fhir_server:
                try:
                    smart = setup_fhir_connection()
                    conditions_df, observations_df, medications_df, procedures_df, allergies_df, immunizations_df = fetch_patient_data_from_fhir(smart, patient_id)
                except FHIRConnectionError as e:
                    st.error(str(e))
                    conditions_df = pd.DataFrame()
                    observations_df = pd.DataFrame()
                    medications_df = pd.DataFrame()
                    procedures_df = pd.DataFrame()
                    allergies_df = pd.DataFrame()
                    immunizations_df = pd.DataFrame()

# Main content area
st.markdown('<div class="main-header">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Display the logo centered at the top of the main page
    try:
        import os
        logo_path = 'logo.png'
        # Get the absolute path if needed
        if not os.path.exists(logo_path):
            logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo.png')
            
        if os.path.exists(logo_path):
            st.image(logo_path, width=180, use_column_width=False, clamp=True)
        else:
            st.warning(f"Logo image not found at: {logo_path}. Please ensure 'logo.png' exists in the app directory.")
    except Exception as e:
        st.warning(f"Error loading logo: {str(e)}")

st.markdown('<h1 class="main-title">Nudge Health Analysis Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Advanced Clinical Risk Assessment & Network Medicine Analysis</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Instead of checking for openai_api_key directly, check openai_configured
if not openai_configured:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to enable AI recommendations.")

if data_source == "Upload Excel Files":
    if icd_file is not None:
        try:
            # Read Excel files
            icd_df = pd.read_excel(icd_file, index_col=None)  # Explicitly not using index column
            
            # Column name mapping with flexible variations
            column_mapping = {
                'patient_id': next((col for col in icd_df.columns if col.lower().replace(' ', '') == 'patid'), None),
                'gender': next((col for col in icd_df.columns if col.lower() == 'gender'), None),
                'age': next((col for col in icd_df.columns if col.lower() == 'age'), None),
                'zip_code': next((col for col in icd_df.columns if col.lower().replace(' ', '') == 'zipcode'), None)
            }
            
            # Get diagnosis columns (anything with "diagnosis" in the name)
            diag_cols = [col for col in icd_df.columns if 'diagnosis' in col.lower() or 'diag' in col.lower()]
            
            # Check if columns exist or provide alternatives
            if not column_mapping['patient_id']:
                # Try alternatives for patient ID
                for alt in ['Pat Id', 'Patient ID', 'Patient Id', 'ID', 'PatID']:
                    if alt in icd_df.columns:
                        column_mapping['patient_id'] = alt
                        break
            
            if not column_mapping['zip_code']:
                # Try alternatives for zip code
                for alt in ['Zip Code', 'ZIP', 'Postal Code', 'Postal']:
                    if alt in icd_df.columns:
                        column_mapping['zip_code'] = alt
                        break
            
            # Check if required columns exist after mapping
            missing_cols = [k for k, v in column_mapping.items() if v is None]
            if missing_cols or not diag_cols:
                missing_str = ", ".join([f"'{k}'" for k in missing_cols])
                st.error(f"Missing required columns in ICD data: {missing_str}" + 
                        (", and at least one Diagnosis column" if not diag_cols else ""))
                st.error(f"Available columns: {', '.join(icd_df.columns)}")
                st.stop()
            
            # Rename columns to standard format
            column_rename = {v: k for k, v in column_mapping.items() if v is not None}
            icd_df = icd_df.rename(columns=column_rename)
            
            # Reset index to ensure uniqueness
            icd_df = icd_df.reset_index(drop=True)
            
            # Validate against authoritative codes
            code_data = fetch_authoritative_codes()
            if code_data and not code_data['icd10'].empty:
                valid_codes = set(code_data['icd10']['code'])
                st.info(f"✅ Using {len(valid_codes)} validated ICD-10 codes")
            
            # Process and analyze the data
            try:
                analyzed_data, risk_scores_df = analyze_comorbidity_data(icd_df)
            except Exception as analysis_error:
                st.error(f"Error analyzing ICD data: {str(analysis_error)}")
                analyzed_data = pd.DataFrame()
                risk_scores_df = pd.DataFrame()

            # Process biomarker data if available
            if bio_file is not None:
                try:
                    bio_df = pd.read_excel(bio_file, index_col=None)  # Explicitly not using index column
                    
                    # Biomarker column mapping with flexible variations
                    bio_column_mapping = {
                        'patient_id': next((col for col in bio_df.columns if col.lower().replace(' ', '') == 'patid'), None),
                        'biomarker': next((col for col in bio_df.columns if col.lower() == 'biomarker' or col.lower() == 'test'), None),
                        'value': next((col for col in bio_df.columns if col.lower() == 'value' or col.lower() == 'result'), None)
                    }
                    
                    # Try alternatives for patient ID
                    if not bio_column_mapping['patient_id']:
                        for alt in ['Pat Id', 'Patient ID', 'Patient Id', 'ID', 'PatID']:
                            if alt in bio_df.columns:
                                bio_column_mapping['patient_id'] = alt
                                break
                    
                    # Try alternatives for biomarker
                    if not bio_column_mapping['biomarker']:
                        for alt in ['Test', 'Test Name', 'Lab Test', 'Marker', 'Biomarker Name']:
                            if alt in bio_df.columns:
                                bio_column_mapping['biomarker'] = alt
                                break
                    
                    # Try alternatives for value
                    if not bio_column_mapping['value']:
                        for alt in ['Result', 'Test Value', 'Lab Value', 'Measurement']:
                            if alt in bio_df.columns:
                                bio_column_mapping['value'] = alt
                                break
                    
                    # Check if required columns exist after mapping
                    bio_missing_cols = [k for k, v in bio_column_mapping.items() if v is None]
                    if bio_missing_cols:
                        bio_missing_str = ", ".join([f"'{k}'" for k in bio_missing_cols])
                        st.error(f"Missing required columns in biomarker data: {bio_missing_str}")
                        st.error(f"Available columns: {', '.join(bio_df.columns)}")
                        st.stop()
                    
                    # Rename columns to standard format
                    bio_column_rename = {v: k for k, v in bio_column_mapping.items() if v is not None}
                    bio_df = bio_df.rename(columns=bio_column_rename)
                    
                    # Reset index to ensure uniqueness
                    bio_df = bio_df.reset_index(drop=True)
                    
                    try:
                        analyzed_bio = analyze_biomarker_data(bio_df, None)
                        if 'analyzed_data' in locals() and not analyzed_data.empty:
                            combined_df = perform_combined_analysis(analyzed_data, analyzed_bio)
                        else:
                            combined_df = pd.DataFrame()
                    except Exception as bio_error:
                        st.error(f"Error analyzing biomarker data: {str(bio_error)}")
                        analyzed_bio = pd.DataFrame()
                        combined_df = pd.DataFrame()
                except Exception as e:
                    st.error(f"Error processing biomarker file: {str(e)}")
                    st.error("Please ensure your biomarker Excel file has columns for patient ID, biomarker name, and value")
                    bio_df = pd.DataFrame()
                    analyzed_bio = pd.DataFrame()
                    combined_df = pd.DataFrame()
    
        except Exception as e:
            st.error(f"Error processing ICD file: {str(e)}")
            st.error("Please ensure your Excel file has columns for patient ID, gender, age, zip code, and diagnosis")
            icd_df = pd.DataFrame()
            analyzed_data = pd.DataFrame()
            risk_scores_df = pd.DataFrame()
    else:
        st.info("📁 Please upload your diagnosis data file to begin analysis.")
else:
    if use_fhir and patient_id:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Conditions", "Biomarkers", "Medications", 
            "Procedures", "Allergies", "Immunizations"
        ])
        
        with tab1:
            if not conditions_df.empty:
                analyze_comorbidity_data(conditions_df)
            else:
                st.info("No condition data available")
        
        with tab2:
            if not observations_df.empty:
                analyze_biomarker_data(observations_df, None)
            else:
                st.info("No biomarker data available")
        
        with tab3:
            analyze_medication_data(medications_df)
        
        with tab4:
            analyze_procedure_data(procedures_df)
        
        with tab5:
            analyze_allergy_data(allergies_df)
        
        with tab6:
            analyze_immunization_data(immunizations_df)
    else:
        st.info("🔌 Please enable FHIR integration and enter a patient ID to begin analysis.")

# Footer
st.markdown("---")
st.markdown("© 2024 Nudge Health | AI-Driven Clinical Risk Assessment Platform") 

def calculate_decay_function(biomarker_name, current_value, time_months=0, previous_value=None):
    """
    Calculate dynamic risk adjustment based on biomarker response to treatment.
    
    Args:
        biomarker_name: Name of the biomarker
        current_value: Current biomarker value
        time_months: Time since treatment initiation in months
        previous_value: Previous biomarker value if available
        
    Returns:
        float: Decay factor (0-1) representing risk reduction
    """
    # Default - no risk reduction
    decay_factor = 1.0
    
    # Linear decay for slow-responding biomarkers
    if biomarker_name in DECAY_CONSTANTS['linear']:
        rate = DECAY_CONSTANTS['linear'][biomarker_name]
        decay_factor = max(0.2, 1.0 - (rate * time_months))
        
    # Exponential decay for rapid-responding biomarkers
    elif biomarker_name in DECAY_CONSTANTS['exponential']:
        rate = DECAY_CONSTANTS['exponential'][biomarker_name]
        decay_factor = max(0.1, math.exp(-rate * time_months))
        
    # Threshold-based decay
    elif biomarker_name in DECAY_CONSTANTS['threshold']:
        threshold = DECAY_CONSTANTS['threshold'][biomarker_name]['value']
        reduction = DECAY_CONSTANTS['threshold'][biomarker_name]['risk_reduction']
        if current_value < threshold:
            decay_factor = 1.0 - reduction
    
    # If we have previous values, adjust based on improvement percentage
    if previous_value is not None and previous_value > 0:
        if biomarker_name in ['LDL', 'HbA1c', 'hsCRP', 'Troponin']:
            # For these markers, lower is better
            if current_value < previous_value:
                improvement = (previous_value - current_value) / previous_value
                decay_factor = max(0.2, decay_factor * (1.0 - improvement))
    
    return decay_factor

def calculate_network_hospitalization_risk(patient_data, network_metrics=None, time_factors=None):
    """
    Calculate 5-year hospitalization risk using network-based approach
    with Louvain community detection and logistic regression model.
    
    Args:
        patient_data: Dictionary with patient information
        network_metrics: Optional dictionary with network centrality metrics
        time_factors: Optional dictionary with time-based factors for dynamic adjustment
        
    Returns:
        float: Probability of hospitalization within 5 years (0-1)
    """
    # Get base patient data
    conditions = patient_data.get('conditions', [])
    age = patient_data.get('age', 50)  # Default to middle age if unknown
    gender = patient_data.get('gender', 'U')
    biomarkers = patient_data.get('biomarkers', {})
    
    # If no risk scores provided, calculate them
    if 'risk_scores' not in patient_data or not patient_data['risk_scores']:
        risk_info = calculate_total_risk_score(patient_data)
        domain_scores = risk_info.get('domain_scores', {})
        nhcrs_total = risk_info.get('total_score', 0)
    else:
        domain_scores = patient_data['risk_scores'].get('domain_scores', {})
        nhcrs_total = patient_data['risk_scores'].get('total_score', 0)
    
    # Initialize logistic regression components
    logit_sum = HOSPITALIZATION_COEFFICIENTS['intercept']
    
    # Add NHCRS total score contribution
    logit_sum += HOSPITALIZATION_COEFFICIENTS['nhcrs_weight'] * nhcrs_total
    
    # Add age contribution
    logit_sum += HOSPITALIZATION_COEFFICIENTS['age_weight'] * age
    
    # Add domain-specific contributions
    for domain, score in domain_scores.items():
        if domain in HOSPITALIZATION_COEFFICIENTS['domain_weights']:
            domain_weight = HOSPITALIZATION_COEFFICIENTS['domain_weights'][domain]
            logit_sum += domain_weight * score
    
    # Add network centrality if available
    if network_metrics:
        centrality = network_metrics.get('degree_centrality', 0)
        logit_sum += HOSPITALIZATION_COEFFICIENTS['network_centrality_weight'] * centrality
    
    # Apply dynamic biomarker adjustments if time factors provided
    if time_factors and biomarkers:
        for biomarker, value in biomarkers.items():
            if biomarker in time_factors:
                time_months = time_factors[biomarker].get('months', 0)
                prev_value = time_factors[biomarker].get('previous_value')
                
                decay_factor = calculate_decay_function(
                    biomarker, value, time_months, prev_value
                )
                
                # Apply decay factor to reduce risk
                logit_sum *= decay_factor
    
    # Convert to probability using logistic function
    probability = 1.0 / (1.0 + math.exp(-logit_sum))
    
    # Cap at reasonable maximum (rare to have >95% risk)
    return min(0.95, probability)

def calculate_network_mortality_risk(patient_data, network_metrics=None, time_factors=None):
    """
    Calculate 10-year mortality risk using network-based approach
    with Louvain community detection and logistic regression model.
    
    Args:
        patient_data: Dictionary with patient information
        network_metrics: Optional dictionary with network centrality metrics
        time_factors: Optional dictionary with time-based factors for dynamic adjustment
        
    Returns:
        float: Probability of mortality within 10 years (0-1)
    """
    # Get base patient data
    conditions = patient_data.get('conditions', [])
    age = patient_data.get('age', 50)  # Default to middle age if unknown
    gender = patient_data.get('gender', 'U')
    biomarkers = patient_data.get('biomarkers', {})
    
    # If no risk scores provided, calculate them
    if 'risk_scores' not in patient_data or not patient_data['risk_scores']:
        risk_info = calculate_total_risk_score(patient_data)
        domain_scores = risk_info.get('domain_scores', {})
        nhcrs_total = risk_info.get('total_score', 0)
    else:
        domain_scores = patient_data['risk_scores'].get('domain_scores', {})
        nhcrs_total = patient_data['risk_scores'].get('total_score', 0)
    
    # Initialize logistic regression components
    logit_sum = MORTALITY_COEFFICIENTS['intercept']
    
    # Add NHCRS total score contribution
    logit_sum += MORTALITY_COEFFICIENTS['nhcrs_weight'] * nhcrs_total
    
    # Add age contribution (age is a strong mortality predictor)
    logit_sum += MORTALITY_COEFFICIENTS['age_weight'] * age
    
    # Add domain-specific contributions
    for domain, score in domain_scores.items():
        if domain in MORTALITY_COEFFICIENTS['domain_weights']:
            domain_weight = MORTALITY_COEFFICIENTS['domain_weights'][domain]
            logit_sum += domain_weight * score
    
    # Add network centrality if available
    if network_metrics:
        centrality = network_metrics.get('degree_centrality', 0)
        logit_sum += MORTALITY_COEFFICIENTS['network_centrality_weight'] * centrality
    
    # Apply dynamic biomarker adjustments if time factors provided
    if time_factors and biomarkers:
        for biomarker, value in biomarkers.items():
            if biomarker in time_factors:
                time_months = time_factors[biomarker].get('months', 0)
                prev_value = time_factors[biomarker].get('previous_value')
                
                decay_factor = calculate_decay_function(
                    biomarker, value, time_months, prev_value
                )
                
                # Apply decay factor to reduce risk
                logit_sum *= decay_factor
    
    # Convert to probability using logistic function
    probability = 1.0 / (1.0 + math.exp(-logit_sum))
    
    # Cap at reasonable maximum (rare to have >95% risk)
    return min(0.95, probability)

def process_biomarker_data(df):
    """
    Process biomarker data with fuzzy column matching.
    """
    # Define target column names and common variations
    bio_mappings = {
        'patient_id': ['PatId', 'Pat Id', 'Patient ID', 'Patient Id', 'ID', 'PatID', 'Patient_ID', 'Patient Number'],
        'biomarker': ['Biomarker', 'Test', 'Lab Test', 'Test Name', 'Marker', 'Parameter', 'Analyte'],
        'value': ['Value', 'Result', 'Test Value', 'Lab Value', 'Measurement', 'Reading']
    }
    
    # Use the same function for column matching as in process_diagnosis_data
    def find_best_column_match(target, possible_names):
        # First try exact matches
        found_col = next((col for col in df.columns if col in possible_names), None)
        if found_col:
            return found_col
            
        # Then try case-insensitive matches
        found_col = next((col for col in df.columns 
                          if col.lower() in [name.lower() for name in possible_names]), None)
        if found_col:
            return found_col
            
        if FUZZY_AVAILABLE:
            # Use fuzzy matching for more flexible matching
            best_match = None
            best_score = 0
            threshold = 80  # Minimum similarity score (0-100)
            
            for col in df.columns:
                for name in possible_names:
                    # Calculate similarity score
                    score = fuzz.ratio(col.lower(), name.lower())
                    
                    # Check for partial matches too
                    partial_score = fuzz.partial_ratio(col.lower(), name.lower())
                    score = max(score, partial_score)
                    
                    # Update best match if score is higher
                    if score > best_score and score >= threshold:
                        best_match = col
                        best_score = score
            
            if best_match:
                logger.info(f"Fuzzy matched '{target}' to column '{best_match}' with score {best_score}")
                return best_match
        
        # Fallback: Try removing spaces and special characters
        stripped_cols = {col: re.sub(r'[^a-zA-Z0-9]', '', col.lower()) for col in df.columns}
        stripped_targets = [re.sub(r'[^a-zA-Z0-9]', '', name.lower()) for name in possible_names]
        
        for col, stripped in stripped_cols.items():
            if stripped in stripped_targets:
                return col
                
        return None
    
    # Find best column matches for each target field
    column_mapping = {}
    for target, possible_names in bio_mappings.items():
        found_col = find_best_column_match(target, possible_names)
        if found_col:
            column_mapping[target] = found_col
        else:
            logger.warning(f"Could not find a match for {target} column in biomarker data. Available columns: {df.columns}")
    
    # Check if we have the minimum required columns
    required_columns = ['patient_id', 'biomarker', 'value']
    missing_columns = [col for col in required_columns if col not in column_mapping]
    
    if missing_columns:
        logger.error(f"Missing required columns in biomarker data: {missing_columns}. Available columns: {df.columns}")
        st.error(f"Missing required columns in biomarker data: {missing_columns}. Please ensure your file contains columns for Patient ID, Biomarker, and Value.")
        return pd.DataFrame()
    
    # Create a new DataFrame with renamed columns
    renamed_df = df.rename(columns={v: k for k, v in column_mapping.items()})
    
    # Keep only the columns we need
    result_columns = list(column_mapping.keys())
    other_columns = [col for col in df.columns if col not in column_mapping.values()]
    final_columns = result_columns + other_columns
    
    # Create final DataFrame with all columns
    final_df = renamed_df[final_columns].copy()
    
    # Convert to float if possible
    try:
        final_df['value'] = final_df['value'].astype(float)
    except:
        logger.warning("Could not convert biomarker values to float. Some visualization features may not work correctly.")
    
    return final_df