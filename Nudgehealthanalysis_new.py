# Import statements from the original file
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import logging
from typing import Dict, List, Optional, Tuple
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
from collections import defaultdict, Counter
import math
import os
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import zipfile
import io
import re
import openai
from fpdf import FPDF
from datetime import datetime
import time

try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

try:
    from community import community_louvain
    COMMUNITY_DETECTION_AVAILABLE = True
except ImportError:
    COMMUNITY_DETECTION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(page_title="Nudge Health AI Clinical Analyzer", layout="wide")

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_authoritative_codes():
    """
    Fetch and process ICD-10 codes from authoritative sources.
    Returns a dictionary with processed code data.
    """
    try:
        # First try to load from local cache
        if os.path.exists('icd10_codes.json'):
            with open('icd10_codes.json', 'r') as f:
                return json.load(f)
        
        # If not in cache, fetch from primary source
        icd10_url = "https://www.cms.gov/files/zip/2024-code-descriptions-tabular-order-updated-02012024.zip"
        response = requests.get(icd10_url)
        
        if response.status_code == 200:
            # Extract and process the ZIP file
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find the relevant file in the ZIP
                excel_files = [f for f in z.namelist() if f.endswith('.xlsx')]
                if not excel_files:
                    raise ValueError("No Excel files found in ZIP")
                
                # Read the first Excel file
                with z.open(excel_files[0]) as f:
                    df = pd.read_excel(io.BytesIO(f.read()))
                
                # Process the DataFrame
                codes_dict = {}
                for _, row in df.iterrows():
                    code = str(row['code']).strip()
                    desc = str(row['description']).strip()
                    codes_dict[code] = {
                        'description': desc,
                        'category': 'Unknown',  # Add categorization logic if needed
                        'risk_weight': 1.0  # Default risk weight
                    }
                
                # Cache the results
                with open('icd10_codes.json', 'w') as f:
                    json.dump(codes_dict, f)
                
                return codes_dict
    except Exception as e:
        logger.warning(f"Failed to fetch ICD codes from primary source: {str(e)}")
        # Return empty dictionary if fetch fails
        return {}

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_biomarker_reference_data():
    """
    Fetch and process biomarker reference ranges from authoritative sources.
    
    Uses the CMS Medicare Coverage Database "Billing and Coding: Biomarkers Overview"
    (Article ID: A56541) for standardized CPT/HCPCS codes and categorization.
    Reference: https://www.cms.gov/medicare-coverage-database/view/article.aspx?articleid=56541
    
    Returns:
        Dictionary with biomarker reference ranges, CPT codes, domain mappings, and metadata.
    """
    try:
        # First try to load from local cache
        if os.path.exists('biomarker_reference.json'):
            with open('biomarker_reference.json', 'r') as f:
                return json.load(f)
        
        # Create comprehensive reference data structure based on medical standards
        # and CMS Medicare Coverage Database Article A56541
        biomarker_ref = {
            # CARDIOMETABOLIC DOMAIN
            # Lipid Panel (CPT 80061) components
            'total_cholesterol': {
                'cpt_code': '82465',
                'description': 'Total Cholesterol',
                'units': 'mg/dL',
                'normal_range': [0, 200],
                'high_range': [200, 240],
                'critical_range': 240,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.6,
                'source': 'American Heart Association/CMS'
            },
            'hdl_cholesterol': {
                'cpt_code': '83718',
                'description': 'High-Density Lipoprotein',
                'units': 'mg/dL',
                'normal_range': [40, 60],
                'low_range': 40,  # Low HDL is a risk factor
                'domain': 'Cardiometabolic',
                'risk_factor': -0.5,  # Negative as higher HDL is protective
                'source': 'American Heart Association/CMS'
            },
            'ldl_cholesterol': {
                'cpt_code': '83721',
                'description': 'Low-Density Lipoprotein',
                'units': 'mg/dL',
                'normal_range': [0, 100],
                'high_range': [100, 160],
                'critical_range': 160,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.8,
                'source': 'American Heart Association/CMS'
            },
            'triglycerides': {
                'cpt_code': '84478',
                'description': 'Triglycerides',
                'units': 'mg/dL',
                'normal_range': [0, 150],
                'high_range': [150, 500],
                'critical_range': 500,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.4,
                'source': 'American Heart Association/CMS'
            },
            'lipoprotein_a': {
                'cpt_code': '83695',
                'description': 'Lipoprotein(a)',
                'units': 'nmol/L',
                'normal_range': [0, 75],
                'high_range': [75, 125],
                'critical_range': 125,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.7,
                'source': 'CMS/AHA'
            },
            
            # Diabetes markers
            'glucose': {
                'cpt_code': '82947',
                'description': 'Fasting Blood Glucose',
                'units': 'mg/dL',
                'normal_range': [70, 99],
                'high_range': [100, 125],
                'critical_range': 126,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.7,
                'source': 'American Diabetes Association/CMS'
            },
            'fasting_glucose': {
                'cpt_code': '82947',  # Same as glucose but specifically fasting
                'description': 'Fasting Blood Glucose',
                'units': 'mg/dL',
                'normal_range': [70, 99],
                'high_range': [100, 125],
                'critical_range': 126,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.7,
                'source': 'American Diabetes Association/CMS'
            },
            'hba1c': {
                'cpt_code': '83036',
                'description': 'Hemoglobin A1c',
                'units': '%',
                'normal_range': [4.0, 5.6],
                'high_range': [5.7, 6.4],
                'critical_range': 6.5,
                'domain': 'Cardiometabolic',
                'risk_factor': 1.2,
                'source': 'American Diabetes Association/CMS'
            },
            'insulin': {
                'cpt_code': '83525',
                'description': 'Insulin Level',
                'units': 'μIU/mL',
                'normal_range': [2.6, 24.9],
                'domain': 'Cardiometabolic',
                'risk_factor': 0.6,
                'source': 'CMS'
            },
            
            # Blood Pressure (manually measured)
            'blood_pressure_systolic': {
                'cpt_code': '99213',  # Office visit code where BP is typically measured
                'description': 'Systolic Blood Pressure',
                'units': 'mmHg',
                'normal_range': [90, 120],
                'high_range': [120, 140],
                'critical_range': 140,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.9,
                'source': 'American Heart Association'
            },
            'blood_pressure_diastolic': {
                'cpt_code': '99213',  # Office visit code where BP is typically measured
                'description': 'Diastolic Blood Pressure',
                'units': 'mmHg',
                'normal_range': [60, 80],
                'high_range': [80, 90],
                'critical_range': 90,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.8,
                'source': 'American Heart Association'
            },
            
            # Kidney function
            'creatinine': {
                'cpt_code': '82565',
                'description': 'Creatinine',
                'units': 'mg/dL',
                'normal_range': [0.7, 1.3],
                'high_range': [1.3, 2.0],
                'critical_range': 2.0,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.8,
                'source': 'CMS'
            },
            'egfr': {
                'cpt_code': '82565',  # Typically calculated from creatinine
                'description': 'Estimated Glomerular Filtration Rate',
                'units': 'mL/min/1.73m²',
                'normal_range': [90, 120],
                'low_range': 60,  # Below 60 indicates kidney disease
                'domain': 'Cardiometabolic',
                'risk_factor': -0.7,  # Negative as lower eGFR is worse
                'source': 'National Kidney Foundation/CMS'
            },
            'bun': {
                'cpt_code': '84520',
                'description': 'Blood Urea Nitrogen',
                'units': 'mg/dL',
                'normal_range': [7, 20],
                'high_range': [20, 40],
                'critical_range': 40,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.6,
                'source': 'CMS'
            },
            'albumin': {
                'cpt_code': '82040',
                'description': 'Albumin',
                'units': 'g/dL',
                'normal_range': [3.4, 5.4],
                'low_range': 3.4,  # Low albumin indicates issues
                'domain': 'Cardiometabolic',
                'risk_factor': -0.5,  # Negative as lower albumin is worse
                'source': 'CMS'
            },
            'microalbumin': {
                'cpt_code': '82043',
                'description': 'Microalbumin',
                'units': 'mg/L',
                'normal_range': [0, 30],
                'high_range': [30, 300],
                'critical_range': 300,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.7,
                'source': 'CMS'
            },
            
            # Cardiac markers
            'troponin': {
                'cpt_code': '84484',
                'description': 'Troponin (cardiac enzyme)',
                'units': 'ng/mL',
                'normal_range': [0, 0.04],
                'high_range': [0.04, 0.5],
                'critical_range': 0.5,
                'domain': 'Cardiometabolic',
                'risk_factor': 1.5,
                'source': 'CMS'
            },
            'bnp': {
                'cpt_code': '83880',
                'description': 'B-type Natriuretic Peptide',
                'units': 'pg/mL',
                'normal_range': [0, 100],
                'high_range': [100, 400],
                'critical_range': 400,
                'domain': 'Cardiometabolic',
                'risk_factor': 1.0,
                'source': 'CMS'
            },
            'nt_probnp': {
                'cpt_code': '83880',
                'description': 'N-terminal pro-B-type Natriuretic Peptide',
                'units': 'pg/mL',
                'normal_range': [0, 300],
                'high_range': [300, 900],
                'critical_range': 900,
                'domain': 'Cardiometabolic',
                'risk_factor': 1.0,
                'source': 'CMS'
            },
            
            # IMMUNE-INFLAMMATION DOMAIN
            # Inflammation markers
            'crp': {
                'cpt_code': '86140',
                'description': 'C-Reactive Protein',
                'units': 'mg/L',
                'normal_range': [0, 3.0],
                'high_range': [3.0, 10.0],
                'critical_range': 10.0,
                'domain': 'Immune-Inflammation',
                'risk_factor': 0.9,
                'source': 'American Heart Association/CMS'
            },
            'hs_crp': {
                'cpt_code': '86141',
                'description': 'High-Sensitivity C-Reactive Protein',
                'units': 'mg/L',
                'normal_range': [0, 1.0],
                'high_range': [1.0, 3.0],
                'critical_range': 3.0,
                'domain': 'Immune-Inflammation',
                'risk_factor': 1.0,
                'source': 'American Heart Association/CMS'
            },
            'esr': {
                'cpt_code': '85652',
                'description': 'Erythrocyte Sedimentation Rate',
                'units': 'mm/hr',
                'normal_range': [0, 20],
                'high_range': [20, 50],
                'critical_range': 50,
                'domain': 'Immune-Inflammation',
                'risk_factor': 0.8,
                'source': 'CMS'
            },
            'ferritin': {
                'cpt_code': '82728',
                'description': 'Ferritin',
                'units': 'ng/mL',
                'normal_range': [20, 250],
                'high_range': [250, 500],
                'critical_range': 500,
                'domain': 'Immune-Inflammation',
                'risk_factor': 0.7,
                'source': 'CMS'
            },
            
            # Autoimmune markers
            'ana': {
                'cpt_code': '86038',
                'description': 'Antinuclear Antibody',
                'units': 'titer',
                'normal_range': [0, 1.0],  # Below 1:40 titer
                'high_range': [1.0, 2.0],  # 1:40 to 1:80 titer
                'critical_range': 2.0,     # Above 1:80 titer
                'domain': 'Immune-Inflammation',
                'risk_factor': 0.8,
                'source': 'CMS'
            },
            'rheumatoid_factor': {
                'cpt_code': '86430',
                'description': 'Rheumatoid Factor',
                'units': 'IU/mL',
                'normal_range': [0, 14],
                'high_range': [14, 70],
                'critical_range': 70,
                'domain': 'Immune-Inflammation',
                'risk_factor': 0.7,
                'source': 'CMS'
            },
            
            # ONCOLOGICAL DOMAIN
            # Cancer markers
            'psa': {
                'cpt_code': '84153',
                'description': 'Prostate Specific Antigen',
                'units': 'ng/mL',
                'normal_range': [0, 4.0],
                'high_range': [4.0, 10.0],
                'critical_range': 10.0,
                'domain': 'Oncological',
                'risk_factor': 1.2,
                'source': 'CMS'
            },
            'cea': {
                'cpt_code': '82378',
                'description': 'Carcinoembryonic Antigen',
                'units': 'ng/mL',
                'normal_range': [0, 3.0],
                'high_range': [3.0, 10.0],
                'critical_range': 10.0,
                'domain': 'Oncological',
                'risk_factor': 1.1,
                'source': 'CMS'
            },
            'afp': {
                'cpt_code': '82105',
                'description': 'Alpha-Fetoprotein',
                'units': 'ng/mL',
                'normal_range': [0, 10.0],
                'high_range': [10.0, 100.0],
                'critical_range': 100.0,
                'domain': 'Oncological',
                'risk_factor': 1.1,
                'source': 'CMS'
            },
            'ca_125': {
                'cpt_code': '86304',
                'description': 'Cancer Antigen 125',
                'units': 'U/mL',
                'normal_range': [0, 35.0],
                'high_range': [35.0, 200.0],
                'critical_range': 200.0,
                'domain': 'Oncological',
                'risk_factor': 1.0,
                'source': 'CMS'
            },
            'ca_19_9': {
                'cpt_code': '86301',
                'description': 'Cancer Antigen 19-9',
                'units': 'U/mL',
                'normal_range': [0, 37.0],
                'high_range': [37.0, 100.0],
                'critical_range': 100.0,
                'domain': 'Oncological',
                'risk_factor': 1.0,
                'source': 'CMS'
            },
            
            # NEURO-MENTAL HEALTH DOMAIN
            # Substance screening
            'drug_screen': {
                'cpt_code': '80305',
                'description': 'Drug Screen',
                'units': 'Qualitative',
                'domain': 'Neuro-Mental Health',
                'risk_factor': 0.8,
                'source': 'CMS'
            },
            'alcohol_metabolites': {
                'cpt_code': '80320',
                'description': 'Alcohol Biomarkers',
                'units': 'Varies',
                'domain': 'Neuro-Mental Health',
                'risk_factor': 0.8,
                'source': 'CMS'
            },
            
            # Nutritional factors related to mental health
            'vitamin_b12': {
                'cpt_code': '82607',
                'description': 'Vitamin B12',
                'units': 'pg/mL',
                'normal_range': [200, 900],
                'low_range': 200,
                'domain': 'Neuro-Mental Health',
                'risk_factor': -0.6,  # Negative as lower levels are worse
                'source': 'CMS'
            },
            'folate': {
                'cpt_code': '82746',
                'description': 'Folate',
                'units': 'ng/mL',
                'normal_range': [2.0, 20.0],
                'low_range': 2.0,
                'domain': 'Neuro-Mental Health',
                'risk_factor': -0.5,  # Negative as lower levels are worse
                'source': 'CMS'
            },
            
            # NEUROLOGICAL & FRAILTY DOMAIN
            # Bone health
            'vitamin_d': {
                'cpt_code': '82306',
                'description': 'Vitamin D, 25-Hydroxy',
                'units': 'ng/mL',
                'normal_range': [30, 100],
                'low_range': 30,
                'domain': 'Neurological-Frailty',
                'risk_factor': -0.7,  # Negative as lower levels are worse
                'source': 'CMS'
            },
            
            # Thyroid function (impacts cognition and frailty)
            'tsh': {
                'cpt_code': '84443',
                'description': 'Thyroid Stimulating Hormone',
                'units': 'mIU/L',
                'normal_range': [0.4, 4.0],
                'low_range': 0.4,
                'high_range': 4.0,
                'domain': 'Neurological-Frailty',
                'risk_factor': 0.6,  # Both high and low are risks
                'source': 'CMS'
            },
            'free_t4': {
                'cpt_code': '84439',
                'description': 'Free Thyroxine (T4)',
                'units': 'ng/dL',
                'normal_range': [0.8, 1.8],
                'low_range': 0.8,
                'high_range': 1.8,
                'domain': 'Neurological-Frailty',
                'risk_factor': 0.5,  # Both high and low are risks
                'source': 'CMS'
            },
            
            # COMPLETE BLOOD COUNT (CBC)
            'wbc': {
                'cpt_code': '85025',  # Part of CBC panel
                'description': 'White Blood Cell Count',
                'units': 'K/uL',
                'normal_range': [4.5, 11.0],
                'low_range': 4.5,
                'high_range': 11.0,
                'domain': 'Immune-Inflammation',
                'risk_factor': 0.7,  # Both high and low are risks
                'source': 'CMS'
            },
            'rbc': {
                'cpt_code': '85025',  # Part of CBC panel
                'description': 'Red Blood Cell Count',
                'units': 'M/uL',
                'normal_range': [4.2, 5.8],
                'low_range': 4.2,
                'high_range': 5.8,
                'domain': 'Cardiometabolic',
                'risk_factor': 0.5,  # Both high and low are risks
                'source': 'CMS'
            },
            'hemoglobin': {
                'cpt_code': '85025',  # Part of CBC panel
                'description': 'Hemoglobin',
                'units': 'g/dL',
                'normal_range': [12.0, 16.0],
                'low_range': 12.0,
                'domain': 'Cardiometabolic',
                'risk_factor': -0.6,  # Negative as lower levels are worse
                'source': 'CMS'
            },
            'hematocrit': {
                'cpt_code': '85025',  # Part of CBC panel
                'description': 'Hematocrit',
                'units': '%',
                'normal_range': [36.0, 48.0],
                'low_range': 36.0,
                'domain': 'Cardiometabolic',
                'risk_factor': -0.6,  # Negative as lower levels are worse
                'source': 'CMS'
            },
            'platelets': {
                'cpt_code': '85025',  # Part of CBC panel
                'description': 'Platelet Count',
                'units': 'K/uL',
                'normal_range': [150, 450],
                'low_range': 150,
                'high_range': 450,
                'domain': 'Immune-Inflammation',
                'risk_factor': 0.6,  # Both high and low are risks
                'source': 'CMS'
            }
        }
        
        # Add timestamp for cache validation
        biomarker_ref['_metadata'] = {
            'updated': datetime.now().strftime('%Y-%m-%d'),
            'source': 'CMS Medicare Coverage Database Article A56541',
            'reference_url': 'https://www.cms.gov/medicare-coverage-database/view/article.aspx?articleid=56541'
        }
        
        # Cache the results
        with open('biomarker_reference.json', 'w') as f:
            json.dump(biomarker_ref, f)
        
        logger.info(f"Successfully loaded {len(biomarker_ref) - 1} biomarker references")  # -1 for metadata
        return biomarker_ref
    
    except Exception as e:
        logger.warning(f"Failed to fetch biomarker reference data: {str(e)}")
        # Return minimal default set if fetch fails
        return {
            'cholesterol': {'normal_range': [0, 200], 'units': 'mg/dL'},
            'glucose': {'normal_range': [70, 99], 'units': 'mg/dL'},
            '_metadata': {'updated': datetime.now().strftime('%Y-%m-%d'), 'source': 'Fallback data'}
        }

def process_diagnosis_data(df):
    """
    Process the input diagnosis data, handling both ICD-10 codes and text descriptions.
    Transforms wide format diagnosis data into long format with domains assigned.
    
    Args:
        df: DataFrame containing patient diagnosis data
        
    Returns:
        Processed DataFrame in long format with domains assigned
    """
    try:
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Standardize column names (case insensitive)
        df_copy.columns = [col.lower() for col in df_copy.columns]
        
        # Map common column variations to standard names
        column_mappings = {
            'pat id': 'patid', 
            'patient id': 'patid',
            'patientid': 'patid',
            'sex': 'gender',
            'age_years': 'age',
            'icd10': 'diagnosis',
            'icd_10': 'diagnosis',
            'icd code': 'diagnosis',
            'icd-10 code': 'diagnosis',
            'diagnosis code': 'diagnosis',
            'condition': 'diagnosis_text',
            'diagnosis description': 'diagnosis_text',
            'description': 'diagnosis_text'
        }
        
        # Apply column mappings where applicable
        for old_col, new_col in column_mappings.items():
            if old_col in df_copy.columns and new_col not in df_copy.columns:
                df_copy.rename(columns={old_col: new_col}, inplace=True)
        
        # Function to find the best matching column for a target
        def find_best_column_match(target, possible_names):
            # First try exact matches
            if target in df_copy.columns:
                return target
            
            # Then try partial matches
            for col in df_copy.columns:
                if any(name in col for name in possible_names):
                    return col
            
            return None
        
        # Find diagnosis columns - both codes and text descriptions
        diag_cols = []
        text_cols = []
        
        # Check for dedicated diagnosis columns
        diag_col = find_best_column_match('diagnosis', ['diagnosis', 'icd', 'code'])
        if diag_col:
            diag_cols.append(diag_col)
        
        # Check for diagnosis text/description columns
        text_col = find_best_column_match('diagnosis_text', ['text', 'desc', 'condition'])
        if text_col:
            text_cols.append(text_col)
        
        # If no dedicated columns found, look for diagnosis1, diagnosis2, etc.
        if not diag_cols:
            diag_pattern = re.compile(r'(diagnosis|icd|dx).*?(\d+)', re.IGNORECASE)
            for col in df_copy.columns:
                if diag_pattern.search(col):
                    diag_cols.append(col)
        
        # If no text description columns found, look for matching patterns
        if not text_cols:
            text_pattern = re.compile(r'(desc|text|condition).*?(\d+)', re.IGNORECASE)
            for col in df_copy.columns:
                if text_pattern.search(col):
                    text_cols.append(col)
        
        # If still no diagnosis columns found, try to infer from data patterns
        if not diag_cols:
            for col in df_copy.columns:
                # Sample values to check for ICD code patterns
                sample_values = df_copy[col].dropna().astype(str).iloc[:10].tolist()
                icd_pattern = re.compile(r'^[A-Z]\d{2}(\.\d+)?$')
                if any(icd_pattern.match(val) for val in sample_values):
                    diag_cols.append(col)
        
        # Ensure we have required columns
        required_columns = ['patid', 'age', 'gender']
        for col in required_columns:
            if col not in df_copy.columns:
                st.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Create patient_id column if it doesn't exist
        if 'patient_id' not in df_copy.columns:
            if 'patid' in df_copy.columns:
                df_copy['patient_id'] = df_copy['patid'].astype(str)
            else:
                st.error("No patient identifier column found")
                return pd.DataFrame()
        
        # Process data into long format if in wide format
        if diag_cols:
            # Check if already in long format or wide format
            if len(diag_cols) == 1 and df_copy[diag_cols[0]].notna().sum() == len(df_copy):
                # Already in long format - just need to standardize
                long_df = df_copy.copy()
                long_df.rename(columns={diag_cols[0]: 'diagnosis'}, inplace=True)
                
                # Add diagnosis_text if available
                if text_cols and len(text_cols) == 1:
                    long_df.rename(columns={text_cols[0]: 'diagnosis_text'}, inplace=True)
            else:
                # Convert from wide to long format
                id_vars = [col for col in df_copy.columns if col not in diag_cols + text_cols]
                
                # Process diagnosis codes
                long_df = pd.melt(
                    df_copy, 
                    id_vars=id_vars,
                    value_vars=diag_cols,
                    var_name='diagnosis_column',
                    value_name='diagnosis'
                )
                
                # Drop rows with missing diagnoses
                long_df = long_df[long_df['diagnosis'].notna()]
                
                # Extract diagnosis text if available and match with codes
                if text_cols:
                    # Create mapping between diagnosis columns and text columns
                    text_mapping = {}
                    
                    # Try to match diagnosis1 with description1, etc.
                    for diag_col in diag_cols:
                        diag_num = re.search(r'(\d+)', diag_col)
                        if diag_num:
                            num = diag_num.group(1)
                            for text_col in text_cols:
                                if num in text_col:
                                    text_mapping[diag_col] = text_col
                                    break
                    
                    # Add text descriptions based on mapping
                    if text_mapping:
                        def get_text_for_diagnosis(row):
                            diag_col = row['diagnosis_column']
                            if diag_col in text_mapping:
                                text_col = text_mapping[diag_col]
                                return df_copy.loc[row.name, text_col]
                            return None
                        
                        long_df['diagnosis_text'] = long_df.apply(get_text_for_diagnosis, axis=1)
            
            # Assign clinical domain for each diagnosis
            long_df['diagnosis'] = long_df['diagnosis'].astype(str).str.strip().str.upper()
            
            # Clean diagnosis codes for better pattern matching (remove punctuation)
            long_df['clean_code'] = long_df['diagnosis'].str.replace('.', '').str.strip()
            
            # Assign clinical domain for each diagnosis
            long_df['domain'] = long_df['clean_code'].apply(assign_clinical_domain)
            
            # Clean up the final dataframe
            if 'diagnosis_column' in long_df.columns:
                long_df.drop('diagnosis_column', axis=1, inplace=True)
            
            # Drop temporary column
            long_df.drop('clean_code', axis=1, inplace=True)
            
            # Rename 'diagnosis' to 'condition' for consistency
            long_df.rename(columns={'diagnosis': 'condition'}, inplace=True)
            if 'diagnosis_text' in long_df.columns:
                # If we have text descriptions and no valid ICD code in condition,
                # use the text as the condition
                mask = long_df['domain'] == 'Unknown'
                if mask.any() and 'diagnosis_text' in long_df.columns:
                    long_df.loc[mask, 'condition'] = long_df.loc[mask, 'diagnosis_text']
                    # Try to assign domain based on text description
                    long_df.loc[mask, 'domain'] = long_df.loc[mask, 'diagnosis_text'].apply(
                        lambda x: assign_domain_from_text(x) if pd.notna(x) else 'Unknown'
                    )
            
            # Filter out rows with Unknown domain if requested
            # long_df = long_df[long_df['domain'] != 'Unknown']
            
            return long_df
        else:
            st.error("No diagnosis columns found in the data")
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error processing diagnosis data: {str(e)}")
        logger.error(f"Error in process_diagnosis_data: {str(e)}")
        return pd.DataFrame()

def assign_domain_from_text(text):
    """
    Assign a clinical domain based on text description when ICD code is not available.
    
    Args:
        text: The diagnosis text description
        
    Returns:
        The clinical domain as a string
    """
    if not text or not isinstance(text, str):
        return "Unknown"
    
    text = text.lower()
    
    # Cardiometabolic keywords
    cardio_keywords = [
        'heart', 'cardiac', 'hypertension', 'diabetes', 'obesity', 'cholesterol', 
        'lipid', 'blood pressure', 'stroke', 'vascular', 'coronary', 'cardiovascular',
        'myocardial', 'infarction', 'atherosclerosis', 'artery', 'metabolic'
    ]
    
    # Immune-Inflammation keywords
    immune_keywords = [
        'arthritis', 'inflammation', 'immune', 'lupus', 'rheumatoid', 'asthma',
        'copd', 'pulmonary', 'respiratory', 'infection', 'pneumonia', 'crohn',
        'colitis', 'psoriasis', 'allergy', 'allergic', 'ankylosing', 'spondylitis'
    ]
    
    # Oncological keywords
    cancer_keywords = [
        'cancer', 'tumor', 'neoplasm', 'malignant', 'carcinoma', 'sarcoma',
        'leukemia', 'lymphoma', 'metastasis', 'metastatic', 'oncology'
    ]
    
    # Neuro-Mental Health keywords
    mental_keywords = [
        'depression', 'anxiety', 'psychiatric', 'bipolar', 'schizophrenia',
        'mental', 'psychological', 'behavior', 'mood', 'substance', 'alcohol',
        'drug', 'addiction', 'dementia', 'alzheimer', 'cognitive', 'migraine'
    ]
    
    # Neurological & Frailty keywords
    neuro_keywords = [
        'brain', 'nerve', 'seizure', 'neuropathy', 'parkinson', 'alzheimer',
        'multiple sclerosis', 'tremor', 'paralysis', 'weakness', 'frailty',
        'fall', 'mobility', 'gait', 'balance', 'osteoporosis', 'fracture'
    ]
    
    # SDOH keywords
    sdoh_keywords = [
        'housing', 'homeless', 'unemployment', 'education', 'literacy', 'food',
        'insecurity', 'social', 'isolation', 'economic', 'poverty', 'income',
        'transportation', 'access'
    ]
    
    # Check for keyword matches
    for keywords, domain in [
        (cardio_keywords, "Cardiometabolic"),
        (immune_keywords, "Immune-Inflammation"),
        (cancer_keywords, "Oncological"),
        (mental_keywords, "Neuro-Mental Health"),
        (neuro_keywords, "Neurological-Frailty"),
        (sdoh_keywords, "SDOH")
    ]:
        if any(keyword in text for keyword in keywords):
            return domain
    
    return "Other"

def analyze_comorbidity_data(icd_df, valid_icd_codes=None):
    """
    Analyze comorbidity patterns in the ICD data.
    Returns a tuple of (domain_df, risk_scores_df)
    """
    try:
        # Process the data into long format if needed
        if 'diagnosis' not in icd_df.columns:
            icd_df = process_diagnosis_data(icd_df)
            if icd_df.empty:
                logger.warning("No valid diagnosis data to analyze")
                return pd.DataFrame(), pd.DataFrame()

        # Get diagnosis columns
        diag_cols = [col for col in icd_df.columns if 'diagnosis' in col.lower()]
        
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
            
            # Create patient info dictionary with all necessary data
            patient_info = {
                'patient_id': patient_id,
                'conditions': patient_conditions,
                'age': patient_data['age'],
                'gender': patient_data['gender'],
                'network_metrics': patient_network_metrics,
                # SDOH data would be added here if available
                'sdoh_data': {}
            }
            
            # Calculate risk score using the enhanced NHCRS model
            risk_score = calculate_total_risk_score(patient_info)
            risk_scores.append(risk_score)
        
        # Convert risk scores to DataFrame
        risk_scores_df = pd.DataFrame(risk_scores)
        
        return domain_df, risk_scores_df
        
    except Exception as e:
        logger.error(f"Error in analyze_comorbidity_data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def analyze_biomarker_data(bio_df, valid_bio_codes):
    """Analyze biomarker data and return processed results"""
    try:
        if bio_df is None or bio_df.empty:
            return pd.DataFrame()
            
        # Process biomarker data
        processed_df = process_biomarker_data(bio_df)
        if processed_df.empty:
            return pd.DataFrame()
            
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in analyze_biomarker_data: {str(e)}")
        return pd.DataFrame()

def perform_combined_analysis(domain_df, biomarker_df):
    """
    Perform combined analysis of domain and biomarker data.
    
    Args:
        domain_df: DataFrame containing domain data from ICD analysis
        biomarker_df: DataFrame containing biomarker data
        
    Returns:
        None (displays results directly in Streamlit)
    """
    try:
        if domain_df.empty or biomarker_df.empty:
            st.warning("Combined analysis requires both ICD and biomarker data.")
            return
            
        # Create a combined patient-level dataset
        patient_ids = list(set(domain_df['patient_id'].unique()) & set(biomarker_df['patient_id'].unique()))
        
        if not patient_ids:
            st.warning("No matching patient IDs found between the two datasets.")
            return
            
        # Create network graph for visualization
        G = create_domain_network(domain_df)
        
        # Calculate network metrics
        if G.number_of_nodes() > 0:
            degree_cent = nx.degree_centrality(G)
            betweenness_cent = nx.betweenness_centrality(G)
            
            # Community detection using Louvain method
            try:
                # Use community detection to find clusters
                from community import best_partition
                partition = best_partition(G)
                communities = defaultdict(list)
                for node, community_id in partition.items():
                    communities[community_id].append(node)
            except:
                # Fallback if community detection fails
                communities = {0: list(G.nodes())}
        
        # Create combined risk scores for each patient
        combined_results = []
        
        for patient_id in patient_ids:
            # Get patient conditions
            patient_conditions = domain_df[domain_df['patient_id'] == patient_id]['condition'].tolist()
            
            # Get patient biomarkers
            patient_bio_row = biomarker_df[biomarker_df['patient_id'] == patient_id].iloc[0]
            
            # Extract biomarker values
            biomarkers = {}
            for col in biomarker_df.columns:
                if col != 'patient_id' and not pd.isna(patient_bio_row[col]):
                    try:
                        biomarkers[col] = float(patient_bio_row[col])
                    except:
                        pass
                        
            # Get patient demographics
            patient_data_row = domain_df[domain_df['patient_id'] == patient_id].iloc[0]
            
            # Get network metrics for this patient's conditions
            network_metrics = {
                'degree_centrality': 0,
                'betweenness_centrality': 0
            }
            
            if G.number_of_nodes() > 0:
                # Average the centrality of all patient conditions
                valid_conditions = [c for c in patient_conditions if c in G.nodes()]
                if valid_conditions:
                    network_metrics['degree_centrality'] = np.mean([
                        degree_cent.get(c, 0) for c in valid_conditions
                    ])
                    network_metrics['betweenness_centrality'] = np.mean([
                        betweenness_cent.get(c, 0) for c in valid_conditions
                    ])
            
            # Dummy SDOH data (would be replaced with real data if available)
            # Values 0-1 where 0 is best (no risk) and 1 is worst (high risk)
            sdoh_data = {
                'medication_adherence': 0.2,  # Good adherence
                'socioeconomic_status': 0.3,  # Middle class
                'housing_stability': 0.1,     # Stable housing
                'healthcare_access': 0.2,     # Good access
                'social_support': 0.3         # Moderate support
            }
            
            # Create complete patient info
            patient_info = {
                'patient_id': patient_id,
                'conditions': patient_conditions,
                'age': patient_data_row['age'],
                'gender': patient_data_row['gender'],
                'biomarkers': biomarkers,
                'network_metrics': network_metrics,
                'sdoh_data': sdoh_data
            }
            
            # Calculate enhanced risk scores
            risk_score_data = calculate_total_risk_score(patient_info)
            combined_results.append(risk_score_data)
        
        # Create DataFrame of results
        results_df = pd.DataFrame(combined_results)
        
        # Display results in Streamlit
        st.header("🔄 Combined Analysis Results")
        
        # Create metrics for overall risk
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_risk_count = len(results_df[results_df['risk_level'] == 'High'])
            st.metric("High Risk Patients", f"{high_risk_count}/{len(results_df)}")
            
        with col2:
            avg_mortality = results_df['mortality_risk_10yr'].mean()
            st.metric("Avg 10-Year Mortality Risk", f"{avg_mortality:.1f}%")
            
        with col3:
            avg_hospitalization = results_df['hospitalization_risk_5yr'].mean()
            st.metric("Avg 5-Year Hospitalization Risk", f"{avg_hospitalization:.1f}%")
        
        # Display patient-level results
        st.subheader("🏥 Patient-Level Risk Analysis")
        
        # Add a selectbox to choose a patient
        if len(patient_ids) > 1:
            selected_patient = st.selectbox("Select Patient", patient_ids)
        else:
            selected_patient = patient_ids[0]
            
        # Get the selected patient's data
        patient_row = results_df[results_df['patient_id'] == selected_patient].iloc[0]
        
        # Display risk metrics for selected patient
        st.subheader(f"Patient {selected_patient} Risk Profile")
        
        # Create metrics for patient risk
        risk_cols = st.columns(3)
        with risk_cols[0]:
            st.metric("NHCRS Score", f"{patient_row['total_score']:.1f}")
        with risk_cols[1]:
            st.metric("10-Year Mortality Risk", f"{patient_row['mortality_risk_10yr']}%")
        with risk_cols[2]:
            st.metric("5-Year Hospitalization Risk", f"{patient_row['hospitalization_risk_5yr']}%")
            
        # Create domain scores visualization
        domain_scores = patient_row['domain_scores']
        
        if domain_scores:
            # Convert to DataFrame for visualization
            domain_df = pd.DataFrame({
                'Domain': list(domain_scores.keys()),
                'Score': list(domain_scores.values())
            })
            
            # Sort by score for better visualization
            domain_df = domain_df.sort_values('Score', ascending=False)
            
            # Create bar chart
            st.subheader("Clinical Domain Risk Scores")
            fig = px.bar(
                domain_df, 
                x='Domain', 
                y='Score', 
                color='Score',
                color_continuous_scale='YlOrRd',
                title="Risk Score by Clinical Domain"
            )
            fig.update_layout(xaxis_title="Clinical Domain", yaxis_title="Risk Score")
            st.plotly_chart(fig)
            
        # Display biomarker data if available
        if biomarker_df is not None and not biomarker_df.empty:
            st.subheader("🧬 Biomarker Analysis")
            
            # Get this patient's biomarker data
            patient_biomarkers = biomarker_df[biomarker_df['patient_id'] == selected_patient]
            
            if not patient_biomarkers.empty:
                # Create a matrix of biomarker names and values
                biomarker_data = []
                
                for col in patient_biomarkers.columns:
                    if col != 'patient_id' and not pd.isna(patient_biomarkers[col].iloc[0]):
                        try:
                            value = float(patient_biomarkers[col].iloc[0])
                            status = "Normal"
                            
                            # Determine if biomarker is out of range
                            # This is a simplified example - would need real reference ranges
                            if col == 'glucose' and value > 126:
                                status = "High"
                            elif col == 'ldl' and value > 130:
                                status = "High"
                            elif col == 'hdl' and value < 40:
                                status = "Low"
                            elif col == 'blood_pressure_systolic' and value > 140:
                                status = "High"
                            
                            biomarker_data.append({
                                'Biomarker': col,
                                'Value': value,
                                'Status': status
                            })
                        except:
                            pass
                
                if biomarker_data:
                    biomarker_df = pd.DataFrame(biomarker_data)
                    
                    # Display as a table
                    st.dataframe(biomarker_df)
        
        # Network visualization (if available)
        if G.number_of_nodes() > 0:
            st.subheader("🔄 Network Analysis")
            
            # Display network information
            st.write(f"Network has {G.number_of_nodes()} conditions and {G.number_of_edges()} connections.")
            
            # Create network visualization (this would be more complex in practice)
            st.write("Network visualization would be shown here.")
        
        # Display results
        combined_df = results_df
        risk_scores = results_df['total_score'].tolist()
        domain_counts = combined_df['domain'].value_counts().to_dict()
        
        # Get the view type from session state
        view_type = st.session_state.get('view_type', 'Population Analysis')
        
        # Show the appropriate view based on user selection
        if view_type == 'Population Analysis':
            st.subheader("Population Overview")
            # Display overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patients", combined_df['patient_id'].nunique())
            with col2:
                high_risk = len([s for s in risk_scores if s > 7])
                st.metric("High Risk Patients", high_risk)
            with col3:
                avg_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
                st.metric("Average Risk Score", f"{avg_score:.2f}")
            with col4:
                # Calculate average mortality risk
                mortality_risks = {patient_id: risk_scores[i] for i, patient_id in enumerate(patient_ids)}
                avg_mortality = sum(mortality_risks.values()) / len(mortality_risks) if mortality_risks else 0
                st.metric("Avg 10-Year Mortality Risk", f"{avg_mortality:.1f}%")
            
            # Display domain distribution
            st.subheader("Clinical Domain Distribution")
            domain_df = pd.DataFrame({
                'Domain': domain_counts.keys(),
                'Count': domain_counts.values()
            })
            domain_chart = px.bar(domain_df, x='Domain', y='Count',
                              color='Domain', title="Distribution of Clinical Domains")
            st.plotly_chart(domain_chart, use_container_width=True)
            
            # Population risk distribution
            st.subheader("Population Risk Distribution")
            risk_df = pd.DataFrame({
                'Patient ID': list(risk_scores.keys()),
                'Risk Score': list(risk_scores.values())
            })
            risk_hist = px.histogram(risk_df, x='Risk Score',
                                nbins=20,
                                title="Distribution of Risk Scores",
                                labels={'Risk Score': 'NHCRS Score'})
            st.plotly_chart(risk_hist, use_container_width=True)
        
        elif view_type == 'Single Patient Analysis':
            st.subheader("Single Patient Analysis")
            # Display patient selection and details
            patient_ids = combined_df['patient_id'].unique()
            if len(patient_ids) > 0:
                selected_patient = st.selectbox("Select Patient for Detailed Analysis:", 
                                             options=patient_ids)
                
                # Get patient data
                patient_data = get_patient_data(combined_df, selected_patient)
                patient_risk = {
                    'total_score': risk_scores.get(selected_patient, 0),
                    'mortality_risk_10yr': mortality_risks.get(selected_patient, 0),
                    'hospitalization_risk_5yr': combined_results.get('hospitalization_risks', {}).get(selected_patient, 0),
                    'domain_scores': combined_results.get('domain_scores', {}).get(selected_patient, {})
                }
                
                # Display patient info
                st.subheader(f"Patient {selected_patient} Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Age: {patient_data.get('age', 'Unknown')}")
                    st.write(f"Gender: {patient_data.get('gender', 'Unknown')}")
                    
                    # Display conditions
                    st.subheader("Conditions")
                    for condition in patient_data.get('conditions', []):
                        st.write(f"- {condition}")
                
                with col2:
                    # Display risk scores
                    st.subheader("Risk Assessment")
                    st.write(f"NHCRS Score: {patient_risk['total_score']:.1f}")
                    st.write(f"10-Year Mortality Risk: {patient_risk['mortality_risk_10yr']:.1f}%")
                    st.write(f"5-Year Hospitalization Risk: {patient_risk['hospitalization_risk_5yr']:.1f}%")
                    
                    # Domain scores
                    if patient_risk['domain_scores']:
                        st.subheader("Domain Scores")
                        for domain, score in patient_risk['domain_scores'].items():
                            st.write(f"{domain}: {score:.2f}")
                
                # Compare patient to population
                st.subheader("Patient vs Population Comparison")
                
                # Risk score comparison
                compare_cols = st.columns(3)
                with compare_cols[0]:
                    patient_score = patient_risk['total_score']
                    population_avg = avg_score
                    delta = f"{patient_score - population_avg:.2f}"
                    st.metric("Risk Score", f"{patient_score:.2f}", delta=delta, 
                            delta_color="inverse")
                    
                with compare_cols[1]:
                    patient_mortality = patient_risk['mortality_risk_10yr']
                    delta_m = f"{patient_mortality - avg_mortality:.1f}%"
                    st.metric("Mortality Risk", f"{patient_mortality:.1f}%", delta=delta_m,
                            delta_color="inverse")
                    
                with compare_cols[2]:
                    # Domain count comparison
                    patient_domains = len(patient_risk['domain_scores'])
                    avg_domains = sum(len(d) for d in combined_results.get('domain_scores', {}).values()) / len(patient_ids)
                    delta_d = f"{patient_domains - avg_domains:.1f}"
                    st.metric("Clinical Domains", patient_domains, delta=delta_d,
                            delta_color="inverse")
                
                # Domain comparison chart
                if patient_risk['domain_scores']:
                    st.subheader("Domain Score Comparison")
                    # Get average domain scores across population
                    all_domain_scores = combined_results.get('domain_scores', {})
                    avg_domain_scores = {}
                    for domain in patient_risk['domain_scores'].keys():
                        domain_values = [scores.get(domain, 0) for scores in all_domain_scores.values()]
                        avg_domain_scores[domain] = sum(domain_values) / len(domain_values) if domain_values else 0
                    
                    # Create comparison dataframe
                    compare_df = pd.DataFrame({
                        'Domain': list(patient_risk['domain_scores'].keys()),
                        'Patient Score': list(patient_risk['domain_scores'].values()),
                        'Population Average': [avg_domain_scores.get(d, 0) for d in patient_risk['domain_scores'].keys()]
                    })
                    
                    # Create comparison chart
                    domain_compare = px.bar(compare_df, x='Domain', y=['Patient Score', 'Population Average'], 
                                           barmode='group', title="Domain Score Comparison")
                    st.plotly_chart(domain_compare, use_container_width=True)
                
                # Generate AI recommendations if API key available
                if 'openai_api_key' in st.session_state:
                    if st.button("Generate AI Clinical Recommendations"):
                        with st.spinner("Generating AI recommendations..."):
                            recommendations = generate_clinical_recommendations(patient_data, patient_risk)
                            st.subheader("AI Clinical Recommendations")
                            st.write(recommendations)
                            
                            # Store recommendations for PDF
                            st.session_state.recommendations = recommendations
                
                # PDF generation
                if st.button("Generate PDF Report"):
                    with st.spinner("Generating PDF report..."):
                        recommendations = st.session_state.get('recommendations', None)
                        pdf_bytes = generate_pdf_report(patient_data, patient_risk, recommendations)
                        
                        # Create download button
                        b64_pdf = base64.b64encode(pdf_bytes).decode()
                        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="patient_{selected_patient}_report.pdf">Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error in perform_combined_analysis: {str(e)}")
        st.error("Failed to perform combined analysis. Please check your data format.")
        st.write(f"Error details: {str(e)}")

def process_domain_data(df: pd.DataFrame, diag_cols: list) -> pd.DataFrame:
    """Process diagnosis data into clinical domains."""
    try:
        domain_records = []
        
        for _, row in df.iterrows():
            patient_data = {
                'patient_id': row.get('patient_id', 'Unknown'),
                'age': row.get('age', 0),
                'gender': row.get('gender', 'Unknown')
            }
            
            # Process each diagnosis
            for col in diag_cols:
                diagnosis = str(row.get(col, '')).strip()
                if pd.notna(diagnosis) and diagnosis != '' and diagnosis.lower() != 'nan':
                    domain = assign_clinical_domain(diagnosis)
                    record = patient_data.copy()
                    record.update({
                        'condition': diagnosis,
                        'domain': domain
                    })
                    domain_records.append(record)
        
        if domain_records:
            domain_df = pd.DataFrame(domain_records)
            return domain_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in process_domain_data: {str(e)}")
        return pd.DataFrame()

def assign_clinical_domain(icd_code: str) -> str:
    """
    Assigns a clinical domain to an ICD-10 code. This is a critical function that determines
    how conditions are categorized into the five functional domains.
    
    Args:
        icd_code: The ICD-10 code to categorize
        
    Returns:
        The clinical domain as a string
    """
    if not icd_code or not isinstance(icd_code, str):
        return "Unknown"
    
    # Normalize the ICD code to handle format variations
    icd_cleaned = icd_code.strip().upper().replace('.', '')
    
    # Check if it's just the first 3 characters (category) or full code
    icd_category = icd_cleaned[:3] if len(icd_cleaned) >= 3 else icd_cleaned
    
    # 1. CARDIOMETABOLIC DOMAIN
    if (
        # Hypertension and hypertensive diseases
        (icd_category in ['I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16']) or
        # Diabetes mellitus
        (icd_category in ['E08', 'E09', 'E10', 'E11', 'E12', 'E13']) or 
        # Obesity
        (icd_category == 'E66') or
        # Hyperlipidemia and dyslipidemia
        (icd_category == 'E78') or
        # Ischemic heart diseases (CAD, angina, MI)
        ('I20' <= icd_category <= 'I25') or
        # Heart failure
        (icd_category == 'I50') or
        # Atrial fibrillation/flutter
        (icd_category == 'I48') or
        # Chronic kidney disease
        (icd_category == 'N18') or
        # Nonalcoholic fatty liver disease
        (icd_cleaned.startswith('K760')) or
        # Diabetic nephropathy
        (icd_cleaned.startswith('E112')) or
        # Cerebrovascular diseases (Stroke, TIA)
        ('I60' <= icd_category <= 'I69') or
        # Peripheral vascular diseases
        ('I70' <= icd_category <= 'I79') or
        # Chronic periodontal disease
        (icd_category == 'K05') or
        # Additional cardiometabolic conditions
        (icd_category == 'E03') or  # Hypothyroidism
        (icd_cleaned.startswith('E162')) or  # Hypoglycemia
        (icd_cleaned.startswith('R030'))  # Elevated blood pressure
    ):
        return "Cardiometabolic"
    
    # 2. IMMUNE-INFLAMMATION DOMAIN
    elif (
        # Rheumatoid arthritis
        (icd_category in ['M05', 'M06']) or
        # Systemic lupus erythematosus (SLE)
        (icd_category == 'M32') or
        # Crohn's disease, Ulcerative colitis (IBD)
        (icd_category in ['K50', 'K51']) or
        # Psoriasis
        (icd_category == 'L40') or
        # Ankylosing spondylitis
        (icd_category == 'M45') or
        # Asthma
        (icd_category == 'J45') or
        # COPD
        (icd_category == 'J44') or
        # Sepsis
        (icd_category in ['A40', 'A41']) or
        # Multiple sclerosis
        (icd_category == 'G35') or
        # Periodontitis
        (icd_category == 'K05') or
        # Lupus erythematosus (cutaneous)
        (icd_category == 'L93') or
        # Systemic sclerosis (scleroderma)
        (icd_category == 'L94') or
        # GERD
        (icd_category == 'K21')
    ):
        return "Immune-Inflammation"
    
    # 3. ONCOLOGICAL (CANCER) DOMAIN
    elif (
        # All malignant neoplasms
        ('C00' <= icd_category <= 'D49') or
        # Carcinoma in situ
        ('D00' <= icd_category <= 'D09') or
        # Personal or family history of cancer
        (icd_category in ['Z80', 'Z85']) or
        # Screening for malignant neoplasms
        (icd_category == 'Z12') or
        # Neoplasms of uncertain behavior
        ('D37' <= icd_category <= 'D48') or
        # Cancer-related pain
        (icd_cleaned.startswith('G893'))
    ):
        return "Oncological"
    
    # 4. NEURO-MENTAL HEALTH DOMAIN
    elif (
        # Schizophrenia & psychotic disorders
        ('F20' <= icd_category <= 'F29') or
        # Mood disorders (depression, bipolar)
        ('F30' <= icd_category <= 'F39') or
        # Anxiety disorders
        ('F40' <= icd_category <= 'F48') or
        # Substance use disorders
        ('F10' <= icd_category <= 'F19') or
        # Dementia
        ('F01' <= icd_category <= 'F03') or
        # Eating disorders
        (icd_category == 'F50') or
        # Alzheimer's disease (cognitive/mental aspects)
        (icd_category == 'G30') or
        # Migraines
        (icd_category == 'G43') or
        # Behavioral/emotional disturbances
        (icd_category == 'R45') or
        # Sleep disorders
        (icd_category == 'G47')
    ):
        return "Neuro-Mental Health"
    
    # 5. NEUROLOGICAL & FRAILTY DOMAIN
    elif (
        # Parkinson's disease
        (icd_category == 'G20') or
        # Alzheimer's and other dementias (frailty aspect)
        ('G30' <= icd_category <= 'G31') or
        # Multiple sclerosis
        (icd_category == 'G35') or
        # Sequelae of stroke
        (icd_category == 'I69') or
        # Osteoporosis
        (icd_category in ['M80', 'M81']) or
        # Gait & mobility disorders, falls
        ('R26' <= icd_category <= 'R29') or
        # Muscle wasting & sarcopenia
        (icd_cleaned.startswith('M625')) or
        # Paralytic syndromes
        ('G81' <= icd_category <= 'G83') or
        # Osteoarthritis
        (icd_category == 'M19') or
        # Cognitive symptoms
        (icd_category == 'R41')
    ):
        return "Neurological-Frailty"
    
    # 6. SOCIAL DETERMINANTS OF HEALTH (SDOH)
    elif (
        # Education/literacy issues
        (icd_category == 'Z55') or
        # Employment/unemployment issues
        (icd_category == 'Z56') or
        # Housing instability, economic issues
        (icd_category == 'Z59') or
        # Social isolation, relationship issues
        (icd_category == 'Z60') or
        # Upbringing-related issues
        (icd_category == 'Z62') or
        # Family/social environment problems
        (icd_category == 'Z63') or
        # Legal/criminal issues affecting health
        (icd_category == 'Z65') or
        # Specific SDOH flags
        (icd_cleaned.startswith('Z590')) or  # Homelessness
        (icd_cleaned.startswith('Z594')) or  # Food insecurity
        (icd_cleaned.startswith('Z598')) or  # Housing issues
        (icd_cleaned.startswith('Z602'))     # Social isolation
    ):
        return "SDOH"
    
    # Default for unmatched codes
    return "Other"

def create_domain_network(domain_df: pd.DataFrame) -> nx.Graph:
    """
    Create a network of conditions based on co-occurrence in patients.
    
    Args:
        domain_df: DataFrame with patient diagnoses and domains
        
    Returns:
        NetworkX Graph with conditions as nodes and co-occurrence as edges
    """
    try:
        G = nx.Graph()
        
        # Ensure we have data and required columns
        if domain_df is None or domain_df.empty:
            logger.warning("No data provided for network creation")
            return G
            
        if 'patient_id' not in domain_df.columns or 'condition' not in domain_df.columns:
            logger.warning("Required columns missing from input data")
            return G
        
        # Get unique conditions and add as nodes
        conditions = domain_df['condition'].unique()
        logger.info(f"Adding {len(conditions)} nodes to network")
        
        # Add each condition as a node with its domain as attribute
        for condition in conditions:
            # Get the most common domain for this condition
            domain_counts = domain_df[domain_df['condition'] == condition]['domain'].value_counts()
            domain = domain_counts.index[0] if len(domain_counts) > 0 else 'Other'
            G.add_node(condition, domain=domain)
        
        # Group by patient to find co-occurring conditions
        patient_conditions = domain_df.groupby('patient_id')['condition'].apply(list).reset_index()
        
        # Add edges for co-occurring conditions
        edge_weights = {}
        
        # Create progress counter
        progress_counter = 0
        total_patients = len(patient_conditions)
        
        for _, row in patient_conditions.iterrows():
            # Update progress counter
            progress_counter += 1
            if progress_counter % 100 == 0:
                logger.info(f"Processing edges: {progress_counter}/{total_patients} patients")
                
            conditions_list = row['condition']
            
            # Only process if patient has multiple conditions
            if len(conditions_list) < 2:
                continue
                
            # Create all pairwise combinations of conditions
            for i in range(len(conditions_list)):
                for j in range(i+1, len(conditions_list)):
                    condition1 = conditions_list[i]
                    condition2 = conditions_list[j]
                    
                    # Skip identical conditions
                    if condition1 == condition2:
                        continue
                        
                    # Create a unique key for this pair (order doesn't matter)
                    edge_key = tuple(sorted([condition1, condition2]))
                    
                    if edge_key in edge_weights:
                        edge_weights[edge_key] += 1
                    else:
                        edge_weights[edge_key] = 1
        
        # Add edges to the graph with weights
        logger.info(f"Adding {len(edge_weights)} edges to network")
        for (condition1, condition2), weight in edge_weights.items():
            G.add_edge(condition1, condition2, weight=weight)
            
        return G
    
    except Exception as e:
        logger.error(f"Error creating domain network: {str(e)}")
        return nx.Graph()

def calculate_condition_correlations(domain_df, method='pearson'):
    """
    Calculate correlations between conditions using various methods.
    
    Args:
        domain_df: DataFrame with patient conditions
        method: Correlation method ('pearson', 'spearman', or 'mutual_info')
        
    Returns:
        DataFrame with correlation matrix
    """
    try:
        # Create a binary matrix of patients x conditions
        patients = domain_df['patient_id'].unique()
        conditions = domain_df['condition'].unique()
        
        # Initialize matrix with zeros
        binary_matrix = pd.DataFrame(0, index=patients, columns=conditions)
        
        # Fill matrix with 1s for existing conditions
        for _, row in domain_df.iterrows():
            binary_matrix.loc[row['patient_id'], row['condition']] = 1
        
        # Calculate correlations based on method
        if method == 'pearson':
            corr_matrix = binary_matrix.corr(method='pearson')
            logger.info("Calculated Pearson correlations between conditions")
        elif method == 'spearman':
            corr_matrix = binary_matrix.corr(method='spearman')
            logger.info("Calculated Spearman correlations between conditions")
        elif method == 'mutual_info':
            # Calculate mutual information for each pair of conditions
            n_conditions = len(conditions)
            mi_matrix = np.zeros((n_conditions, n_conditions))
            
            for i in range(n_conditions):
                for j in range(i, n_conditions):
                    if i == j:
                        mi_matrix[i, j] = 1.0
                    else:
                        mi = mutual_info_score(binary_matrix[conditions[i]], binary_matrix[conditions[j]])
                        # Normalize MI to 0-1 range
                        mi_matrix[i, j] = mi
                        mi_matrix[j, i] = mi
            
            corr_matrix = pd.DataFrame(mi_matrix, index=conditions, columns=conditions)
            logger.info("Calculated Mutual Information between conditions")
        else:
            logger.warning(f"Unknown correlation method: {method}. Using Pearson.")
            corr_matrix = binary_matrix.corr(method='pearson')
        
        return corr_matrix
        
    except Exception as e:
        logger.error(f"Error calculating correlations: {str(e)}")
        return pd.DataFrame()

def create_correlation_network(corr_matrix, threshold=0.3):
    """
    Create a network graph from a correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold to include edges
        
    Returns:
        NetworkX graph
    """
    try:
        G = nx.Graph()
        
        # Add nodes
        conditions = corr_matrix.index
        G.add_nodes_from(conditions)
        
        # Add edges for correlations above threshold
        for i in range(len(conditions)):
            for j in range(i+1, len(conditions)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    G.add_edge(conditions[i], conditions[j], weight=abs(corr))
        
        return G
        
    except Exception as e:
        logger.error(f"Error creating correlation network: {str(e)}")
        return nx.Graph()

def visualize_network_with_communities(G, domain_df):
    """
    Visualize the condition network with communities and domain coloring.
    
    Args:
        G: NetworkX graph of conditions
        domain_df: DataFrame containing domain information
        
    Returns:
        Plotly figure object
    """
    if G is None or G.number_of_nodes() == 0:
        return None
        
    try:
        # Set up domain colors
        domain_colors = {
            'Cardiometabolic': '#e41a1c',         # Red
            'Immune-Inflammation': '#377eb8',     # Blue
            'Oncological': '#4daf4a',             # Green
            'Neuro-Mental Health': '#984ea3',     # Purple
            'Neurological-Frailty': '#ff7f00',    # Orange
            'SDOH': '#ffff33',                    # Yellow
            'Other': '#999999'                    # Gray
        }
        
        # Position nodes using force-directed layout
        pos = None
        
        # Try to use nx.spring_layout but handle potential issues
        try:
            # Use spring layout with fixed parameters for better visualization
            pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
            logger.info("Using spring layout for network visualization")
        except Exception as e:
            logger.warning(f"Error with spring layout: {str(e)}")
            # Fall back to simpler layout
            pos = nx.random_layout(G)
            logger.info("Falling back to random layout")
        
        # Create edge traces with width based on weight
        edge_trace = []
        
        # Process edges and get the max weight for scaling
        max_weight = 1
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            max_weight = max(max_weight, weight)
            
        # Create edge traces
        for u, v, data in G.edges(data=True):
            # Get positions
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Scale weight for visual width (min 1, max 10)
            weight = data.get('weight', 1)
            scaled_width = 1 + (weight / max_weight) * 9
            
            # Create edge trace
            trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=scaled_width, color='rgba(180,180,180,0.7)'),
                hoverinfo='none'
            )
            edge_trace.append(trace)
        
        # Create node traces by domain for more efficient plotting and better coloring
        domain_node_traces = {}
        domain_counts = {}
        
        # Initialize a node trace for each domain
        for domain, color in domain_colors.items():
            domain_node_traces[domain] = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers',
                name=domain,
                marker=dict(
                    color=color,
                    size=15,
                    line=dict(width=1, color='#888')
                ),
                hoverinfo='text'
            )
            domain_counts[domain] = 0
        
        # Add nodes to their respective domain traces
        for node in G.nodes():
            x, y = pos[node]
            
            # Get node domain, degree, and weight for label
            domain = G.nodes[node].get('domain', 'Other')
            if domain not in domain_colors:
                domain = 'Other'
                
            # Count domains
            domain_counts[domain] += 1
            
            # Get node degree
            degree = G.degree(node)
            
            # Prepare hover text
            hover_text = f"Condition: {node}<br>Domain: {domain}<br>Connections: {degree}"
            
            # Add to the appropriate domain trace
            domain_node_traces[domain]['x'] = domain_node_traces[domain]['x'] + (x,)
            domain_node_traces[domain]['y'] = domain_node_traces[domain]['y'] + (y,)
            domain_node_traces[domain]['text'] = domain_node_traces[domain]['text'] + (hover_text,)
            # Scale node size by degree (connections)
            size = 10 + (degree * 2)
            if 'marker.size' not in domain_node_traces[domain]:
                domain_node_traces[domain]['marker']['size'] = []
            domain_node_traces[domain]['marker']['size'] = domain_node_traces[domain]['marker']['size'] + (size,)
        
        # Create a figure
        fig = go.Figure(data=edge_trace + list(domain_node_traces.values()))
        
        # Filter out empty domains for the legend
        active_domains = [domain for domain, count in domain_counts.items() if count > 0]
        
        # Create a better title with domain breakdown
        domain_breakdown = ", ".join([f"{domain}: {count}" for domain, count in domain_counts.items() if count > 0])
        title = f"Condition Network Map<br><sub>Domains: {domain_breakdown}</sub>"
        
        # Update layout
        fig.update_layout(
            title=title,
            titlefont=dict(size=16),
            showlegend=True,
            legend=dict(title="Clinical Domains"),
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error visualizing network: {str(e)}")
        return None

def get_condition_severity(condition: str, biomarkers: dict = None) -> str:
    """
    Determine condition severity based on ICD code and biomarkers.
    
    Args:
        condition: Condition or ICD-10 code
        biomarkers: Dictionary of biomarker values (optional)
        
    Returns:
        Severity level ('Low', 'Medium', or 'High')
    """
    try:
        # Default to medium severity
        severity = 'Medium'
        
        # Check condition keywords
        condition_lower = condition.lower()
        
        # High severity keywords
        high_severity = ['severe', 'acute', 'malignant', 'failure', 'critical', 'advanced',
                         'stage 3', 'stage 4', 'stage iii', 'stage iv', 'metastatic']
        if any(keyword in condition_lower for keyword in high_severity):
            severity = 'High'
            
        # Low severity keywords
        low_severity = ['mild', 'benign', 'routine', 'history of', 'controlled', 'remission',
                       'stage 0', 'stage 1', 'stage i', 'stage 0a', 'stage 0b']
        if any(keyword in condition_lower for keyword in low_severity):
            severity = 'Low'
            
        # Adjust based on biomarkers if available
        if biomarkers:
            # Check critical biomarker thresholds
            if biomarkers.get('glucose', 0) > 200:  # Severe hyperglycemia
                severity = 'High'
            if biomarkers.get('blood_pressure_systolic', 0) > 180:  # Severe hypertension
                severity = 'High'
            if biomarkers.get('ldl', 0) > 190:  # Very high LDL
                severity = 'High'
            if biomarkers.get('egfr', 0) < 30:  # Severe kidney disease
                severity = 'High'
            if biomarkers.get('hba1c', 0) > 9.0:  # Poor glycemic control
                severity = 'High'
            if biomarkers.get('crp', 0) > 10:  # High inflammation
                severity = 'High'
                
        return severity
        
    except Exception as e:
        logger.error(f"Error in get_condition_severity: {str(e)}")
        return 'Medium'

def calculate_total_risk_score(patient_data: dict) -> dict:
    """
    Calculate the total risk score for a patient based on their conditions, biomarkers, etc.
    
    Args:
        patient_data: Dictionary containing patient information including:
                     - conditions: list of conditions/diagnoses
                     - age: patient age
                     - gender: patient gender
                     - biomarkers: dictionary of biomarker values
                     - sdoh_data: dictionary of social determinants of health data
                     
    Returns:
        Dictionary containing:
        - total_score: total NHCRS score
        - domain_scores: dictionary of scores for each domain
        - mortality_risk_10yr: 10-year mortality risk percentage
        - hospitalization_risk_5yr: 5-year hospitalization risk percentage
    """
    try:
        # Extract required data
        conditions = patient_data.get('conditions', [])
        age = patient_data.get('age', 0)
        gender = patient_data.get('gender', 'Unknown')
        biomarkers = patient_data.get('biomarkers', {})
        sdoh_data = patient_data.get('sdoh_data', {})
        
        # Initialize default domain scores
        domain_scores = {
            'Cardiometabolic': 0.0,
            'Immune-Inflammation': 0.0,
            'Oncological': 0.0,
            'Neuro-Mental Health': 0.0,
            'Neurological-Frailty': 0.0,
            'SDOH': 0.0
        }
        
        # Get domain counts if available
        domain_condition_counts = patient_data.get('domain_condition_counts', {})
        
        # Skip risk calculation for patients with no conditions
        if not conditions and not biomarkers:
            # Return minimal risk scores
            return {
                'total_score': 1.0,  # Baseline score
                'domain_scores': domain_scores,
                'mortality_risk_10yr': 1.0,  # Baseline mortality risk
                'hospitalization_risk_5yr': 0.5  # Baseline hospitalization risk
            }
        
        # Calculate risk score for each domain
        total_conditions = len(conditions)
        
        # Calculate risk score for each domain using conditions
        if conditions:
            for domain in domain_scores.keys():
                # If we have precalculated domain counts, use them
                if domain in domain_condition_counts:
                    # Calculate proportion of conditions in this domain and scale to be more sensitive
                    domain_proportion = domain_condition_counts[domain] / max(1, total_conditions)
                    # Base domain risk on proportion and count
                    condition_count = domain_condition_counts[domain]
                    
                    # Scale the score to make it more sensitive
                    # Higher condition counts should have exponentially higher impact
                    score_factor = 1.0  # Default factor
                    if condition_count > 5:
                        score_factor = 1.5
                    elif condition_count > 10:
                        score_factor = 2.0
                    elif condition_count > 15:
                        score_factor = 2.5
                        
                    # Use a more sensitive formula with increased weight for higher counts
                    domain_score = (domain_proportion * 5.0) * math.sqrt(condition_count) * score_factor
                    domain_scores[domain] = min(10.0, domain_score)  # Cap at 10.0
                else:
                    # Legacy calculation - calculate domain-specific risk
                    domain_score = calculate_domain_risk_score(
                        conditions, domain, age, gender, biomarkers, sdoh_data
                    )
                    domain_scores[domain] = domain_score
        
        # Enhance scores with biomarker data if available
        if biomarkers:
            # Add biomarker contributions to each domain
            for domain in domain_scores.keys():
                biomarker_score = calculate_biomarker_component(biomarkers, domain)
                
                # Add the biomarker score, but ensure we don't exceed max
                domain_scores[domain] = min(10.0, domain_scores[domain] + biomarker_score)
        
        # Apply age and gender factors to each domain
        for domain in domain_scores.keys():
            gender_age_factor = get_gender_age_factor(domain, gender, age)
            
            # Apply the gender-age factor as a multiplier
            domain_scores[domain] = min(10.0, domain_scores[domain] * gender_age_factor)
            
        # Apply SDOH modifiers if available
        if sdoh_data:
            for domain in domain_scores.keys():
                sdoh_modifier = calculate_sdoh_modifier(sdoh_data, domain)
                
                # Apply the SDOH modifier
                domain_scores[domain] = min(10.0, domain_scores[domain] * sdoh_modifier)
        
        # Calculate final NHCRS total score with weighted domain contributions
        domain_weights = {
            'Cardiometabolic': 0.25,          # High impact on mortality and hospitalization
            'Immune-Inflammation': 0.15,      # Moderate impact
            'Oncological': 0.25,              # High impact on mortality
            'Neuro-Mental Health': 0.15,      # Moderate impact on hospitalization
            'Neurological-Frailty': 0.15,     # Moderate impact, high in elderly
            'SDOH': 0.05                      # Lower direct impact
        }
        
        # Apply domain weights to calculate total score
        weighted_score = 0.0
        for domain, score in domain_scores.items():
            weighted_score += score * domain_weights.get(domain, 0.0)
            
        # Add a network component if available
        network_metrics = patient_data.get('network_metrics', {})
        if network_metrics:
            degree_cent = network_metrics.get('degree_centrality', 0)
            betweenness_cent = network_metrics.get('betweenness_centrality', 0)
            
            # Add a network component to the score (0-2 points)
            network_score = (degree_cent * 1.0) + (betweenness_cent * 1.0)
            weighted_score += network_score
        
        # Get final NHCRS total (baseline + weighted contributions)
        total_score = 1.0 + weighted_score  # Baseline of 1.0
        
        # Calculate mortality and hospitalization risks based on total score
        mortality_risk = calculate_mortality_risk(total_score)
        hospitalization_risk = calculate_hospitalization_risk(total_score)
        
        # Return results
        return {
            'total_score': total_score,
            'domain_scores': domain_scores,
            'mortality_risk_10yr': mortality_risk,
            'hospitalization_risk_5yr': hospitalization_risk
        }
    
    except Exception as e:
        logger.error(f"Error calculating risk score: {str(e)}")
        return {
            'total_score': 1.0,
            'domain_scores': {'Other': 1.0},
            'mortality_risk_10yr': 1.0,
            'hospitalization_risk_5yr': 0.5
        }

def calculate_domain_risk_score(conditions: list, domain: str, patient_age: int, patient_gender: str, biomarkers: dict = None, sdoh_data: dict = None) -> float:
    """
    Calculate risk score for a specific clinical domain using the enhanced NHCRS model.
    
    Args:
        conditions: List of conditions in this domain
        domain: Domain name (cardiometabolic, immune_inflammation, etc.)
        patient_age: Patient's age
        patient_gender: Patient's gender ('M' or 'F')
        biomarkers: Dict of biomarker values (optional)
        sdoh_data: Dict of social determinants of health data (optional)
        
    Returns:
        Domain-specific risk score
    """
    try:
        # Initialize base score
        base_score = 1.0
        
        # Calculate condition severity component
        condition_score = 0
        for condition in conditions:
            severity = get_condition_severity(condition, biomarkers)
            # Convert severity to numeric value
            severity_value = {'Low': 0.5, 'Medium': 1.0, 'High': 1.5}.get(severity, 1.0)
            # Apply condition-specific weight
            beta_i = get_condition_weight(condition, domain)
            condition_score += beta_i * severity_value
        
        # Calculate biomarker component if available
        biomarker_score = 0
        if biomarkers:
            biomarker_score = calculate_biomarker_component(biomarkers, domain)
        
        # Calculate SDOH component if available
        sdoh_factor = 1.0
        if sdoh_data:
            sdoh_factor = calculate_sdoh_modifier(sdoh_data, domain)
        
        # Get gender and age adjustment factor
        gaf = get_gender_age_factor(domain, patient_gender, patient_age)
        
        # Calculate total domain score using the formula:
        # R_d = (sum(β_i * S_i) + sum(γ_j * B_j)) * GAF * SDOH_factor
        domain_score = (condition_score + biomarker_score) * gaf * sdoh_factor
        
        # Cap at maximum value of 10 per domain
        return min(domain_score, 10.0)
        
    except Exception as e:
        logger.error(f"Error in calculate_domain_risk_score: {str(e)}")
        return 1.0

def get_condition_weight(condition: str, domain: str) -> float:
    """
    Get condition-specific weight (β_i) based on clinical significance.
    These values would ideally be derived from published hazard ratios or odds ratios.
    """
    # Default weight
    default_weight = 1.0
    
    # High-impact conditions by domain
    high_impact_conditions = {
        'cardiometabolic': ['heart failure', 'stroke', 'myocardial infarction', 'coronary', 'diabetes', 'i21', 'i22', 'i50'],
        'immune_inflammation': ['sepsis', 'severe infection', 'pneumonia', 'covid', 'rheumatoid arthritis'],
        'oncologic': ['malignant', 'metastatic', 'cancer'],
        'neuro_mental_health': ['suicidal', 'psychosis', 'schizophrenia', 'severe depression'],
        'neurological_frailty': ['alzheimer', 'parkinson', 'multiple sclerosis', 'dementia']
    }
    
    # Medium-impact conditions by domain
    medium_impact_conditions = {
        'cardiometabolic': ['hypertension', 'hyperlipidemia', 'obesity', 'atrial fibrillation'],
        'immune_inflammation': ['asthma', 'chronic bronchitis', 'inflammatory bowel'],
        'oncologic': ['benign tumor', 'neoplasm'],
        'neuro_mental_health': ['anxiety', 'depression', 'bipolar', 'substance abuse'],
        'neurological_frailty': ['seizure', 'neuropathy', 'tremor']
    }
    
    # Convert condition to lowercase for matching
    condition_lower = condition.lower()
    
    # Check if this is a high-impact condition
    if any(keyword in condition_lower for keyword in high_impact_conditions.get(domain, [])):
        return 2.0
    
    # Check if this is a medium-impact condition
    if any(keyword in condition_lower for keyword in medium_impact_conditions.get(domain, [])):
        return 1.5
    
    return default_weight

def calculate_biomarker_component(biomarkers: dict, domain: str) -> float:
    """
    Calculate domain-specific biomarker component.
    
    Args:
        biomarkers: Dictionary of biomarker values
        domain: Clinical domain
        
    Returns:
        Biomarker component score for this domain
    """
    score = 0.0
    
    # Define domain-specific biomarkers and their weights (γ_j)
    domain_biomarkers = {
        'cardiometabolic': {
            'glucose': {'weight': 0.5, 'threshold': 126, 'high_threshold': 200},
            'cholesterol': {'weight': 0.3, 'threshold': 200, 'high_threshold': 240},
            'ldl': {'weight': 0.4, 'threshold': 130, 'high_threshold': 160},
            'hdl': {'weight': 0.3, 'threshold': 40, 'high_threshold': 30, 'inverse': True},
            'triglycerides': {'weight': 0.3, 'threshold': 150, 'high_threshold': 200},
            'blood_pressure_systolic': {'weight': 0.5, 'threshold': 140, 'high_threshold': 160},
            'blood_pressure_diastolic': {'weight': 0.3, 'threshold': 90, 'high_threshold': 100},
            'bmi': {'weight': 0.3, 'threshold': 30, 'high_threshold': 35},
            'hba1c': {'weight': 0.6, 'threshold': 6.5, 'high_threshold': 8.0}
        },
        'immune_inflammation': {
            'crp': {'weight': 0.6, 'threshold': 3, 'high_threshold': 10},
            'esr': {'weight': 0.4, 'threshold': 20, 'high_threshold': 50},
            'wbc': {'weight': 0.4, 'threshold': 11, 'high_threshold': 15},
            'neutrophils': {'weight': 0.3, 'threshold': 7.5, 'high_threshold': 10},
            'lymphocytes': {'weight': 0.2, 'threshold': 4.5, 'high_threshold': 6}
        },
        'oncologic': {
            'cea': {'weight': 0.5, 'threshold': 3, 'high_threshold': 10},
            'psa': {'weight': 0.5, 'threshold': 4, 'high_threshold': 10},
            'ca125': {'weight': 0.5, 'threshold': 35, 'high_threshold': 100},
            'afp': {'weight': 0.5, 'threshold': 10, 'high_threshold': 50}
        },
        'neuro_mental_health': {
            'cortisol': {'weight': 0.4, 'threshold': 20, 'high_threshold': 30}
        },
        'neurological_frailty': {
            'vitamin_d': {'weight': 0.3, 'threshold': 20, 'high_threshold': 12, 'inverse': True},
            'vitamin_b12': {'weight': 0.3, 'threshold': 200, 'high_threshold': 150, 'inverse': True},
            'albumin': {'weight': 0.4, 'threshold': 3.5, 'high_threshold': 3.0, 'inverse': True}
        }
    }
    
    # Calculate score for each relevant biomarker
    for biomarker, value in biomarkers.items():
        # Skip if biomarker not relevant to this domain or value is missing
        if (biomarker not in domain_biomarkers.get(domain, {}) or 
            pd.isna(value) or 
            not isinstance(value, (int, float))):
            continue
            
        params = domain_biomarkers[domain][biomarker]
        weight = params['weight']
        threshold = params['threshold']
        high_threshold = params['high_threshold']
        inverse = params.get('inverse', False)
        
        # Calculate normalized score (0 to 1 scale)
        if inverse:
            # For inverse biomarkers (lower is worse)
            if value <= high_threshold:
                normalized_score = 1.0  # Highest risk
            elif value <= threshold:
                normalized_score = 0.5  # Moderate risk
            else:
                normalized_score = 0.0  # Normal
        else:
            # For regular biomarkers (higher is worse)
            if value >= high_threshold:
                normalized_score = 1.0  # Highest risk
            elif value >= threshold:
                normalized_score = 0.5  # Moderate risk
            else:
                normalized_score = 0.0  # Normal
                
        # Add weighted biomarker score
        score += weight * normalized_score
    
    return score

def get_gender_age_factor(domain: str, gender: str, age: int) -> float:
    """
    Calculate gender and age adjustment factor for a specific domain.
    
    Args:
        domain: Clinical domain
        gender: Patient gender ('M' or 'F')
        age: Patient age
        
    Returns:
        Adjustment factor (multiplicative)
    """
    # Default factor
    factor = 1.0
    
    # Normalize gender input
    gender = str(gender).upper()[:1]  # Take first letter, uppercase
    
    # Domain-specific gender/age adjustments
    if domain == 'cardiometabolic':
        # Men have higher baseline cardiovascular risk before age 55
        if gender == 'M' and age < 55:
            factor = 1.3
        # Women's risk increases post-menopause
        elif gender == 'F' and age >= 55:
            factor = 1.2
            
    elif domain == 'oncologic':
        # Women under 50 have higher cancer risk (e.g., breast, cervical)
        if gender == 'F' and age < 50:
            factor = 1.8
        # Men over 50 have higher cancer risk (e.g., prostate, colorectal)
        elif gender == 'M' and age >= 50:
            factor = 1.5
            
    elif domain == 'neuro_mental_health':
        # Women have higher rates of depression and anxiety
        if gender == 'F':
            factor = 1.4
            
    elif domain == 'neurological_frailty':
        # Risk increases with age
        if age >= 80:
            factor = 2.0
        elif age >= 70:
            factor = 1.5
        elif age >= 60:
            factor = 1.2
    
    return factor

def calculate_sdoh_modifier(sdoh_data: dict, domain: str) -> float:
    """
    Calculate SDOH (Social Determinants of Health) modifier for a specific domain.
    
    Args:
        sdoh_data: Dictionary of SDOH indicators
        domain: Clinical domain
        
    Returns:
        SDOH modifier (multiplicative factor)
    """
    # Default modifier (no impact)
    modifier = 1.0
    
    if not sdoh_data:
        return modifier
        
    # SDOH risk factors with domain-specific weights
    sdoh_weights = {
        'cardiometabolic': {
            'medication_adherence': 0.3, 
            'diet_quality': 0.25,
            'physical_activity': 0.25,
            'socioeconomic_status': 0.1,
            'housing_stability': 0.1
        },
        'immune_inflammation': {
            'medication_adherence': 0.3,
            'housing_stability': 0.2,
            'socioeconomic_status': 0.2,
            'healthcare_access': 0.3
        },
        'oncologic': {
            'healthcare_access': 0.3,
            'socioeconomic_status': 0.2,
            'medication_adherence': 0.3,
            'social_support': 0.2
        },
        'neuro_mental_health': {
            'social_support': 0.3,
            'socioeconomic_status': 0.2,
            'housing_stability': 0.2,
            'healthcare_access': 0.15,
            'medication_adherence': 0.15
        },
        'neurological_frailty': {
            'social_support': 0.25,
            'housing_stability': 0.25,
            'socioeconomic_status': 0.2,
            'healthcare_access': 0.15,
            'medication_adherence': 0.15
        }
    }
    
    # Get weights for this domain
    domain_weights = sdoh_weights.get(domain, {})
    
    # Calculate total SDOH impact
    sdoh_impact = 0
    for factor, weight in domain_weights.items():
        # SDOH factors are scaled 0-1 where 0 is best and 1 is worst
        if factor in sdoh_data:
            sdoh_impact += weight * sdoh_data[factor]
    
    # Convert to modifier (max increase of 50% for worst SDOH)
    modifier = 1.0 + sdoh_impact
    
    return modifier

def calculate_decay(initial_value: float, time_months: float, decay_type: str = 'linear', 
                   decay_params: dict = None) -> float:
    """
    Calculate decay in risk over time, e.g., after intervention or treatment.
    
    Args:
        initial_value: Initial risk score value
        time_months: Time elapsed since intervention (in months)
        decay_type: Type of decay function ('linear', 'exponential', or 'threshold')
        decay_params: Parameters for the decay function
        
    Returns:
        Adjusted value after decay
    """
    if decay_params is None:
        decay_params = {}
    
    # Set default parameters if not provided
    if decay_type == 'linear':
        # Linear decay: D(t) = max(0, 1 - k * t)
        k = decay_params.get('k', 0.05)  # Default 5% decrease per month
        decay_factor = max(0, 1 - k * time_months)
        
    elif decay_type == 'exponential':
        # Exponential decay: D(t) = exp(-λ * t)
        lambda_param = decay_params.get('lambda', 0.1)  # Default decay rate
        decay_factor = math.exp(-lambda_param * time_months)
        
    elif decay_type == 'threshold':
        # Threshold-based decay with steps
        thresholds = decay_params.get('thresholds', [3, 6, 12])  # Months
        factors = decay_params.get('factors', [0.8, 0.5, 0.3])  # Corresponding factors
        
        # Find the appropriate threshold
        decay_factor = 1.0
        for i, threshold in enumerate(thresholds):
            if time_months >= threshold and i < len(factors):
                decay_factor = factors[i]
                
    else:
        # Default: no decay
        decay_factor = 1.0
    
    # Apply decay factor to initial value
    return initial_value * decay_factor

def calculate_nhcrs_total(domain_scores: dict, baseline_intercept: float = 1.0) -> float:
    """
    Calculate Nudge Health Clinical Risk Score (NHCRS) from domain scores.
    
    Args:
        domain_scores: Dictionary of domain-specific risk scores
        baseline_intercept: Baseline intercept value (α)
        
    Returns:
        Total NHCRS score
    """
    # Domain scaling factors (λ_d) - can be adjusted based on validation
    domain_weights = {
        'cardiometabolic': 1.0,
        'immune_inflammation': 0.8,
        'oncologic': 1.2,
        'neuro_mental_health': 0.7,
        'neurological_frailty': 0.9,
        'other': 0.5
    }
    
    # Calculate weighted sum of domain scores
    weighted_sum = 0
    for domain, score in domain_scores.items():
        domain_weight = domain_weights.get(domain, 0.5)  # Default weight for unknown domains
        weighted_sum += domain_weight * score
    
    # Add baseline intercept
    total_score = baseline_intercept + weighted_sum
    
    return total_score

def calculate_mortality_risk(nhcrs_total: float) -> float:
    """
    Convert NHCRS to 10-year mortality risk probability using logistic function.
    
    Args:
        nhcrs_total: Total Nudge Health Clinical Risk Score
        
    Returns:
        10-year mortality risk as a probability (0-1)
    """
    # Logistic regression parameters (would be calibrated on outcome data)
    a = -5.0  # Intercept (negative means low baseline risk)
    b = 0.3   # Coefficient for NHCRS
    
    # Logistic function: P = 1 / (1 + exp(-(a + b*NHCRS)))
    z = a + b * nhcrs_total
    probability = 1 / (1 + math.exp(-z))
    
    return probability

def calculate_hospitalization_risk(nhcrs_total: float) -> float:
    """
    Convert NHCRS to 5-year hospitalization risk probability using logistic function.
    
    Args:
        nhcrs_total: Total Nudge Health Clinical Risk Score
        
    Returns:
        5-year hospitalization risk as a probability (0-1)
    """
    # Logistic regression parameters (would be calibrated on outcome data)
    c = -3.5  # Intercept (hospitalization more common than mortality)
    d = 0.25  # Coefficient for NHCRS
    
    # Logistic function: P = 1 / (1 + exp(-(c + d*NHCRS)))
    z = c + d * nhcrs_total
    probability = 1 / (1 + math.exp(-z))
    
    return probability

def process_biomarker_data(df):
    """
    Process biomarker data from various file formats.
    
    Args:
        df: DataFrame containing biomarker data
        
    Returns:
        DataFrame with standardized biomarker columns
    """
    try:
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Standardize column names (case insensitive)
        df_copy.columns = [col.lower() for col in df_copy.columns]
        
        # Map common column variations to standard names
        column_mappings = {
            'pat id': 'patid', 
            'patient id': 'patid',
            'patientid': 'patid',
            'sex': 'gender',
            'age_years': 'age'
        }
        
        # Apply column mappings where applicable
        for old_col, new_col in column_mappings.items():
            if old_col in df_copy.columns and new_col not in df_copy.columns:
                df_copy.rename(columns={old_col: new_col}, inplace=True)
        
        # Function to find the best matching column for a target
        def find_best_column_match(target, possible_names):
            # First try exact matches
            if target in df_copy.columns:
                return target
            
            # Then try partial matches
            for col in df_copy.columns:
                if any(name in col.lower() for name in possible_names):
                    return col
            
            return None
        
        # Create patient_id column if it doesn't exist
        if 'patient_id' not in df_copy.columns:
            if 'patid' in df_copy.columns:
                df_copy['patient_id'] = df_copy['patid'].astype(str)
            else:
                st.error("No patient identifier column found")
                return pd.DataFrame()
        
        # Identify potential biomarker columns
        exclude_cols = ['patient_id', 'patid', 'gender', 'age', 'zip_code']
        biomarker_cols = [col for col in df_copy.columns if col not in exclude_cols]
        
        # Map recognized biomarker names to standardized names
        biomarker_mappings = {
            # Lipid panel
            'ldl': 'ldl_cholesterol',
            'ldl cholesterol': 'ldl_cholesterol',
            'ldl-c': 'ldl_cholesterol',
            'hdl': 'hdl_cholesterol',
            'hdl cholesterol': 'hdl_cholesterol',
            'hdl-c': 'hdl_cholesterol',
            'total cholesterol': 'total_cholesterol',
            'cholesterol': 'total_cholesterol',
            'triglycerides': 'triglycerides',
            'tg': 'triglycerides',
            
            # Diabetes markers
            'glucose': 'glucose',
            'fasting glucose': 'fasting_glucose',
            'a1c': 'hba1c',
            'hba1c': 'hba1c',
            'hemoglobin a1c': 'hba1c',
            
            # Inflammation markers
            'crp': 'crp',
            'c-reactive protein': 'crp',
            'hs-crp': 'hs_crp',
            'high-sensitivity crp': 'hs_crp',
            'esr': 'esr',
            'erythrocyte sedimentation rate': 'esr',
            
            # Cardiac markers
            'troponin': 'troponin',
            'bnp': 'bnp',
            'nt-probnp': 'nt_probnp',
            'brain natriuretic peptide': 'bnp',
            
            # Kidney function
            'creatinine': 'creatinine',
            'egfr': 'egfr',
            'estimated gfr': 'egfr',
            'bun': 'bun',
            'blood urea nitrogen': 'bun',
            'albumin': 'albumin',
            'microalbumin': 'microalbumin',
            
            # Liver function
            'alt': 'alt',
            'alanine aminotransferase': 'alt',
            'ast': 'ast',
            'aspartate aminotransferase': 'ast',
            'alp': 'alp',
            'alkaline phosphatase': 'alp',
            'ggt': 'ggt',
            'gamma-glutamyl transferase': 'ggt',
            'bilirubin': 'bilirubin',
            
            # Blood count
            'wbc': 'wbc',
            'white blood cells': 'wbc',
            'rbc': 'rbc',
            'red blood cells': 'rbc',
            'hemoglobin': 'hemoglobin',
            'hb': 'hemoglobin',
            'hematocrit': 'hematocrit',
            'hct': 'hematocrit',
            'platelets': 'platelets',
            'plt': 'platelets',
            
            # Vitamins and minerals
            'vitamin d': 'vitamin_d',
            '25-oh vitamin d': 'vitamin_d',
            'vitamin b12': 'vitamin_b12',
            'folate': 'folate',
            'iron': 'iron',
            'ferritin': 'ferritin',
            
            # Thyroid function
            'tsh': 'tsh',
            'thyroid stimulating hormone': 'tsh',
            'ft4': 'free_t4',
            'free t4': 'free_t4',
            'ft3': 'free_t3',
            'free t3': 'free_t3'
        }
        
        # Standardize biomarker column names
        renamed_cols = {}
        for col in biomarker_cols:
            col_clean = col.strip().lower().replace(' ', '_')
            if col_clean in biomarker_mappings:
                renamed_cols[col] = biomarker_mappings[col_clean]
            elif col.lower() in biomarker_mappings:
                renamed_cols[col] = biomarker_mappings[col.lower()]
        
        # Apply the standardized names
        if renamed_cols:
            df_copy.rename(columns=renamed_cols, inplace=True)
        
        # Convert all biomarker values to numeric
        biomarker_cols = [col for col in df_copy.columns if col not in exclude_cols]
        for col in biomarker_cols:
            try:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            except:
                logger.warning(f"Could not convert column {col} to numeric")
        
        return df_copy
    
    except Exception as e:
        st.error(f"Error processing biomarker data: {str(e)}")
        logger.error(f"Error in process_biomarker_data: {str(e)}")
        return pd.DataFrame()

def perform_integrated_analysis(icd_df, biomarker_df=None):
    """
    Unified analysis framework that integrates network analysis and risk score calculations.
    This optimized function combines both analyses to improve performance, following
    the explicit workflow:
    
    1. Load & preprocess patient data (Excel/EHR FHIR)
    2. Map ICD codes to domains explicitly
    3. Calculate domain scores with gender/age adjustments
    4. Calculate NHCRS total score
    5. Calculate dynamic probabilities (hospitalization/mortality risk)
    6. Perform network clustering
    7. Generate clinical recommendations (if OpenAI API key available)
    8. Generate structured PDF reports
    
    Args:
        icd_df: DataFrame containing ICD diagnosis data
        biomarker_df: Optional DataFrame containing biomarker data
        
    Returns:
        Dictionary containing all analysis results
    """
    try:
        start_time = time.time()
        st.info("Starting integrated analysis...")
        progress_bar = st.progress(0)
        
        # 1. PREPROCESS DATA
        # -------------------
        # Validate input data
        if icd_df is None and biomarker_df is None:
            st.error("No valid data provided for analysis")
            return None
        
        # Process the diagnosis data if available
        domain_df = None
        if icd_df is not None and not icd_df.empty:
            # Process into domains if not already done
            if 'domain' not in icd_df.columns:
                # Get diagnosis columns
                diag_cols = [col for col in icd_df.columns if 'diagnosis' in col.lower() or 'condition' in col.lower()]
                # Map ICD codes to domains explicitly
                domain_df = process_domain_data(icd_df, diag_cols)
            else:
                domain_df = icd_df.copy()
            
            progress_bar.progress(15)
        
        # 2. NETWORK ANALYSIS
        # ------------------
        # Network initialization
        G = None
        degree_cent = {}
        betweenness_cent = {}
        communities = {0: []}
        network_metrics = {}
        node_centrality = {}
        
        if domain_df is not None and not domain_df.empty:
            # Create network for centrality analysis
            G = create_domain_network(domain_df)
            progress_bar.progress(25)
            
            # Calculate network metrics
            if G.number_of_nodes() > 0:
                # Calculate centrality metrics
                degree_cent = nx.degree_centrality(G)
                betweenness_cent = nx.betweenness_centrality(G)
                
                # Store network-level metrics
                network_metrics = {
                    'node_count': G.number_of_nodes(),
                    'edge_count': G.number_of_edges(),
                    'density': nx.density(G),
                    'avg_clustering': nx.average_clustering(G) if G.number_of_nodes() > 2 else 0
                }
                
                # 6. Network clustering explicitly (Louvain method)
                # ------------------------------------------------
                try:
                    if COMMUNITY_DETECTION_AVAILABLE:
                        partition = community_louvain.best_partition(G)
                        communities = defaultdict(list)
                        for node, community_id in partition.items():
                            communities[community_id].append(node)
                        network_metrics['communities'] = len(communities)
                        network_metrics['modularity'] = community_louvain.modularity(partition, G)
                    else:
                        communities = {0: list(G.nodes())}
                        network_metrics['communities'] = 1
                        network_metrics['modularity'] = 0
                        st.warning("Community detection library not available. Install python-louvain for better analysis.")
                except Exception as e:
                    logger.error(f"Error in community detection: {str(e)}")
                    communities = {0: list(G.nodes())}
                    network_metrics['communities'] = 1
                    network_metrics['modularity'] = 0
                
                # Create node metrics dictionary for easy access
                for node in G.nodes():
                    node_centrality[node] = {
                        'degree_centrality': degree_cent.get(node, 0),
                        'betweenness_centrality': betweenness_cent.get(node, 0),
                        'domain': assign_clinical_domain(node)
                    }
            
            progress_bar.progress(40)
        
        # 3. RISK SCORE CALCULATION
        # ------------------------
        # Get patient IDs from available data
        patient_ids = []
        if domain_df is not None and not domain_df.empty:
            patient_ids.extend(domain_df['patient_id'].unique())
        if biomarker_df is not None and not biomarker_df.empty:
            patient_ids.extend(biomarker_df['patient_id'].unique())
        
        # Remove duplicates and ensure we have patients to analyze
        patient_ids = list(set(patient_ids))
        if not patient_ids:
            st.error("No valid patient IDs found in the data")
            return None
        
        # Prepare results containers
        patient_results = {}
        risk_scores = {}
        domain_scores = {}
        mortality_risks = {}
        hospitalization_risks = {}
        
        # Track progress for patient calculations
        total_patients = len(patient_ids)
        current_progress = 40
        
        # Process each patient
        for i, patient_id in enumerate(patient_ids):
            # Update progress bar
            progress_value = current_progress + int(50 * (i / total_patients))
            progress_bar.progress(progress_value)
            
            # Get patient conditions if ICD data available
            patient_conditions = []
            patient_demographics = {'age': 0, 'gender': 'Unknown'}
            
            if domain_df is not None and not domain_df.empty:
                patient_rows = domain_df[domain_df['patient_id'] == patient_id]
                if not patient_rows.empty:
                    patient_conditions = patient_rows['condition'].tolist()
                    # Get demographics from the first row
                    first_row = patient_rows.iloc[0]
                    patient_demographics = {
                        'age': first_row.get('age', 0),
                        'gender': first_row.get('gender', 'Unknown')
                    }
            
            # Extract biomarker values if available
            biomarkers = {}
            if biomarker_df is not None and not biomarker_df.empty:
                patient_bio_rows = biomarker_df[biomarker_df['patient_id'] == patient_id]
                if not patient_bio_rows.empty:
                    patient_bio_row = patient_bio_rows.iloc[0]
                    for col in biomarker_df.columns:
                        if col != 'patient_id' and not pd.isna(patient_bio_row[col]):
                            try:
                                biomarkers[col] = float(patient_bio_row[col])
                            except:
                                pass
            
            # Calculate network-based metrics for this patient
            network_metrics_patient = {
                'degree_centrality': 0,
                'betweenness_centrality': 0
            }
            
            if G is not None and G.number_of_nodes() > 0:
                # Average the centrality of all patient conditions
                valid_conditions = [c for c in patient_conditions if c in G.nodes()]
                if valid_conditions:
                    network_metrics_patient['degree_centrality'] = np.mean([
                        degree_cent.get(c, 0) for c in valid_conditions
                    ])
                    network_metrics_patient['betweenness_centrality'] = np.mean([
                        betweenness_cent.get(c, 0) for c in valid_conditions
                    ])
            
            # 2. Map ICD codes to domains explicitly
            # -------------------------------------
            # Count conditions by domain for this patient
            domain_condition_counts = {}
            if patient_conditions:
                # Group conditions by domain
                for condition in patient_conditions:
                    domain = assign_clinical_domain(condition)
                    if domain not in domain_condition_counts:
                        domain_condition_counts[domain] = 0
                    domain_condition_counts[domain] += 1
            
            # Prepare patient data for risk calculation
            patient_info = {
                'patient_id': patient_id,
                'conditions': patient_conditions,
                'age': patient_demographics.get('age', 0),
                'gender': patient_demographics.get('gender', 'Unknown'),
                'network_metrics': network_metrics_patient,
                'biomarkers': biomarkers,
                'sdoh_data': {},  # Placeholder for SDOH data
                'domain_condition_counts': domain_condition_counts
            }
            
            # 3 & 4. Calculate domain scores and NHCRS total with gender/age adjustments
            # -------------------------------------------------------------------------
            risk_result = calculate_total_risk_score(patient_info)
            
            # Store results
            patient_results[patient_id] = patient_info
            risk_scores[patient_id] = risk_result.get('total_score', 0)
            domain_scores[patient_id] = risk_result.get('domain_scores', {})
            
            # 5. Calculate dynamic probabilities (hospitalization/mortality risk)
            # -----------------------------------------------------------------
            mortality_risks[patient_id] = risk_result.get('mortality_risk_10yr', 0)
            hospitalization_risks[patient_id] = risk_result.get('hospitalization_risk_5yr', 0)
        
        progress_bar.progress(90)
        
        # Calculate domain distribution across all patients
        all_domains = []
        for patient_id in domain_scores:
            all_domains.extend(domain_scores[patient_id].keys())
        domain_counts = dict(Counter(all_domains))
        
        # Prepare combined patient dataframe
        combined_rows = []
        for patient_id in patient_results:
            patient_data = patient_results[patient_id]
            row = {
                'patient_id': patient_id,
                'age': patient_data.get('age', 0),
                'gender': patient_data.get('gender', 'Unknown'),
                'conditions': patient_data.get('conditions', []),
                'condition_count': len(patient_data.get('conditions', [])),
                'domain_scores': domain_scores.get(patient_id, {}),
                'total_score': risk_scores.get(patient_id, 0),
                'mortality_risk_10yr': mortality_risks.get(patient_id, 0),
                'hospitalization_risk_5yr': hospitalization_risks.get(patient_id, 0),
                'degree_centrality': patient_data.get('network_metrics', {}).get('degree_centrality', 0),
                'betweenness_centrality': patient_data.get('network_metrics', {}).get('betweenness_centrality', 0)
            }
            combined_rows.append(row)
        
        combined_df = pd.DataFrame(combined_rows)
        
        # Complete progress
        progress_bar.progress(100)
        
        # Log performance
        end_time = time.time()
        analysis_time = end_time - start_time
        logger.info(f"Integrated analysis completed in {analysis_time:.2f} seconds")
        
        # Return combined results
        return {
            'G': G,
            'domain_df': domain_df,
            'combined_df': combined_df,
            'network_metrics': network_metrics,
            'patient_results': patient_results,
            'risk_scores': risk_scores,
            'domain_scores': domain_scores,
            'mortality_risks': mortality_risks,
            'hospitalization_risks': hospitalization_risks,
            'domain_counts': domain_counts,
            'communities': communities if 'communities' in locals() else {0: list(G.nodes()) if G else []},
            'node_centrality': node_centrality
        }
            
    except Exception as e:
        logger.error(f"Error in integrated analysis: {str(e)}")
        st.error(f"Error in analysis: {str(e)}")
        return None

def main():
    # Set up the sidebar
    st.sidebar.title("Nudge Health AI Clinical Analyzer")
    
    # Add logo or image placeholder
    st.sidebar.image("https://via.placeholder.com/150x150?text=Nudge+Health", width=150)
    
    # Fetch the latest reference data
    with st.spinner("Updating reference data..."):
        icd_codes = fetch_authoritative_codes()
        biomarker_refs = fetch_biomarker_reference_data()
        st.session_state.icd_codes = icd_codes
        st.session_state.biomarker_refs = biomarker_refs
    
    # OpenAI API key handling
    try:
        # Check if running on Streamlit Cloud (where secrets are configured)
        is_cloud = os.environ.get('STREAMLIT_SHARING', '') == 'true' or os.environ.get('IS_STREAMLIT_CLOUD', '') == 'true'
        
        # Try to get the API key from secrets
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        
        if api_key and api_key != "your-api-key-here":
            st.session_state.openai_api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
            st.sidebar.success("OpenAI API key configured")
        elif not is_cloud:  # Only show input field if not on Streamlit Cloud
            # Only show input field if no API key in secrets and not on cloud
            st.sidebar.subheader("OpenAI Settings (Optional)")
            api_key = st.sidebar.text_input("OpenAI API Key (for AI recommendations)", 
                                          type="password", 
                                          help="Enter your OpenAI API key to enable AI clinical recommendations")
            if api_key:
                st.session_state.openai_api_key = api_key
                os.environ["OPENAI_API_KEY"] = api_key
                st.sidebar.success("OpenAI API key configured")
            else:
                st.sidebar.info("No OpenAI API key found. AI recommendations will be disabled.")
    except Exception as e:
        # Only show the input field if we're not on Streamlit Cloud
        is_cloud = os.environ.get('STREAMLIT_SHARING', '') == 'true' or os.environ.get('IS_STREAMLIT_CLOUD', '') == 'true'
        if not is_cloud:
            st.sidebar.subheader("OpenAI Settings (Optional)")
            api_key = st.sidebar.text_input("OpenAI API Key (for AI recommendations)", 
                                          type="password", 
                                          help="Enter your OpenAI API key to enable AI clinical recommendations")
            if api_key:
                st.session_state.openai_api_key = api_key
                os.environ["OPENAI_API_KEY"] = api_key
                st.sidebar.success("OpenAI API key configured")
            else:
                st.sidebar.info("No OpenAI API key found. AI recommendations will be disabled.")
    
    st.sidebar.divider()
    
    # Select authentication type
    auth_type = st.sidebar.radio(
        "Select Authentication Type:",
        ["Local Upload", "FHIR Integration"],
        key="auth_type_radio"
    )
    
    # Remove the separate analysis detail selection
    # Instead just keep the view type selection
    view_type = st.sidebar.radio(
        "Select View Type:",
        ["Population Analysis", "Single Patient Analysis"],
        key="view_type_radio"
    )
    
    # Store view type in session state for later use
    st.session_state.view_type = view_type
    
    # Main content area
    st.title("Clinical Data Analysis Platform")
    
    # Local file upload
    if auth_type == "Local Upload":
        # Create two columns for file uploads
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Comorbidity Data")
            icd_file = st.file_uploader("Upload ICD-10 Excel file", 
                                      type=["xlsx", "xls"],
                                      key="icd_upload",
                                      help="Excel file with columns: PatId, Gender, Age, and Diagnosis columns")
        
        with col2:
            st.subheader("Upload Biomarker Data")
            bio_file = st.file_uploader("Upload Biomarker Excel file",
                                      type=["xlsx", "xls"],
                                      key="bio_upload",
                                      help="Excel file with biomarker measurements")
        
        # Process the uploaded files
        icd_df = None
        bio_df = None
        
        # Process ICD-10 data if present
        if icd_file is not None:
            try:
                df = pd.read_excel(icd_file)
                
                # Fix column names - support PatId and Pat Id variations
                if 'Pat Id' in df.columns and 'PatId' not in df.columns:
                    df.rename(columns={'Pat Id': 'PatId'}, inplace=True)
                
                icd_df = process_diagnosis_data(df)
                if icd_df is not None and not icd_df.empty:
                    st.success(f"Successfully processed {len(icd_df)} ICD-10 records for {len(icd_df['patient_id'].unique())} patients.")
                    
                    # Show ICD data summary
                    with st.expander("View ICD-10 Data Summary"):
                        st.write({
                            "Total Patients": len(icd_df['patient_id'].unique()),
                            "Total Diagnoses": len(icd_df),
                            "Domains Found": icd_df['domain'].unique().tolist() if 'domain' in icd_df.columns else []
                        })
            except Exception as e:
                st.error(f"Error processing ICD-10 data: {str(e)}")
        
        # Process biomarker data if present
        if bio_file is not None:
            try:
                df = pd.read_excel(bio_file)
                
                # Fix column names - support PatId and Pat Id variations
                if 'Pat Id' in df.columns and 'PatId' not in df.columns:
                    df.rename(columns={'Pat Id': 'PatId'}, inplace=True)
                
                bio_df = process_biomarker_data(df)
                if bio_df is not None and not bio_df.empty:
                    st.success(f"Successfully processed biomarker data for {bio_df['patient_id'].nunique()} patients.")
                    
                    # Show biomarker data summary
                    with st.expander("View Biomarker Data Summary"):
                        st.write({
                            "Total Patients": bio_df['patient_id'].nunique(),
                            "Biomarkers Available": [col for col in bio_df.columns if col != 'patient_id']
                        })
            except Exception as e:
                st.error(f"Error processing biomarker data: {str(e)}")
        
        # Perform analysis based on available data
        if icd_df is not None or bio_df is not None:
            with st.spinner("Performing analysis... This may take a moment."):
                if icd_df is not None and bio_df is not None:
                    st.info("Performing combined analysis with both comorbidity and biomarker data...")
                    analysis_results = perform_integrated_analysis(icd_df, bio_df)
                elif icd_df is not None:
                    st.info("Performing comorbidity analysis...")
                    analysis_results = perform_integrated_analysis(icd_df, None)
                else:
                    st.info("Performing biomarker analysis...")
                    analysis_results = perform_integrated_analysis(None, bio_df)
                
                if analysis_results:
                    st.success("Analysis completed successfully!")
                    
                    # Get the view type from session state
                    view_type = st.session_state.get('view_type', 'Population Analysis')
                    
                    # Use tabs for better organization of different analysis components
                    tabs = st.tabs(["Overview", "Network Analysis", "Risk Analysis", "Domain Analysis"])
                    
                    with tabs[0]:  # Overview tab
                        st.subheader("Analysis Overview")
                        combined_df = analysis_results.get('combined_df')
                        risk_scores = analysis_results.get('risk_scores')
                        domain_counts = analysis_results.get('domain_counts')
                        mortality_risks = analysis_results.get('mortality_risks')
                        
                        # Metrics overview
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Patients", combined_df['patient_id'].nunique())
                        with col2:
                            high_risk = len([s for s in risk_scores.values() if s > 7])
                            st.metric("High Risk Patients", high_risk)
                        with col3:
                            avg_score = sum(risk_scores.values()) / len(risk_scores) if risk_scores else 0
                            st.metric("Average Risk Score", f"{avg_score:.2f}")
                        with col4:
                            # Calculate average mortality risk
                            avg_mortality = sum(mortality_risks.values()) / len(mortality_risks) if mortality_risks else 0
                            st.metric("Avg Mortality Risk", f"{avg_mortality:.1f}%")
                            
                        with tabs[1]:  # Network Analysis tab
                            st.subheader("Condition Network Analysis")
                            G = analysis_results.get('G')
                            domain_df = analysis_results.get('domain_df')
                            
                            if G is not None and G.number_of_nodes() > 0:
                                network_fig = visualize_network_with_communities(G, domain_df)
                                if network_fig:
                                    st.plotly_chart(network_fig, use_container_width=True)
                                
                                # Display network metrics
                                network_metrics = analysis_results.get('network_metrics', {})
                                st.subheader("Network Metrics")
                                metrics_cols = st.columns(4)
                                with metrics_cols[0]:
                                    st.metric("Nodes", network_metrics.get('node_count', 0))
                                with metrics_cols[1]:
                                    st.metric("Edges", network_metrics.get('edge_count', 0))
                                with metrics_cols[2]:
                                    st.metric("Communities", network_metrics.get('communities', 0))
                                with metrics_cols[3]:
                                    st.metric("Density", f"{network_metrics.get('density', 0):.3f}")
                                    
                                # Display correlation analysis if calculated
                                if 'correlation_matrix' in analysis_results:
                                    st.subheader("Condition Correlations")
                                    corr_matrix = analysis_results.get('correlation_matrix')
                                    # Display heatmap of top correlations
                                    if not corr_matrix.empty:
                                        # Get top 20 conditions by degree centrality
                                        top_conditions = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:20]
                                        top_condition_names = [c[0] for c in top_conditions]
                                        
                                        # Filter correlation matrix to these conditions
                                        top_corr = corr_matrix.loc[top_condition_names, top_condition_names]
                                        
                                        fig = px.imshow(top_corr, 
                                                     x=top_corr.columns, 
                                                     y=top_corr.index,
                                                     color_continuous_scale='RdBu_r', 
                                                     zmin=-1, zmax=1)
                                        st.plotly_chart(fig, use_container_width=True)
                            
                        with tabs[2]:  # Risk Analysis tab
                            st.subheader("Risk Score Analysis")
                            
                            # Population risk distribution
                            st.subheader("Population Risk Distribution")
                            risk_df = pd.DataFrame({
                                'Patient ID': list(risk_scores.keys()),
                                'Risk Score': list(risk_scores.values())
                            })
                            risk_hist = px.histogram(risk_df, x='Risk Score',
                                                nbins=20,
                                                title="Distribution of Risk Scores",
                                                labels={'Risk Score': 'NHCRS Score'})
                            st.plotly_chart(risk_hist, use_container_width=True)
                            
                            # Risk categories
                            risk_categories = {
                                'Low Risk (0-3)': len([s for s in risk_scores.values() if s <= 3]),
                                'Moderate Risk (3-7)': len([s for s in risk_scores.values() if s > 3 and s <= 7]),
                                'High Risk (7-10)': len([s for s in risk_scores.values() if s > 7 and s <= 10]),
                                'Severe Risk (>10)': len([s for s in risk_scores.values() if s > 10])
                            }
                            
                            risk_cat_df = pd.DataFrame({
                                'Category': risk_categories.keys(),
                                'Count': risk_categories.values()
                            })
                            
                            risk_cat_chart = px.pie(risk_cat_df, names='Category', values='Count',
                                                 title="Risk Score Categories")
                            st.plotly_chart(risk_cat_chart, use_container_width=True)
                            
                        with tabs[3]:  # Domain Analysis tab
                            st.subheader("Clinical Domain Analysis")
                            
                            # Display domain distribution
                            domain_df = pd.DataFrame({
                                'Domain': domain_counts.keys(),
                                'Count': domain_counts.values()
                            })
                            domain_chart = px.bar(domain_df, x='Domain', y='Count',
                                              color='Domain', title="Distribution of Clinical Domains")
                            st.plotly_chart(domain_chart, use_container_width=True)
                            
                            # Domain-specific risk scores
                            domain_scores = analysis_results.get('domain_scores', {})
                            if domain_scores:
                                # Calculate average domain scores across population
                                avg_domain_scores = {}
                                for patient_id, scores in domain_scores.items():
                                    for domain, score in scores.items():
                                        if domain not in avg_domain_scores:
                                            avg_domain_scores[domain] = []
                                        avg_domain_scores[domain].append(score)
                                
                                # Calculate averages
                                avg_scores = {domain: sum(scores)/len(scores) if scores else 0 
                                            for domain, scores in avg_domain_scores.items()}
                                
                                # Create dataframe for visualization
                                avg_domain_df = pd.DataFrame({
                                    'Domain': avg_scores.keys(),
                                    'Average Score': avg_scores.values()
                                })
                                
                                domain_score_chart = px.bar(avg_domain_df, x='Domain', y='Average Score',
                                                        color='Domain', title="Average Domain Risk Scores")
                                st.plotly_chart(domain_score_chart, use_container_width=True)
                        
                        # Show the appropriate view based on user selection
                        if view_type == 'Population Analysis':
                            st.subheader("Population Analysis")
                            st.info("Use the tabs above to explore different aspects of the population analysis.")
                            
                        elif view_type == 'Single Patient Analysis':
                            st.subheader("Single Patient Analysis")
                            # Display patient selection and details
                            patient_ids = analysis_results.get('combined_df')['patient_id'].unique()
                            if len(patient_ids) > 0:
                                selected_patient = st.selectbox("Select Patient for Detailed Analysis:", 
                                                             options=patient_ids)
                                
                                # Get patient data
                                patient_data = analysis_results.get('patient_results').get(selected_patient, {})
                                patient_risk = {
                                    'total_score': analysis_results.get('risk_scores', {}).get(selected_patient, 0),
                                    'mortality_risk_10yr': analysis_results.get('mortality_risks', {}).get(selected_patient, 0),
                                    'hospitalization_risk_5yr': analysis_results.get('hospitalization_risks', {}).get(selected_patient, 0),
                                    'domain_scores': analysis_results.get('domain_scores', {}).get(selected_patient, {})
                                }
                                
                                # Use tabs for better organization of patient details
                                patient_tabs = st.tabs(["Patient Info", "Risk Scores", "Population Comparison"])
                                
                                with patient_tabs[0]:  # Patient Info tab
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.subheader("Patient Information")
                                        st.write(f"Patient ID: {selected_patient}")
                                        st.write(f"Age: {patient_data.get('age', 'Unknown')}")
                                        st.write(f"Gender: {patient_data.get('gender', 'Unknown')}")
                                    
                                    with col2:
                                        # Display network metrics for this patient
                                        st.subheader("Network Position")
                                        network_metrics = patient_data.get('network_metrics', {})
                                        st.write(f"Degree Centrality: {network_metrics.get('degree_centrality', 0):.3f}")
                                        st.write(f"Betweenness Centrality: {network_metrics.get('betweenness_centrality', 0):.3f}")
                                    
                                    # Display conditions
                                    st.subheader("Conditions")
                                    conditions = patient_data.get('conditions', [])
                                    if conditions:
                                        # Organize by domain
                                        domain_conditions = defaultdict(list)
                                        for condition in conditions:
                                            domain = assign_clinical_domain(condition)
                                            domain_conditions[domain].append(condition)
                                        
                                        # Display by domain with expanders
                                        for domain, conditions_list in domain_conditions.items():
                                            with st.expander(f"{domain.replace('_', ' ').title()} ({len(conditions_list)} conditions)"):
                                                for condition in conditions_list:
                                                    st.write(f"- {condition}")
                                    else:
                                        st.write("No conditions recorded for this patient.")
                                    
                                    # Display biomarkers if available
                                    biomarkers = patient_data.get('biomarkers', {})
                                    if biomarkers:
                                        st.subheader("Biomarkers")
                                        biomarker_df = pd.DataFrame({
                                            'Biomarker': biomarkers.keys(),
                                            'Value': biomarkers.values()
                                        })
                                        st.dataframe(biomarker_df)
                                    
                                    with patient_tabs[1]:  # Risk Scores tab
                                        st.subheader("Risk Assessment")
                                        
                                        # Risk metrics
                                        risk_cols = st.columns(3)
                                        with risk_cols[0]:
                                            st.metric("NHCRS Score", f"{patient_risk['total_score']:.1f}")
                                        with risk_cols[1]:
                                            st.metric("10-Year Mortality Risk", f"{patient_risk['mortality_risk_10yr']:.1f}%")
                                        with risk_cols[2]:
                                            st.metric("5-Year Hospitalization Risk", f"{patient_risk['hospitalization_risk_5yr']:.1f}%")
                                        
                                        # Domain scores visualization
                                        if patient_risk['domain_scores']:
                                            domain_scores_df = pd.DataFrame({
                                                'Domain': list(patient_risk['domain_scores'].keys()),
                                                'Score': list(patient_risk['domain_scores'].values())
                                            })
                                            
                                            domain_fig = px.bar(domain_scores_df, x='Domain', y='Score',
                                                              color='Score', color_continuous_scale='Reds',
                                                              title="Patient Domain Risk Scores")
                                            st.plotly_chart(domain_fig, use_container_width=True)
                                            
                                            # Radar chart for domain visualization
                                            domains = list(patient_risk['domain_scores'].keys())
                                            scores = list(patient_risk['domain_scores'].values())
                                            
                                            # Close the loop for radar chart
                                            domains.append(domains[0])
                                            scores.append(scores[0])
                                            
                                            radar_fig = go.Figure()
                                            radar_fig.add_trace(go.Scatterpolar(
                                                r=scores,
                                                theta=domains,
                                                fill='toself',
                                                name='Patient Domain Scores'
                                            ))
                                            
                                            radar_fig.update_layout(
                                                polar=dict(
                                                    radialaxis=dict(
                                                        visible=True,
                                                        range=[0, 10]
                                                    )
                                                ),
                                                title="Domain Risk Profile"
                                            )
                                            st.plotly_chart(radar_fig, use_container_width=True)
                                    
                                    with patient_tabs[2]:  # Population Comparison tab
                                        st.subheader("Patient vs Population Comparison")
                                        
                                        # Risk score comparison
                                        risk_scores = analysis_results.get('risk_scores', {})
                                        mortality_risks = analysis_results.get('mortality_risks', {})
                                        avg_score = sum(risk_scores.values()) / len(risk_scores) if risk_scores else 0
                                        avg_mortality = sum(mortality_risks.values()) / len(mortality_risks) if mortality_risks else 0
                                        
                                        compare_cols = st.columns(3)
                                        with compare_cols[0]:
                                            patient_score = patient_risk['total_score']
                                            delta = f"{patient_score - avg_score:.2f}"
                                            st.metric("Risk Score", f"{patient_score:.2f}", delta=delta, 
                                                    delta_color="inverse")
                                            
                                        with compare_cols[1]:
                                            patient_mortality = patient_risk['mortality_risk_10yr']
                                            delta_m = f"{patient_mortality - avg_mortality:.1f}%"
                                            st.metric("Mortality Risk", f"{patient_mortality:.1f}%", delta=delta_m,
                                                    delta_color="inverse")
                                            
                                        with compare_cols[2]:
                                            # Domain count comparison
                                            patient_domains = len(patient_risk['domain_scores'])
                                            domain_scores = analysis_results.get('domain_scores', {})
                                            avg_domains = sum(len(d) for d in domain_scores.values()) / len(patient_ids)
                                            delta_d = f"{patient_domains - avg_domains:.1f}"
                                            st.metric("Clinical Domains", patient_domains, delta=delta_d,
                                                    delta_color="inverse")
                                        
                                        # Domain comparison chart
                                        if patient_risk['domain_scores']:
                                            st.subheader("Domain Score Comparison")
                                            # Get average domain scores across population
                                            all_domain_scores = analysis_results.get('domain_scores', {})
                                            avg_domain_scores = {}
                                            for domain in patient_risk['domain_scores'].keys():
                                                domain_values = [scores.get(domain, 0) for scores in all_domain_scores.values()]
                                                avg_domain_scores[domain] = sum(domain_values) / len(domain_values) if domain_values else 0
                                            
                                            # Create comparison dataframe
                                            compare_df = pd.DataFrame({
                                                'Domain': list(patient_risk['domain_scores'].keys()),
                                                'Patient Score': list(patient_risk['domain_scores'].values()),
                                                'Population Average': [avg_domain_scores.get(d, 0) for d in patient_risk['domain_scores'].keys()]
                                            })
                                            
                                            # Create comparison chart
                                            domain_compare = px.bar(compare_df, x='Domain', y=['Patient Score', 'Population Average'], 
                                                               barmode='group', title="Domain Score Comparison")
                                            st.plotly_chart(domain_compare, use_container_width=True)
                                        
                                        # Percentile rank information
                                        st.subheader("Patient Percentile Ranks")
                                        
                                        # Calculate percentile for this patient's risk score
                                        all_scores = list(risk_scores.values())
                                        all_scores.sort()
                                        patient_score = patient_risk['total_score']
                                        percentile = sum(1 for s in all_scores if s < patient_score) / len(all_scores) * 100
                                        
                                        st.write(f"This patient's risk score is higher than {percentile:.1f}% of the population.")
                                        
                                        # Visualize position in population
                                        fig = px.histogram(all_scores, 
                                                        title="Patient Position in Population Distribution",
                                                        labels={'value': 'Risk Score', 'count': 'Number of Patients'})
                                        fig.add_vline(x=patient_score, line_width=3, line_dash="dash", line_color="red")
                                        fig.add_annotation(x=patient_score, y=0, 
                                                        text=f"Patient: {patient_score:.1f}",
                                                        showarrow=True, arrowhead=1)
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Generate AI recommendations if API key available
                                    if 'openai_api_key' in st.session_state:
                                        if st.button("Generate AI Clinical Recommendations"):
                                            with st.spinner("Generating AI recommendations..."):
                                                recommendations = generate_clinical_recommendations(patient_data, patient_risk)
                                                st.subheader("AI Clinical Recommendations")
                                                st.write(recommendations)
                                                
                                                # Store recommendations for PDF
                                                st.session_state.recommendations = recommendations
                                    
                                    # PDF generation
                                    if st.button("Generate PDF Report"):
                                        with st.spinner("Generating PDF report..."):
                                            recommendations = st.session_state.get('recommendations', None)
                                            pdf_bytes = generate_pdf_report(patient_data, patient_risk, recommendations)
                                            
                                            # Create download button
                                            b64_pdf = base64.b64encode(pdf_bytes).decode()
                                            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="patient_{selected_patient}_report.pdf">Download PDF Report</a>'
                                            st.markdown(href, unsafe_allow_html=True)
                            
    # FHIR Integration
    else:
        st.info("FHIR Integration is coming soon! Please use local file upload for now.")
    
    # Add footer
    st.sidebar.divider()
    st.sidebar.markdown("© 2025 Nudge Health AI. All rights reserved.")

# Helper function to extract patient data from DataFrame
def get_patient_data(df, patient_id):
    """Extract relevant data for a specific patient"""
    patient_df = df[df['PatId'] == patient_id]
    
    # Extract basic demographics
    age = patient_df['Age'].iloc[0] if 'Age' in patient_df.columns and not patient_df.empty else 'Unknown'
    gender = patient_df['Gender'].iloc[0] if 'Gender' in patient_df.columns and not patient_df.empty else 'Unknown'
    
    # Extract conditions
    conditions = []
    for col in patient_df.columns:
        if 'diag' in col.lower() or 'icd' in col.lower() or 'condition' in col.lower():
            values = patient_df[col].dropna().unique()
            conditions.extend([str(v) for v in values if str(v).strip()])
    
    # Create patient data dictionary
    patient_data = {
        'patient_id': patient_id,
        'age': age,
        'gender': gender,
        'conditions': conditions
    }
    
    return patient_data

# Helper function to calculate risk scores for a specific patient
def calculate_patient_risk_scores(patient_data, G=None, network_metrics=None):
    """Calculate risk scores for a patient based on network metrics"""
    # Get conditions and their domains
    conditions = patient_data.get('conditions', [])
    domains = [assign_clinical_domain(condition) for condition in conditions]
    domain_counts = {domain: domains.count(domain) for domain in set(domains)}
    
    # Calculate domain scores
    domain_scores = {}
    for domain, count in domain_counts.items():
        # Base score from count
        base_score = min(count * 0.5, 3.0)
        
        # Add severity component
        severity_component = 0
        for condition in conditions:
            if assign_clinical_domain(condition) == domain:
                severity = get_condition_severity(condition)
                severity_component += severity * 0.3
        
        # Network component if network data is available
        network_component = 0
        if G is not None and network_metrics is not None:
            for condition in conditions:
                if assign_clinical_domain(condition) == domain and condition in network_metrics.get('centrality', {}):
                    network_component += network_metrics['centrality'].get(condition, 0) * 2
        
        # Combined domain score
        domain_scores[domain] = base_score + severity_component + network_component
    
    # Calculate total risk score (NHCRS)
    total_score = sum(domain_scores.values())
    
    # Calculate mortality risk (simplified model)
    age_factor = int(patient_data.get('age', 50)) / 20  # Age scaling factor
    mortality_risk_10yr = min(95, total_score * 3 * age_factor)
    
    # Calculate hospitalization risk
    hospitalization_risk_5yr = min(90, total_score * 5)
    
    # Prepare risk scores dictionary
    risk_scores = {
        'total_score': total_score,
        'domain_scores': domain_scores,
        'mortality_risk_10yr': mortality_risk_10yr,
        'hospitalization_risk_5yr': hospitalization_risk_5yr
    }
    
    return risk_scores

def generate_clinical_recommendations(patient_data, risk_scores):
    """
    Generate clinical recommendations using OpenAI based on patient data and risk scores.
    
    Args:
        patient_data: Dictionary containing patient information
        risk_scores: Dictionary containing risk score information
        
    Returns:
        String with clinical recommendations
    """
    try:
        # Set OpenAI API key from session state
        if 'openai_api_key' in st.session_state:
            openai.api_key = st.session_state.openai_api_key
        else:
            return "OpenAI API key not found. Please provide a valid API key to generate recommendations."
        
        # Prepare detailed patient information for the prompt
        age = patient_data.get('age', 'Unknown')
        gender = patient_data.get('gender', 'Unknown')
        conditions = patient_data.get('conditions', [])
        domains = risk_scores.get('domain_scores', {})
        total_score = risk_scores.get('total_score', 0)
        mortality_risk = risk_scores.get('mortality_risk_10yr', 0)
        
        # Format the domain scores for better readability
        domain_text = ""
        for domain, score in domains.items():
            domain_text += f"- {domain.replace('_', ' ').title()}: {score:.2f}/10\n"
        
        # Format conditions list
        conditions_text = "\n".join([f"- {condition}" for condition in conditions])
        
        # Construct the prompt for the AI
        prompt = f"""
        You are an expert clinical decision support system. Based on the following patient data, 
        provide evidence-based clinical recommendations focusing on risk mitigation, monitoring, and 
        lifestyle interventions. Format your response in clear sections.
        
        PATIENT INFORMATION:
        - Age: {age}
        - Gender: {gender}
        - Conditions: 
        {conditions_text}
        
        RISK ASSESSMENT:
        - Total Nudge Health Clinical Risk Score (NHCRS): {total_score:.2f}
        - 10-Year Mortality Risk: {mortality_risk:.1f}%
        - Clinical Domain Scores:
        {domain_text}
        
        Provide specific, actionable recommendations in the following categories:
        1. Additional Diagnostic Testing (prioritized by importance)
        2. Monitoring Protocol (what to monitor and frequency)
        3. Pharmacological Interventions (if appropriate)
        4. Lifestyle Optimization
        5. Preventive Measures
        
        For each recommendation, briefly explain the rationale based on the patient's specific risk profile.
        """
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a clinical decision support system providing evidence-based recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        # Extract the recommendation text
        recommendation_text = response.choices[0].message.content.strip()
        return recommendation_text
        
    except Exception as e:
        logger.error(f"Error generating clinical recommendations: {str(e)}")
        return f"Error generating recommendations: {str(e)}"

def generate_pdf_report(patient_data, risk_scores, recommendations=None):
    """
    Generate a PDF report containing patient information and risk scores.
    
    Args:
        patient_data: Dictionary containing patient information
        risk_scores: Dictionary containing risk score information
        recommendations: Optional AI-generated recommendations
        
    Returns:
        Bytes containing the PDF file
    """
    try:
        # Create PDF object
        pdf = FPDF()
        pdf.add_page()
        
        # Set up header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "Nudge Health AI Clinical Analysis Report", 0, 1, 'C')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
        pdf.line(10, 30, 200, 30)
        pdf.ln(10)
        
        # Patient information
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Patient Overview", 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        pdf.cell(60, 8, f"Patient ID: {patient_data.get('patient_id', 'Unknown')}", 0, 1)
        pdf.cell(60, 8, f"Age: {patient_data.get('age', 'Unknown')}", 0, 1)
        pdf.cell(60, 8, f"Gender: {patient_data.get('gender', 'Unknown')}", 0, 1)
        pdf.ln(5)
        
        # Conditions
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Conditions:", 0, 1)
        pdf.set_font('Arial', '', 10)
        for condition in patient_data.get('conditions', []):
            pdf.cell(0, 6, f"• {condition}", 0, 1)
        pdf.ln(5)
        
        # Risk Scores
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Risk Assessment", 0, 1, 'L')
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f"NHCRS Total Score: {risk_scores.get('total_score', 0):.2f}", 0, 1)
        pdf.cell(0, 8, f"10-Year Mortality Risk: {risk_scores.get('mortality_risk_10yr', 0):.1f}%", 0, 1)
        pdf.cell(0, 8, f"5-Year Hospitalization Risk: {risk_scores.get('hospitalization_risk_5yr', 0):.1f}%", 0, 1)
        pdf.ln(5)
        
        # Domain Scores
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Clinical Domain Scores:", 0, 1)
        pdf.set_font('Arial', '', 10)
        for domain, score in risk_scores.get('domain_scores', {}).items():
            formatted_domain = domain.replace('_', ' ').title()
            pdf.cell(0, 6, f"• {formatted_domain}: {score:.2f}/10", 0, 1)
        pdf.ln(5)
        
        # Network Analysis if available
        if 'network_metrics' in patient_data:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "Network Analysis", 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            metrics = patient_data['network_metrics']
            pdf.cell(0, 6, f"Degree Centrality: {metrics.get('degree_centrality', 0):.3f}", 0, 1)
            pdf.cell(0, 6, f"Betweenness Centrality: {metrics.get('betweenness_centrality', 0):.3f}", 0, 1)
            pdf.ln(5)
        
        # AI Recommendations if available
        if recommendations:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "AI Clinical Recommendations", 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            
            # Split the recommendations into lines and add to PDF
            lines = recommendations.split('\n')
            for line in lines:
                # Handle markdown-like formatting
                if line.strip().startswith('#'):
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 8, line.strip('# '), 0, 1)
                    pdf.set_font('Arial', '', 10)
                elif line.strip().startswith('*') or line.strip().startswith('-'):
                    pdf.cell(0, 6, f"  {line.strip('*- ')}", 0, 1)
                else:
                    pdf.multi_cell(0, 5, line)
            
        # Add footer with copyright
        pdf.ln(20)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, "© 2025 Nudge Health AI. All rights reserved.", ln=True, align='C')
        
        # Return the PDF as bytes
        return pdf.output(dest='S').encode('latin-1')
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        # Return a simple error PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "Error Generating Report", 0, 1, 'C')
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"An error occurred: {str(e)}", 0, 1, 'L')
        return pdf.output(dest='S').encode('latin-1')

# Display patient risk scores in a formatted way
def display_patient_risk_scores(patient_data, risk_scores):
    """Display patient risk scores in the Streamlit UI"""
    st.subheader(f"Risk Assessment for Patient {patient_data['patient_id']}")
    
    # Display overall scores
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NHCRS Score", f"{risk_scores['total_score']:.1f}")
    with col2:
        st.metric("10-Year Mortality Risk", f"{risk_scores['mortality_risk_10yr']:.1f}%")
    with col3:
        st.metric("5-Year Hospitalization Risk", f"{risk_scores['hospitalization_risk_5yr']:.1f}%")
    
    # Display domain scores
    st.subheader("Clinical Domain Scores")
    domain_scores = risk_scores['domain_scores']
    
    # Convert to DataFrame for charting
    domain_df = pd.DataFrame({
        'Domain': domain_scores.keys(),
        'Score': domain_scores.values()
    })
    
    # Create bar chart
    fig = px.bar(domain_df, x='Domain', y='Score', 
                color='Score', color_continuous_scale='Reds',
                title="Risk Scores by Clinical Domain")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 