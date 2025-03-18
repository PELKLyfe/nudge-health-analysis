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
    Fetch and process biomarker reference ranges from authoritative sources like CDC.
    Returns a dictionary with biomarker reference ranges and metadata.
    """
    try:
        # First try to load from local cache
        if os.path.exists('biomarker_reference.json'):
            with open('biomarker_reference.json', 'r') as f:
                return json.load(f)
        
        # If not in cache, fetch from primary source(s)
        # CDC NHANES data for common biomarkers
        cdc_url = "https://wwwn.cdc.gov/nchs/nhanes/vitamind/analyticalnote.aspx?b=2017&e=2020&d=HDL&x=gender"
        
        # This is a placeholder - in a real implementation, you'd need to parse the CDC data
        # which is not directly available in simple JSON/CSV format
        
        # For now, we'll create a default reference data structure based on medical standards
        biomarker_ref = {
            'glucose': {
                'description': 'Fasting Blood Glucose',
                'units': 'mg/dL',
                'low_range': 70,
                'normal_range': [70, 99],
                'high_range': [100, 125],
                'critical_range': 126,
                'source': 'American Diabetes Association',
                'updated': datetime.now().strftime('%Y-%m-%d')
            },
            'cholesterol': {
                'description': 'Total Cholesterol',
                'units': 'mg/dL',
                'low_range': None,
                'normal_range': [0, 200],
                'high_range': [200, 240],
                'critical_range': 240,
                'source': 'American Heart Association',
                'updated': datetime.now().strftime('%Y-%m-%d')
            },
            'hdl': {
                'description': 'High-Density Lipoprotein',
                'units': 'mg/dL',
                'low_range': 40,
                'normal_range': [40, 60],
                'high_range': None,  # Higher is better for HDL
                'critical_range': None,
                'source': 'American Heart Association',
                'updated': datetime.now().strftime('%Y-%m-%d')
            },
            'ldl': {
                'description': 'Low-Density Lipoprotein',
                'units': 'mg/dL',
                'low_range': None,
                'normal_range': [0, 100],
                'high_range': [100, 160],
                'critical_range': 160,
                'source': 'American Heart Association',
                'updated': datetime.now().strftime('%Y-%m-%d')
            },
            'triglycerides': {
                'description': 'Triglycerides',
                'units': 'mg/dL',
                'low_range': None,
                'normal_range': [0, 150],
                'high_range': [150, 500],
                'critical_range': 500,
                'source': 'American Heart Association',
                'updated': datetime.now().strftime('%Y-%m-%d')
            },
            'blood_pressure_systolic': {
                'description': 'Systolic Blood Pressure',
                'units': 'mmHg',
                'low_range': None,
                'normal_range': [90, 120],
                'high_range': [120, 140],
                'critical_range': 140,
                'source': 'American Heart Association',
                'updated': datetime.now().strftime('%Y-%m-%d')
            },
            'blood_pressure_diastolic': {
                'description': 'Diastolic Blood Pressure',
                'units': 'mmHg',
                'low_range': None,
                'normal_range': [60, 80],
                'high_range': [80, 90],
                'critical_range': 90,
                'source': 'American Heart Association',
                'updated': datetime.now().strftime('%Y-%m-%d')
            },
            'hba1c': {
                'description': 'Hemoglobin A1c',
                'units': '%',
                'low_range': None,
                'normal_range': [4.0, 5.6],
                'high_range': [5.7, 6.4],
                'critical_range': 6.5,
                'source': 'American Diabetes Association',
                'updated': datetime.now().strftime('%Y-%m-%d')
            },
            'crp': {
                'description': 'C-Reactive Protein',
                'units': 'mg/L',
                'low_range': None,
                'normal_range': [0, 3.0],
                'high_range': [3.0, 10.0],
                'critical_range': 10.0,
                'source': 'American Heart Association',
                'updated': datetime.now().strftime('%Y-%m-%d')
            }
        }
        
        # Cache the results
        with open('biomarker_reference.json', 'w') as f:
            json.dump(biomarker_ref, f)
        
        return biomarker_ref
    
    except Exception as e:
        logger.warning(f"Failed to fetch biomarker reference data: {str(e)}")
        # Return empty dictionary if fetch fails
        return {}

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
    for target, possible_names in id_mappings.items():
        found_col = find_best_column_match(target, possible_names)
        if found_col:
            column_mapping[target] = found_col
        else:
            logger.warning(f"Could not find a match for {target} column")
            
    # Get diagnosis columns
    diagnosis_cols = []
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
        # Look for numbered columns that might be diagnoses
        numbered_cols = [col for col in df.columns 
                        if re.search(r'\d+$', col) or any(c.isdigit() for c in col)]
        if numbered_cols:
            diagnosis_cols = numbered_cols
            logger.warning(f"No diagnosis columns found, using numbered columns: {numbered_cols}")
    
    # Remove duplicates
    diagnosis_cols = list(set(diagnosis_cols))
    
    # Sort diagnosis columns naturally if they contain numbers
    try:
        def extract_number(col):
            match = re.search(r'\d+', col)
            return int(match.group()) if match else 0
            
        diagnosis_cols.sort(key=extract_number)
    except:
        # Fall back to alphabetical sort
        diagnosis_cols.sort()
    
    if not diagnosis_cols:
        logger.error(f"No diagnosis columns found in dataframe")
        return pd.DataFrame()
        
    logger.info(f"Using diagnosis columns: {diagnosis_cols}")
    
    # Process each row
    for idx, row in df.iterrows():
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
    
    # Convert to DataFrame and ensure unique index
    if records:
        result_df = pd.DataFrame(records)
        # Reset index to ensure uniqueness
        result_df = result_df.reset_index(drop=True)
        
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
    Assign a clinical domain based on ICD code or description.
    Maps each ICD-10 code to one of the five functional domains:
    - Cardiometabolic
    - Immune-Inflammation
    - Oncological (Cancer)
    - Neuro-Mental Health
    - Neurological & Frailty
    """
    # Convert to lowercase for case-insensitive matching
    code_lower = str(icd_code).lower()
    
    # Define domain mapping with keywords and ICD code prefixes
    domains = {
        'cardiometabolic': [
            # Keywords
            'heart', 'cardiac', 'hypertension', 'diabetes', 'obesity', 'cholesterol', 'lipid', 
            'atherosclerosis', 'coronary', 'stroke', 'vascular', 'metabolic', 'dyslipidemia',
            # ICD-10 code prefixes
            'i0', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'e08', 'e09', 'e10', 'e11', 'e13',
            'e66', 'e78'
        ],
        'immune_inflammation': [
            # Keywords
            'arthritis', 'immune', 'inflammation', 'asthma', 'allergy', 'lupus', 'rheumatoid',
            'infection', 'infectious', 'bacterial', 'viral', 'autoimmune', 'inflammatory',
            # ICD-10 code prefixes
            'a', 'b', 'j', 'l', 'm05', 'm06', 'm08', 'm30', 'm31', 'm32', 'm33', 'm34', 'm35', 'm36'
        ],
        'oncologic': [
            # Keywords
            'cancer', 'tumor', 'neoplasm', 'malignant', 'carcinoma', 'sarcoma', 'leukemia', 
            'lymphoma', 'metastatic', 'oncology',
            # ICD-10 code prefixes
            'c', 'd0', 'd1', 'd2', 'd3', 'd4'
        ],
        'neuro_mental_health': [
            # Keywords
            'depression', 'anxiety', 'psychiatric', 'bipolar', 'schizophrenia', 'mental',
            'psychological', 'behavior', 'mood', 'substance', 'alcohol', 'drug', 'dementia',
            # ICD-10 code prefixes
            'f'
        ],
        'neurological_frailty': [
            # Keywords
            'brain', 'nerve', 'seizure', 'neuropathy', 'parkinson', 'alzheimer', 'multiple sclerosis',
            'tremor', 'paralysis', 'weakness', 'frailty', 'fall', 'mobility', 'gait', 'balance',
            # ICD-10 code prefixes
            'g', 'r26', 'r27', 'r29', 'r41', 'r42', 'r47', 'r54', 'r55', 'r56'
        ]
    }
    
    # First check if the code starts with any of the ICD-10 prefixes
    for domain, keywords in domains.items():
        for keyword in keywords:
            if (keyword.isalnum() and len(keyword) <= 3 and code_lower.startswith(keyword)) or \
               (len(keyword) > 3 and keyword in code_lower):
                return domain
                
    # If no match found, return 'other'
    return 'other'

def create_domain_network(domain_df: pd.DataFrame) -> nx.Graph:
    """Create a network of clinical domains and conditions."""
    try:
        G = nx.Graph()
        
        # Add nodes for each unique condition
        conditions = domain_df['condition'].unique()
        G.add_nodes_from(conditions)
        
        # Create edges between conditions that occur in the same patient
        for patient_id in domain_df['patient_id'].unique():
            patient_conditions = domain_df[domain_df['patient_id'] == patient_id]['condition'].tolist()
            for i in range(len(patient_conditions)):
                for j in range(i + 1, len(patient_conditions)):
                    if G.has_edge(patient_conditions[i], patient_conditions[j]):
                        G[patient_conditions[i]][patient_conditions[j]]['weight'] += 1
                    else:
                        G.add_edge(patient_conditions[i], patient_conditions[j], weight=1)
        
        return G
        
    except Exception as e:
        logger.error(f"Error in create_domain_network: {str(e)}")
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
    Create an interactive network visualization with community detection.
    
    Args:
        G: NetworkX graph object
        domain_df: DataFrame with domain information
        
    Returns:
        Plotly figure object
    """
    try:
        if G.number_of_nodes() == 0:
            return None
            
        # Community detection using Louvain method
        if COMMUNITY_DETECTION_AVAILABLE:
            partition = community_louvain.best_partition(G)
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
            
            # Get number of communities
            num_communities = len(communities)
            logger.info(f"Detected {num_communities} communities in the network")
        else:
            # Fallback if community detection not available
            partition = {node: 0 for node in G.nodes()}
            communities = {0: list(G.nodes())}
            num_communities = 1
            logger.warning("Community detection package not available. Using single community.")
        
        # Create a layout for the network
        try:
            # Use force-directed layout
            pos = nx.spring_layout(G, k=0.15, seed=42)
        except:
            # Fallback to alternative layout
            pos = nx.kamada_kawai_layout(G)
            
        # Get node information
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node degree and community
            degree = G.degree(node)
            community = partition.get(node, 0)
            
            # Get domain for this condition
            domain = assign_clinical_domain(node)
            domain_formatted = domain.replace('_', ' ').title()
            
            # Node text for hover info
            node_text.append(f"Condition: {node}<br>Degree: {degree}<br>Domain: {domain_formatted}<br>Community: {community}")
            
            # Node size based on degree
            node_size.append(10 + degree * 2)
            
            # Node color based on community
            node_color.append(community)
        
        # Get edge information
        edge_x = []
        edge_y = []
        edge_width = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Add line endpoints
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge width based on weight
            weight = edge[2].get('weight', 1)
            edge_width.append(1 + weight * 0.5)
        
        # Create edges trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        # Create nodes trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Community',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=1, color='#888')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                      layout=go.Layout(
                          title='Condition Network with Communities',
                          titlefont=dict(size=16),
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20, l=5, r=5, t=40),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                      ))
        
        # Add annotations for community descriptions
        annotations = []
        for community_id, nodes in communities.items():
            if len(nodes) > 0:
                # Get central position of this community
                comm_x = sum(pos[node][0] for node in nodes if node in pos) / len(nodes)
                comm_y = sum(pos[node][1] for node in nodes if node in pos) / len(nodes)
                
                # Get dominant domain in this community
                domain_counts = {}
                for node in nodes:
                    domain = assign_clinical_domain(node)
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
                dominant_domain = max(domain_counts, key=domain_counts.get)
                domain_text = dominant_domain.replace('_', ' ').title()
                
                # Add annotation
                annotations.append(
                    dict(
                        x=comm_x,
                        y=comm_y,
                        xref="x",
                        yref="y",
                        text=f"Community {community_id}<br>({domain_text})",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40
                    )
                )
        
        fig.update_layout(annotations=annotations)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in visualize_network_with_communities: {str(e)}")
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
    Calculate total risk score based on conditions and network metrics using the NHCRS model.
    
    Args:
        patient_data: Dictionary containing patient information including conditions
        
    Returns:
        Dictionary containing risk score information
    """
    try:
        # Initialize domain scores
        domain_scores = defaultdict(float)
        
        # Extract biomarker data if available
        biomarkers = patient_data.get('biomarkers', {})
        
        # Extract SDOH data if available
        sdoh_data = patient_data.get('sdoh_data', {})
        
        # Process each condition to calculate domain-specific scores
        for condition in patient_data['conditions']:
            domain = assign_clinical_domain(condition)
            domain_score = calculate_domain_risk_score(
                [condition], 
                domain,
                patient_data['age'],
                patient_data['gender'],
                biomarkers,
                sdoh_data
            )
            domain_scores[domain] += domain_score
        
        # Apply network analysis modifier if available
        network_multiplier = 1.0
        if 'network_metrics' in patient_data:
            if patient_data['network_metrics'].get('degree_centrality', 0) > 0.5:
                network_multiplier += 0.2
            if patient_data['network_metrics'].get('betweenness_centrality', 0) > 0.3:
                network_multiplier += 0.1
                
        # Calculate NHCRS total score
        nhcrs_total = calculate_nhcrs_total(dict(domain_scores))
        
        # Apply network multiplier
        nhcrs_total *= network_multiplier
        
        # Calculate outcome probabilities
        mortality_risk = calculate_mortality_risk(nhcrs_total)
        hospitalization_risk = calculate_hospitalization_risk(nhcrs_total)
        
        # Convert to percentages
        mortality_risk_pct = round(mortality_risk * 100, 1)
        hospitalization_risk_pct = round(hospitalization_risk * 100, 1)
        
        # Determine risk level
        risk_level = 'Low'
        if nhcrs_total > 10:
            risk_level = 'High'
        elif nhcrs_total > 5:
            risk_level = 'Medium'
            
        return {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'total_score': nhcrs_total,
            'risk_level': risk_level,
            'domain_scores': dict(domain_scores),
            'network_multiplier': network_multiplier,
            'mortality_risk_10yr': mortality_risk_pct,
            'hospitalization_risk_5yr': hospitalization_risk_pct
        }
        
    except Exception as e:
        logger.error(f"Error in calculate_total_risk_score: {str(e)}")
        return {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'total_score': 0,
            'risk_level': 'Unknown',
            'domain_scores': {},
            'network_multiplier': 1.0,
            'mortality_risk_10yr': 0.0,
            'hospitalization_risk_5yr': 0.0
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
    """Process biomarker data into a standardized format."""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
            
        # Define common biomarker names and their variations
        biomarker_mappings = {
            'glucose': ['glucose', 'glu', 'blood sugar', 'fbs'],
            'cholesterol': ['cholesterol', 'chol', 'tc'],
            'hdl': ['hdl', 'hdl-c', 'good cholesterol'],
            'ldl': ['ldl', 'ldl-c', 'bad cholesterol'],
            'triglycerides': ['triglycerides', 'trig', 'tg'],
            'blood_pressure_systolic': ['systolic', 'sbp', 'bp_systolic'],
            'blood_pressure_diastolic': ['diastolic', 'dbp', 'bp_diastolic']
        }
        
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
                # Use fuzzy matching
                best_match = None
                best_score = 0
                threshold = 80
                
                for col in df.columns:
                    for name in possible_names:
                        score = fuzz.ratio(col.lower(), name.lower())
                        if score > best_score and score >= threshold:
                            best_match = col
                            best_score = score
                            
                if best_match:
                    return best_match
                    
            return None
            
        # Find and rename columns
        column_mapping = {}
        for standard_name, variations in biomarker_mappings.items():
            matched_col = find_best_column_match(standard_name, variations)
            if matched_col:
                column_mapping[matched_col] = standard_name
                
        if not column_mapping:
            logger.warning("No biomarker columns found")
            return pd.DataFrame()
            
        # Create new DataFrame with standardized column names
        result_df = df.copy()
        result_df = result_df.rename(columns=column_mapping)
        
        # Convert values to float where possible
        for col in column_mapping.values():
            try:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            except:
                logger.warning(f"Could not convert {col} to numeric values")
                
        return result_df
        
    except Exception as e:
        logger.error(f"Error in process_biomarker_data: {str(e)}")
        return pd.DataFrame()

def perform_integrated_analysis(icd_df, biomarker_df=None):
    """
    Unified analysis framework that integrates network analysis and risk score calculations.
    This optimized function combines both analyses to improve performance.
    
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
        
        # Process the data into long format if needed
        if 'diagnosis' not in icd_df.columns:
            icd_df = process_diagnosis_data(icd_df)
            if icd_df.empty:
                logger.warning("No valid diagnosis data to analyze")
                return None
        
        progress_bar.progress(10)
        
        # Get diagnosis columns
        diag_cols = [col for col in icd_df.columns if 'diagnosis' in col.lower()]
        
        # Process data into domains - this is needed for both analysis types
        domain_df = process_domain_data(icd_df, diag_cols)
        progress_bar.progress(20)
        
        # Create network for centrality analysis
        G = create_domain_network(domain_df)
        progress_bar.progress(30)
        
        # Calculate network metrics (used in both analyses)
        network_metrics = {}
        degree_cent = {}
        betweenness_cent = {}
        
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
            
            # Community detection using Louvain method
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
            except:
                communities = {0: list(G.nodes())}
                network_metrics['communities'] = 1
                network_metrics['modularity'] = 0
        
        progress_bar.progress(50)
        
        # Create a dictionary to store all node metrics for easy access
        node_centrality = {}
        for node in G.nodes():
            node_centrality[node] = {
                'degree_centrality': degree_cent.get(node, 0),
                'betweenness_centrality': betweenness_cent.get(node, 0),
                'domain': assign_clinical_domain(node)
            }
        
        # Calculate patient-level metrics and risk scores (with or without biomarkers)
        patient_ids = domain_df['patient_id'].unique()
        
        # Prepare results containers
        patient_results = {}
        risk_scores = {}
        domain_scores = {}
        mortality_risks = {}
        hospitalization_risks = {}
        
        # Prepare biomarker data if available
        if biomarker_df is not None and not biomarker_df.empty:
            # Filter to common patient IDs
            common_ids = set(patient_ids) & set(biomarker_df['patient_id'].unique())
            if common_ids:
                patient_ids = list(common_ids)
        
        # Track progress for patient calculations
        total_patients = len(patient_ids)
        
        # Process each patient
        for i, patient_id in enumerate(patient_ids):
            # Update progress bar - gradually move from 50% to 90%
            progress_value = 50 + int(40 * (i / total_patients))
            progress_bar.progress(progress_value)
            
            # Get patient conditions
            patient_conditions = domain_df[domain_df['patient_id'] == patient_id]['condition'].tolist()
            
            # Get patient demographics
            patient_rows = domain_df[domain_df['patient_id'] == patient_id]
            if patient_rows.empty:
                continue
                
            patient_data_row = patient_rows.iloc[0]
            
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
            
            if G.number_of_nodes() > 0:
                # Average the centrality of all patient conditions
                valid_conditions = [c for c in patient_conditions if c in G.nodes()]
                if valid_conditions:
                    network_metrics_patient['degree_centrality'] = np.mean([
                        degree_cent.get(c, 0) for c in valid_conditions
                    ])
                    network_metrics_patient['betweenness_centrality'] = np.mean([
                        betweenness_cent.get(c, 0) for c in valid_conditions
                    ])
            
            # Prepare patient data for risk calculation
            patient_info = {
                'patient_id': patient_id,
                'conditions': patient_conditions,
                'age': patient_data_row.get('age', 0),
                'gender': patient_data_row.get('gender', 'Unknown'),
                'network_metrics': network_metrics_patient,
                'biomarkers': biomarkers,
                'sdoh_data': {}  # Placeholder for SDOH data
            }
            
            # Calculate risk scores using the enhanced NHCRS model
            risk_result = calculate_total_risk_score(patient_info)
            
            # Store results
            patient_results[patient_id] = patient_info
            risk_scores[patient_id] = risk_result.get('total_score', 0)
            domain_scores[patient_id] = risk_result.get('domain_scores', {})
            mortality_risks[patient_id] = risk_result.get('mortality_risk_10yr', 0)
            hospitalization_risks[patient_id] = risk_result.get('hospitalization_risk_5yr', 0)
        
        # Calculate domain distribution
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
            'communities': communities if 'communities' in locals() else {0: list(G.nodes())},
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
    
    # OpenAI API key input in sidebar
    st.sidebar.subheader("OpenAI Settings (Optional)")
    api_key = st.sidebar.text_input("OpenAI API Key (for AI recommendations)", 
                                   type="password", 
                                   help="Enter your OpenAI API key to enable AI clinical recommendations")
    if api_key:
        st.session_state.openai_api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    
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
        # Data source selection
        data_source = st.radio(
            "Select Data Source:",
            ["Separate Files (ICD-10 & Biomarkers)", "Combined Excel File"],
            key="main_data_source_selection"
        )
        
        # UI for file uploads
        col1, col2 = st.columns(2)
        
        with col1:
            # ICD-10 data upload
            st.subheader("Upload ICD-10 Data")
            icd_file = st.file_uploader("Upload ICD-10 Excel file", 
                                         type=["xlsx", "xls"], 
                                         key="main_icd_upload",
                                         help="Excel file with columns: PatId, Gender, Age, and Diagnosis columns")
        
        with col2:
            # Biomarker data upload
            st.subheader("Upload Biomarker Data")
            bio_file = st.file_uploader("Upload Biomarker Excel file",
                                         type=["xlsx", "xls"],
                                         key="main_bio_upload",
                                         help="Excel file with biomarker measurements")
        
        # Process the data after upload
        if icd_file is not None:
            try:
                st.subheader("ICD-10 Data Processing")
                # Read the Excel file into a DataFrame
                icd_df = pd.read_excel(icd_file)
                
                # Fix column names - support PatId and Pat Id variations
                if 'Pat Id' in icd_df.columns and 'PatId' not in icd_df.columns:
                    icd_df.rename(columns={'Pat Id': 'PatId'}, inplace=True)
                
                # Process the DataFrame
                icd_df = process_diagnosis_data(icd_df)
                
                if icd_df is not None and not icd_df.empty:
                    st.success(f"Successfully processed {len(icd_df)} ICD-10 records for {len(icd_df['patient_id'].unique())} patients.")
                    
                    # Prepare a summary
                    icd_summary = {
                        "Total Patients": len(icd_df['patient_id'].unique()),
                        "Total Diagnoses": len(icd_df),
                        "Domains Found": icd_df['domain'].unique().tolist() if 'domain' in icd_df.columns else []
                    }
                    
                    # Display data summary
                    with st.expander("View ICD-10 Data Summary"):
                        st.write(icd_summary)
                    
                    # Process biomarker data if available
                    bio_df = None
                    if bio_file is not None:
                        try:
                            # Read the biomarker file into a DataFrame
                            bio_df = pd.read_excel(bio_file)
                            
                            # Fix column names - support PatId and Pat Id variations
                            if 'Pat Id' in bio_df.columns and 'PatId' not in bio_df.columns:
                                bio_df.rename(columns={'Pat Id': 'PatId'}, inplace=True)
                                
                            bio_df = process_biomarker_data(bio_df)
                            if not bio_df.empty:
                                st.success(f"Successfully processed biomarker data for {bio_df['patient_id'].nunique()} patients.")
                        except Exception as e:
                            st.error(f"Error processing biomarker data: {str(e)}")
                    
                    # Perform integrated analysis
                    with st.spinner("Performing comprehensive analysis... This may take a moment."):
                        analysis_results = perform_integrated_analysis(icd_df, bio_df)
                    
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
                                        
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.error("Please ensure your file contains the required columns: PatId, Gender, Age, Zip Code, and at least one Diagnosis column (ICD10 code or description)")
    
    # FHIR Integration
    else:
        st.info("FHIR Integration is coming soon! Please use local file upload for now.")
    
    # Add footer
    st.sidebar.divider()
    st.sidebar.markdown("© 2023 Nudge Health AI. All rights reserved.")

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
            
        # Footer
        pdf.set_y(-15)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, f"Nudge Health AI Analysis - Page {pdf.page_no()}/{{nb}}", 0, 0, 'C')
        
        # Return the PDF as bytes
        return pdf.output(dest='S').encode('latin1')
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        # Return a simple error PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "Error Generating Report", 0, 1, 'C')
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"An error occurred: {str(e)}", 0, 1, 'L')
        return pdf.output(dest='S').encode('latin1')

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