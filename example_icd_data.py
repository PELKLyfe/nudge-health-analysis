import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate example ICD-10 data
def generate_icd_data(num_patients=100):
    # Common ICD-10 codes for testing
    icd_codes = [
        'E11.9',  # Type 2 diabetes without complications
        'I10',    # Essential hypertension
        'F41.9',  # Anxiety disorder, unspecified
        'J45.909',# Unspecified asthma, uncomplicated
        'M54.5',  # Low back pain
        'K21.9',  # GERD without esophagitis
    ]
    
    # Generate random data
    data = []
    for patient_id in range(1, num_patients + 1):
        # Each patient gets 1-4 random conditions
        num_conditions = np.random.randint(1, 5)
        conditions = np.random.choice(icd_codes, num_conditions, replace=False)
        
        for condition in conditions:
            # Generate random dates within last year
            date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            data.append({
                'patient_id': f'P{patient_id:03d}',
                'code': condition,
                'diagnosis_date': date.strftime('%Y-%m-%d'),
                'active': np.random.choice(['Yes', 'No'], p=[0.8, 0.2])
            })
    
    df = pd.DataFrame(data)
    return df

# Generate example data
icd_df = generate_icd_data()

# Save to Excel
icd_df.to_excel('example_icd_data.xlsx', index=False) 