import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate example biomarker data
def generate_biomarker_data(num_patients=100):
    # Common biomarkers for testing
    biomarkers = {
        'HDL': {'mean': 50, 'std': 10, 'unit': 'mg/dL'},
        'LDL': {'mean': 100, 'std': 20, 'unit': 'mg/dL'},
        'TRIG': {'mean': 150, 'std': 30, 'unit': 'mg/dL'},
        'A1C': {'mean': 5.7, 'std': 0.5, 'unit': '%'},
        'GLU': {'mean': 100, 'std': 15, 'unit': 'mg/dL'},
        'BP_SYS': {'mean': 120, 'std': 10, 'unit': 'mmHg'},
        'BP_DIA': {'mean': 80, 'std': 8, 'unit': 'mmHg'}
    }
    
    data = []
    for patient_id in range(1, num_patients + 1):
        # Generate 3-5 measurements per patient over the last year
        num_measurements = np.random.randint(3, 6)
        dates = sorted([datetime.now() - timedelta(days=np.random.randint(1, 365)) 
                       for _ in range(num_measurements)])
        
        for date in dates:
            for marker, params in biomarkers.items():
                value = np.random.normal(params['mean'], params['std'])
                data.append({
                    'patient_id': f'P{patient_id:03d}',
                    'code': marker,
                    'value': round(value, 2),
                    'unit': params['unit'],
                    'measurement_date': date.strftime('%Y-%m-%d')
                })
    
    df = pd.DataFrame(data)
    return df

# Generate example data
bio_df = generate_biomarker_data()

# Save to Excel
bio_df.to_excel('example_biomarker_data.xlsx', index=False) 