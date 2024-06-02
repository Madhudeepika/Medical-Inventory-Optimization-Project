  cleaned_df)

import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

# Load your dataset
df = pd.read_csv(r'med_Inv.csv')

# Your data processing and visualization code here...

# Generate the EDA report
profile = ProfileReport(df)
profile.to_file("output.html")  # Save the report to an HTML file
