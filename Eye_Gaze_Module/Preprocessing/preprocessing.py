# ==========================================================
# Eye Gaze Data Preprocessing (Tobii Eye Tracker - NIMHANS)
# ==========================================================

import pandas as pd
import os
import numpy as np

# ----------------------------------------------------------
# STEP 1: File Paths
# ----------------------------------------------------------
file_paths = [
    "participant1.xlsx",
    "participant2.xlsx",
    "participant3.xlsx",
    "participant4.xlsx"
]

# ----------------------------------------------------------
# STEP 2: Load and Combine Data
# ----------------------------------------------------------
all_data = []

for file in file_paths:
    if os.path.exists(file):
        df = pd.read_excel(file)
        df["participant_id"] = file.split(".")[0]
        all_data.append(df)
    else:
        print(f"⚠️ File not found: {file}")

combined_df = pd.concat(all_data, ignore_index=True)

print("✅ Data loaded and combined")
print("Shape:", combined_df.shape)

# ----------------------------------------------------------
# STEP 3: Timestamp Conversion
# ----------------------------------------------------------
for col in ['device_time_stamp', 'system_time_stamp']:
    combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce')

# ----------------------------------------------------------
# STEP 4: Remove Missing Values
# ----------------------------------------------------------
combined_df.dropna(subset=[
    'left_gaze_point_on_display_area',
    'right_gaze_point_on_display_area',
    'left_pupil_diameter',
    'right_pupil_diameter',
    'stimulus_id'
], inplace=True)

# ----------------------------------------------------------
# STEP 5: Ensure Numeric Format
# ----------------------------------------------------------
combined_df['left_pupil_diameter'] = pd.to_numeric(
    combined_df['left_pupil_diameter'], errors='coerce'
)
combined_df['right_pupil_diameter'] = pd.to_numeric(
    combined_df['right_pupil_diameter'], errors='coerce'
)

# ----------------------------------------------------------
# STEP 6: Feature Engineering
# ----------------------------------------------------------

# Average pupil size
combined_df['avg_pupil_diameter'] = (
    combined_df['left_pupil_diameter'] +
    combined_df['right_pupil_diameter']
) / 2

# Parse gaze coordinates from string to tuple
def parse_tuple(s):
    try:
        return tuple(map(float, s.strip('()').split(',')))
    except:
        return (np.nan, np.nan)

combined_df['left_gaze_xy'] = combined_df[
    'left_gaze_point_on_display_area'
].apply(parse_tuple)

combined_df['right_gaze_xy'] = combined_df[
    'right_gaze_point_on_display_area'
].apply(parse_tuple)

# Extract X and Y
combined_df[['left_gaze_x', 'left_gaze_y']] = pd.DataFrame(
    combined_df['left_gaze_xy'].tolist(), index=combined_df.index
)

combined_df[['right_gaze_x', 'right_gaze_y']] = pd.DataFrame(
    combined_df['right_gaze_xy'].tolist(), index=combined_df.index
)

# Average gaze position
combined_df['avg_gaze_x'] = (
    combined_df['left_gaze_x'] +
    combined_df['right_gaze_x']
) / 2

combined_df['avg_gaze_y'] = (
    combined_df['left_gaze_y'] +
    combined_df['right_gaze_y']
) / 2

# ----------------------------------------------------------
# STEP 7: Cleanup
# ----------------------------------------------------------
combined_df.drop(columns=[
    'left_gaze_xy',
    'right_gaze_xy'
], inplace=True)

combined_df.reset_index(drop=True, inplace=True)

# ----------------------------------------------------------
# STEP 8: Save Preprocessed Data
# ----------------------------------------------------------
output_path = "Data/preprocessed_gaze_data.csv"

combined_df.to_csv(output_path, index=False)

print("\n✅ Preprocessed dataset saved at:", output_path)
print("\n🔍 Preview:")
print(combined_df.head())