# ==========================================================
# Eye Gaze Dataset Augmentation using SMOTE + Noise Injection
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# ----------------------------------------------------------
# STEP 1: Load Dataset
# ----------------------------------------------------------
input_path = "Data/augmented_gaze_data.csv"

df = pd.read_csv(input_path)

print("✅ Dataset loaded")
print("Original shape:", df.shape)

# ----------------------------------------------------------
# STEP 2: Feature Selection
# ----------------------------------------------------------
feature_cols = [
    'avg_x', 'avg_y',
    'std_x', 'std_y',
    'avg_pupil',
    'saccade_speed'
]

X = df[feature_cols]
y = df['Emotion']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------
# STEP 3: SMOTE Balancing
# ----------------------------------------------------------
print("\nApplying SMOTE...")

smote = SMOTE(
    random_state=42,
    k_neighbors=2  # safer for small classes
)

X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

# ----------------------------------------------------------
# STEP 4: Gaussian Noise Augmentation
# ----------------------------------------------------------
def add_gaussian_noise(X, noise_factor=0.02):
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

X_augmented = add_gaussian_noise(X_resampled)

# ----------------------------------------------------------
# STEP 5: Reconstruct Dataset
# ----------------------------------------------------------
X_final = scaler.inverse_transform(X_augmented)

balanced_df = pd.DataFrame(X_final, columns=feature_cols)

balanced_df['Emotion'] = le.inverse_transform(y_resampled)

# Add metadata placeholders
balanced_df['Subject'] = 'synthetic'
balanced_df['stimulus_id'] = 'synthetic'

# Reorder columns
balanced_df = balanced_df[
    ['Subject', 'Emotion', 'stimulus_id'] + feature_cols
]

# ----------------------------------------------------------
# STEP 6: Save Dataset
# ----------------------------------------------------------
output_path = "Data/augmented_gaze_data_balanced.csv"

balanced_df.to_csv(output_path, index=False)

print("\n✅ Balanced dataset saved at:", output_path)
print("New dataset shape:", balanced_df.shape)

print("\n📊 Class Distribution:")
print(balanced_df['Emotion'].value_counts())

print("\n🔍 Sample Data:")
print(balanced_df.head())