# automate_NamaKamu.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

class TitanicPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.le_sex = LabelEncoder()
        self.columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age']
        
    def fit_transform(self, df, save_path='preprocessing_objects'):
        """Preprocess raw Titanic data"""
        # Buat copy
        df_processed = df.copy()
        
        # 1. Handle Missing Values
        df_processed['Age'] = df_processed.groupby(['Pclass', 'Sex'])['Age'].transform(
            lambda x: x.fillna(x.median()))
        df_processed['Cabin'] = df_processed['Cabin'].fillna('Unknown')
        df_processed['Embarked'] = df_processed['Embarked'].fillna(
            df_processed['Embarked'].mode()[0])
        
        # 2. Feature Engineering
        df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
        
        df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Rare', 'Countess': 'Rare', 'Ms': 'Rare', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Rare',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        df_processed['Title'] = df_processed['Title'].map(title_mapping)
        
        df_processed['CabinLetter'] = df_processed['Cabin'].str[0]
        df_processed['CabinLetter'] = df_processed['CabinLetter'].fillna('Unknown')
        
        df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                         bins=[0, 12, 18, 35, 60, 100],
                                         labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
        
        # 3. Encoding
        df_processed['Sex_encoded'] = self.le_sex.fit_transform(df_processed['Sex'])
        
        categorical_cols = ['Pclass', 'Embarked', 'Title', 'CabinLetter', 'AgeGroup']
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        
        # 4. Drop columns
        df_processed = df_processed.drop(columns=self.columns_to_drop + ['Sex'])
        
        # 5. Separate features and target
        X = df_processed.drop('Survived', axis=1)
        y = df_processed['Survived']
        
        # 6. Scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Save preprocessing objects
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        with open(f'{save_path}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f'{save_path}/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.le_sex, f)
        
        return X_scaled, y, X.columns.tolist()
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        df_processed = df.copy()
        
        # Apply same transformations (without fitting)
        df_processed['Age'] = df_processed.groupby(['Pclass', 'Sex'])['Age'].transform(
            lambda x: x.fillna(x.median()))
        df_processed['Cabin'] = df_processed['Cabin'].fillna('Unknown')
        df_processed['Embarked'] = df_processed['Embarked'].fillna('S')
        
        df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
        
        df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Rare', 'Countess': 'Rare', 'Ms': 'Rare', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Rare',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        df_processed['Title'] = df_processed['Title'].map(title_mapping)
        
        df_processed['CabinLetter'] = df_processed['Cabin'].str[0]
        df_processed['CabinLetter'] = df_processed['CabinLetter'].fillna('Unknown')
        
        df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                         bins=[0, 12, 18, 35, 60, 100],
                                         labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
        
        df_processed['Sex_encoded'] = self.le_sex.transform(df_processed['Sex'])
        
        categorical_cols = ['Pclass', 'Embarked', 'Title', 'CabinLetter', 'AgeGroup']
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        
        df_processed = df_processed.drop(columns=self.columns_to_drop + ['Sex'])
        
        # Ensure all columns exist (for new data)
        expected_columns = self.get_expected_columns()
        for col in expected_columns:
            if col not in df_processed.columns and col != 'Survived':
                df_processed[col] = 0
        
        X = df_processed[expected_columns].drop('Survived', axis=1, errors='ignore')
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def get_expected_columns(self):
        """Return expected columns after preprocessing"""
        # This should be saved during fit
        return ['SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'Sex_encoded',
                'Pclass_2', 'Pclass_3', 'Embarked_Q', 'Embarked_S',
                'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
                'CabinLetter_B', 'CabinLetter_C', 'CabinLetter_D', 'CabinLetter_E',
                'CabinLetter_F', 'CabinLetter_G', 'CabinLetter_T', 'CabinLetter_Unknown',
                'AgeGroup_Teen', 'AgeGroup_Young Adult', 'AgeGroup_Adult', 'AgeGroup_Senior']

# Contoh penggunaan
if __name__ == "__main__":
    import os
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # CORRECT PATH (relative from preprocessing folder)
    data_path = os.path.join(current_dir, '../titanic_raw/train.csv')
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Preprocess
    preprocessor = TitanicPreprocessor()
    X_scaled, y, feature_names = preprocessor.fit_transform(data)
    
    print(f"Preprocessing selesai!")
    print(f"X shape: {X_scaled.shape}")
    print(f"y shape: {y.shape}")
    print(f"Jumlah fitur: {len(feature_names)}")
    
    # Simpan data preprocessed
    output_dir = os.path.join(current_dir, '../titanic_preprocessed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, 'X_scaled.npy'), X_scaled)
    np.save(os.path.join(output_dir, 'y.npy'), y.values)
    with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(f"\nData tersimpan di: {output_dir}/")
    
    # Cek file tersimpan
    print("\nFiles created:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")