# Set matplotlib backend first to avoid Tkinter warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, 
                           f1_score, classification_report, confusion_matrix)
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
try:
    df = pd.read_csv('dataset_med.csv')
    print(f"Dataset loaded successfully with shape: {df.shape}")
    
    # Downsample for development (comment out for full run)
    df = df.sample(frac=0.1, random_state=42)
    print(f"Using subset of data with shape: {df.shape}")
    
except FileNotFoundError:
    print("Error: File 'dataset_med.csv' not found.")
    exit()

# Data preprocessing function
def preprocess_data(df, is_training=True):
    # Create categorical features
    if is_training:
        # For training data, create the bins and categories
        df['age_group'] = pd.cut(df['age'], bins=[0,40,60,80,100], 
                                labels=['0-40','41-60','61-80','80+'])
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,100], 
                                  labels=['Underweight','Normal','Overweight','Obese'])
    else:
        # For prediction data, manually assign categories based on the same bins
        df['age_group'] = pd.cut(df['age'], bins=[0,40,60,80,100], 
                                labels=['0-40','41-60','61-80','80+']).astype('object')
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,100], 
                                  labels=['Underweight','Normal','Overweight','Obese']).astype('object')
    
    # Drop problematic columns that might cause leakage
    cols_to_drop = ['id', 'diagnosis_date', 'end_treatment_date', 'country']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    return df

df_processed = preprocess_data(df.copy(), is_training=True)

# Define features and target
X = df_processed.drop('survived', axis=1)
y = df_processed['survived']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing
numerical_features = ['age', 'bmi', 'cholesterol_level']
categorical_features = ['gender', 'cancer_stage', 'family_history', 'smoking_status',
                       'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 
                       'treatment_type', 'age_group', 'bmi_category']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define modeling approaches with improved parameters
approaches = {
    # Approach 1: Balanced Random Forest
    'Balanced_RF': {
        'model': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', BalancedRandomForestClassifier(
                random_state=42,
                sampling_strategy='auto',
                n_jobs=2,
                class_weight='balanced_subsample'
            ))
        ]),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, 20, None],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2']
        }
    },
    
    # Approach 2: SMOTE + Random Forest
    'SMOTE_RF': {
        'model': ImbPipeline([
            ('preprocessor', preprocessor),
            ('sampling', SMOTE(random_state=42, sampling_strategy=0.5)),
            ('classifier', RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=2
            ))
        ]),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'sampling__k_neighbors': [3, 5, 7]
        }
    }
}

# Model evaluation with better CV strategy
results = {}
best_score = 0
best_model = None

print("\n=== Evaluating Different Approaches ===")
for approach_name, approach in approaches.items():
    print(f"\nEvaluating {approach_name}...")
    
    try:
        search = RandomizedSearchCV(
            approach['model'],
            approach['params'],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=2,
            verbose=3,
            n_iter=10,
            refit=True
        )
        
        search.fit(X_train, y_train)
        
        # Evaluate best model
        current_model = search.best_estimator_
        y_pred = current_model.predict(X_test)
        y_proba = current_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'best_params': search.best_params_,
            'model': current_model
        }
        
        # Print classification report
        print(f"\nClassification Report for {approach_name}:")
        print(classification_report(y_test, y_pred))
        
        # Plot and save confusion matrix
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {approach_name}')
        plt.savefig(f'confusion_matrix_{approach_name}.png')
        plt.close()
        
        results[approach_name] = metrics
        print(f"{approach_name} - ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model = current_model
            
    except Exception as e:
        print(f"Error in {approach_name}: {str(e)}")
        continue

# Results summary
print("\n=== Final Results ===")
results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df[['roc_auc', 'balanced_accuracy', 'f1_score']].sort_values('roc_auc', ascending=False))

# Feature Importance from best model
if best_model is not None:
    best_approach = results_df['roc_auc'].idxmax()
    print(f"\nBest approach: {best_approach} (ROC AUC: {best_score:.4f})")
    
    try:
        # Get feature names
        num_features = numerical_features.copy()
        cat_transformer = best_model.named_steps['preprocessor'].named_transformers_['cat']
        cat_features = cat_transformer.get_feature_names_out(categorical_features)
        all_features = np.concatenate([num_features, cat_features])
        
        # Get importances
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            importances = best_model.named_steps['classifier'].feature_importances_
        elif hasattr(best_model.named_steps['classifier'], 'estimators_'):
            importances = np.mean([est.feature_importances_ for est in 
                                 best_model.named_steps['classifier'].estimators_], axis=0)
        
        # Create importance dataframe
        feature_importance = pd.DataFrame({
            'feature': all_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot and save top 20 features
        plt.figure(figsize=(12,8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title(f'Top 20 Features - {best_approach}')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
    except Exception as e:
        print(f"Could not plot feature importances: {str(e)}")

    # Save the best model
    joblib.dump(best_model, 'best_lung_cancer_model.pkl')
    print("\nBest model saved as 'best_lung_cancer_model.pkl'")
else:
    print("\nNo successful models were trained.")

# Enhanced prediction function with formatted output
def predict_survival(patient_data, model_path='best_lung_cancer_model.pkl'):
    """
    Predict survival probability with clear, formatted output
    Args:
        patient_data: Dictionary containing patient features
        model_path: Path to saved model
    Returns:
        Dictionary with prediction results and status
    """
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Check for required numerical features
        missing_features = [f for f in numerical_features if f not in patient_data]
        if missing_features:
            return {
                "error": f"Missing numerical features: {', '.join(missing_features)}",
                "status": "error"
            }
        
        # Prepare input data - convert to DataFrame first
        patient_df = pd.DataFrame([patient_data])
        
        # Add derived features (age_group and bmi_category will be created in preprocess_data)
        patient_df = preprocess_data(patient_df, is_training=False)
        
        # Make prediction
        proba = model.predict_proba(patient_df)[0, 1]
        prediction = "Yes" if proba >= 0.5 else "No"
        confidence = "high" if abs(proba - 0.5) > 0.3 else "moderate" if abs(proba - 0.5) > 0.15 else "low"
        
        # Get top contributing features
        num_features = numerical_features.copy()
        cat_transformer = model.named_steps['preprocessor'].named_transformers_['cat']
        cat_features = cat_transformer.get_feature_names_out(categorical_features)
        all_features = np.concatenate([num_features, cat_features])
        
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        elif hasattr(model.named_steps['classifier'], 'estimators_'):
            importances = np.mean([est.feature_importances_ for est in 
                                 model.named_steps['classifier'].estimators_], axis=0)
        
        X_processed = model.named_steps['preprocessor'].transform(patient_df)
        
        # Create contribution dataframe
        contributions = pd.DataFrame({
            'feature': all_features,
            'importance': importances,
            'value': X_processed[0]
        }).sort_values('importance', ascending=False)
        
        # Get top 3 contributing factors
        top_factors = []
        for _, row in contributions.head(3).iterrows():
            if row['feature'] in numerical_features:
                factor_name = row['feature'].replace('_', ' ').title()
                if factor_name == 'Bmi':
                    factor_name = 'BMI'
                factor_value = f"{float(patient_data[row['feature']]):.1f}"
                top_factors.append(f"{factor_name}: {factor_value}")
            elif row['value'] == 1:
                feature_parts = row['feature'].split('_')
                factor_name = feature_parts[0].title()
                factor_value = ' '.join(feature_parts[1:]).title()
                top_factors.append(f"{factor_name}: {factor_value}")
        
        return {
            "prediction": prediction,
            "probability": float(proba),
            "confidence": confidence,
            "top_factors": top_factors,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

# Example usage
if best_model is not None:
    print("\n=== Example Prediction ===")
    example_patient = {
        'age': 60,
        'gender': 'Female',
        'cancer_stage': 'Stage II',
        'family_history': 'No',
        'smoking_status': 'Former smoker',
        'bmi': 24,
        'cholesterol_level': 180,
        'hypertension': 0,
        'asthma': 0,
        'cirrhosis': 0,
        'other_cancer': 0,
        'treatment_type': 'Surgery'
    }

    prediction = predict_survival(example_patient)
    
    if prediction.get('status') == 'success':
        print("\nPrediction Results:")
        print(f"Likely to Survive: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']} ({prediction['probability']:.1%})")
        print("\nTop Contributing Factors:")
        for i, factor in enumerate(prediction['top_factors'], 1):
            print(f"{i}. {factor}")
    else:
        print(f"\nPrediction failed: {prediction.get('error', 'Unknown error')}")