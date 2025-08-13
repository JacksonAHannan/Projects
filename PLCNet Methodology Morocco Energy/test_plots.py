#!/usr/bin/env python3
"""
Test script to evaluate PLCNet using pre-trained model.
This loads the existing dualpath_lstm_cnn_model.keras file instead of retraining.
"""

import sys
import os
import numpy as np

# Add the project directory to path
project_dir = "/Users/jacksonhannan/Desktop/Python Projects/PCLNet Attempt Morocco Energy"
sys.path.insert(0, project_dir)
os.chdir(project_dir)

# Import the main PLCNet module
try:
    import PLCNet_V1 as plcnet
    import tensorflow as tf
    
    # Override config to disable UI and enable plotting
    plcnet.UI_ENABLE = False
    plcnet.PLOT_RESULTS = True
    
    print("🧪 Running PLCNet evaluation with pre-trained model...")
    print(f"PLOT_RESULTS: {plcnet.PLOT_RESULTS}")
    print(f"UI_ENABLE: {plcnet.UI_ENABLE}")
    print("-" * 50)
    
    # Load and prepare data (same as training)
    print("Loading and preparing data...")
    df = plcnet.load_and_merge()
    if plcnet.SAVE_MERGED_CSV:
        out_path = os.path.join(plcnet.SCRIPT_DIR, "Energy_Data_Morocco", "merged_energy_data.csv")
        df.to_csv(out_path, index=False)
        print(f"Merged data saved to {out_path} (rows={len(df)})")

    print("Preparing data sequences ...")
    X_train, y_train, X_test, y_test, scaler, target_index = plcnet.prepare_data(df)
    print(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}, Features: {X_train.shape[-1]}")
    
    # Load the pre-trained model
    model_path = os.path.join(plcnet.SCRIPT_DIR, "dualpath_lstm_cnn_model.keras")
    
    if os.path.exists(model_path):
        print(f"📁 Loading pre-trained model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        model.summary()
        
        # Evaluate the model on test data
        print("\n🔍 Evaluating model performance...")
        results = plcnet.evaluate_and_report(model, X_test, y_test, y_train)
        
        # Display plots
        print(f"Debug: PLOT_RESULTS = {plcnet.PLOT_RESULTS}")
        print(f"Debug: About to attempt plotting...")
        
        if plcnet.PLOT_RESULTS:
            print("📈 Displaying plots...")
            plcnet.plot_model_comparisons(results, y_test, max_samples=500)
        else:
            print("Plotting disabled (PLOT_RESULTS = False)")
        
        print("\n✅ Evaluation completed successfully!")
        if "metrics" in results:
            print("\n📊 Final Metrics:")
            for model_name, metrics in results["metrics"].items():
                print(f"  {model_name}:")
                print(f"    • MAE (Mean Absolute Error): {metrics['MAE']:.4f}")
                print(f"    • RMSE (Root Mean Square Error): {metrics['RMSE']:.4f}")
                print(f"    • MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.2f}%")
                print(f"    • SMAPE (Symmetric Mean Absolute Percentage Error): {metrics['SMAPE']:.2f}%")
                print(f"    • R² (Coefficient of Determination): {metrics['R2']:.4f}")
                if 'MASE' in metrics:
                    print(f"    • MASE (Mean Absolute Scaled Error): {metrics['MASE']:.4f}")
                print()
    
    else:
        print(f"❌ Pre-trained model not found at {model_path}")
        print("Please run the training first to generate the model file.")
        print("You can do this by running: python PLCNet_V1.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the virtual environment is activated:")
    print("  source plcnet_env/bin/activate")
    
except Exception as e:
    print(f"❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()
