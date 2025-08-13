# PLCNet Project Environment Setup

This project uses a Python virtual environment to manage dependencies cleanly.

## Quick Start

1. **Activate the environment:**
   ```bash
   source plcnet_env/bin/activate
   ```

2. **Run the PLCNet application:**
   ```bash
   python PLCNet_V3.py
   ```

3. **Deactivate when done:**
   ```bash
   deactivate
   ```

## Environment Details

- **Python Version:** 3.11.4
- **Virtual Environment:** `plcnet_env/`
- **Dependencies:** Listed in `requirements.txt`

## Key Dependencies

- **TensorFlow 2.15.1:** Deep learning framework for LSTM/CNN models
- **scikit-learn 1.7.1:** Machine learning utilities and metrics
- **pandas 2.3.1:** Data manipulation and analysis
- **numpy 1.26.4:** Numerical computing
- **matplotlib 3.10.5:** Plotting and visualization

## Recreating the Environment

If you need to recreate the environment from scratch:

```bash
# Remove existing environment
rm -rf plcnet_env

# Create new virtual environment
python3 -m venv plcnet_env

# Activate environment
source plcnet_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Troubleshooting

- **Import errors:** Make sure the virtual environment is activated
- **Missing packages:** Run `pip install -r requirements.txt` again
- **TensorFlow issues:** Ensure you're using Python 3.8-3.11 (we're using 3.11.4)
