# Setup & Installation Guide

Complete guide to setting up the Stock & Options Backtesting Framework on your system.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Verification](#verification)
- [Common Issues](#common-issues)
- [Upgrading](#upgrading)
- [Uninstallation](#uninstallation)

---

## System Requirements

### Minimum Requirements

- **Python**: 3.9 or higher (tested up to Python 3.13)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Disk Space**: 500 MB for framework + cache
- **Internet**: Required for market data downloads
- **OS**: Windows, macOS, or Linux

### Recommended Specifications

- **Python**: 3.11 or 3.12 (best performance)
- **RAM**: 16 GB for large-scale backtests
- **Disk Space**: 2-5 GB for extensive caching
- **CPU**: Multi-core for parallel optimization

---

## Installation

### Step 1: Install Python

If you don't have Python installed:

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer, **check "Add Python to PATH"**
3. Verify: `python --version`

**macOS:**
```bash
# Using Homebrew
brew install python@3.11

# Verify
python3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Verify
python3 --version
```

### Step 2: Clone Repository

```bash
# Using Git
git clone https://github.com/your-username/trading.git
cd trading

# OR download ZIP from GitHub and extract
```

### Step 3: Create Virtual Environment

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# You should see (venv) in your prompt
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# You should see (venv) in your prompt
```

### Step 4: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# This will install:
# - yfinance (market data)
# - numpy (numerical computing)
# - pandas (data structures)
# - scipy (scientific computing)
# - plotly (visualization)
# - tqdm (progress bars)
```

### Step 5: Verify Installation

```bash
# Run verification script
python -c "import yfinance, numpy, pandas, scipy, plotly; print('All packages installed successfully!')"
```

You should see: `All packages installed successfully!`

---

## Configuration

### Data Cache Setup

The framework automatically creates a cache directory `.yfinance_cache/` for storing downloaded market data.

**Configure Cache Settings:**

Edit `yfinance_cache.py` if you want to change cache behavior:

```python
# Default settings
CACHE_DIR = ".yfinance_cache"  # Cache directory
CACHE_EXPIRY_HOURS = 24        # Hours before cache expires
```

**Manual Cache Management:**

```python
from yfinance_cache import clear_cache, get_cache

# Clear all cached data
clear_cache()

# Get cache instance with custom settings
cache = get_cache(cache_dir="custom_cache", cache_expiry_hours=48)
```

### Environment Variables (Optional)

For live trading with Alpaca, set environment variables:

**Windows:**
```bash
set ALPACA_API_KEY=your_api_key_here
set ALPACA_SECRET_KEY=your_secret_key_here
```

**macOS/Linux:**
```bash
export ALPACA_API_KEY=your_api_key_here
export ALPACA_SECRET_KEY=your_secret_key_here
```

**Persistent Setup (Recommended):**

Create a `.env` file:
```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Load with:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ALPACA_API_KEY')
```

**Note:** `.env` files are gitignored to protect your keys.

---

## Verification

### Test 1: Basic Imports

```python
# test_imports.py
from functions import greeks, option_price
from fourier import run_fourier_backtest, plot_stock
from technical_retrievals import get_rates, get_historical_volatility
from yfinance_cache import download_cached

print("✓ All modules imported successfully")
```

Run: `python test_imports.py`

### Test 2: Data Download

```python
# test_download.py
from yfinance_cache import download_cached

data = download_cached('AAPL', '2025-01-01', '2025-01-31')
print(f"✓ Downloaded {len(data)} days of AAPL data")
```

Run: `python test_download.py`

### Test 3: Greeks Calculation

```python
# test_greeks.py
from functions import greeks

result = greeks('AAPL', '2026-01-16', 150.0, 'call', status=True, silent=True)
print(f"✓ Greeks calculated: Delta={result['delta']:.4f}")
print(f"✓ Buy Score: {result['buy_score']}/100")
```

Run: `python test_greeks.py`

### Test 4: Simple Backtest

```python
# test_backtest.py
from fourier import run_fourier_backtest

results = run_fourier_backtest(
    ticker='SPY',
    start_date='2025-11-01',
    end_date='2025-12-01',
    n_harmonics=10,
    smoothing_sigma=2.0,
    overbought_threshold=5.0,
    oversold_threshold=-5.0,
    initial_capital=10000.0
)

print(f"✓ Backtest completed: Return = {results['backtest']['total_return']:.2f}%")
```

Run: `python test_backtest.py`

---

## Common Issues

### Issue 1: `ModuleNotFoundError`

**Error:**
```
ModuleNotFoundError: No module named 'yfinance'
```

**Solution:**
```bash
# Make sure virtual environment is activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Issue 2: `ImportError: DLL load failed`

**Error (Windows):**
```
ImportError: DLL load failed while importing _multiarray_umath
```

**Solution:**
```bash
# Reinstall numpy
pip uninstall numpy
pip install numpy --no-cache-dir

# Or install specific version
pip install numpy==1.24.0
```

### Issue 3: Permission Errors

**Error:**
```
PermissionError: [Errno 13] Permission denied: '.yfinance_cache'
```

**Solution:**
```bash
# Check directory permissions
# Windows: Right-click folder → Properties → Security
# macOS/Linux:
chmod -R 755 .yfinance_cache

# Or delete and recreate
rm -rf .yfinance_cache
```

### Issue 4: Rate Limiting

**Error:**
```
HTTPError: 429 Too Many Requests
```

**Solution:**
- Use caching (framework does this automatically)
- Add delays between requests
- Wait a few minutes and retry

### Issue 5: Slow Downloads

**Problem:** Data downloads taking too long

**Solutions:**
1. Use smaller date ranges
2. Pre-cache data: `download_cached(ticker, start, end)`
3. Use larger tick sizes: `tick_size='1d'` instead of `'1h'`
4. Check internet connection speed

### Issue 6: Memory Errors

**Error:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. Reduce date range
2. Use fewer harmonics
3. Close other applications
4. Upgrade RAM if possible

### Issue 7: Plotly Not Showing Charts

**Problem:** `fig.show()` doesn't open browser

**Solutions:**
```python
# Option 1: Specify renderer
import plotly.io as pio
pio.renderers.default = 'browser'

# Option 2: Save to HTML
fig.write_html('chart.html')

# Option 3: Use different renderer
pio.renderers.default = 'notebook'  # For Jupyter
```

---

## Upgrading

### Update Framework

```bash
# Activate virtual environment
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt
```

### Update Individual Packages

```bash
# Update yfinance
pip install --upgrade yfinance

# Update all packages
pip install --upgrade yfinance numpy pandas scipy plotly tqdm

# Or update everything
pip list --outdated
pip install --upgrade [package-name]
```

### Clear Old Cache

```bash
# Python
python -c "from yfinance_cache import clear_cache; clear_cache()"

# Or manually delete
# Windows: rmdir /s .yfinance_cache
# macOS/Linux: rm -rf .yfinance_cache
```

---

## Uninstallation

### Remove Framework

```bash
# Deactivate virtual environment
deactivate

# Delete project directory
cd ..
# Windows: rmdir /s trading
# macOS/Linux: rm -rf trading
```

### Keep Data, Remove Environment

```bash
# Just delete virtual environment
# Windows: rmdir /s venv
# macOS/Linux: rm -rf venv

# Recreate later:
# python -m venv venv
# pip install -r requirements.txt
```

---

## Advanced Setup

### Jupyter Notebook Integration

```bash
# Install Jupyter
pip install jupyter notebook ipykernel

# Add kernel
python -m ipykernel install --user --name=trading

# Start Jupyter
jupyter notebook
```

### IDE Configuration

**Visual Studio Code:**
1. Install Python extension
2. Select interpreter: Ctrl+Shift+P → "Python: Select Interpreter"
3. Choose `venv/Scripts/python.exe` (Windows) or `venv/bin/python` (macOS/Linux)

**PyCharm:**
1. File → Settings → Project → Python Interpreter
2. Add → Existing Environment
3. Select `venv/Scripts/python.exe` or `venv/bin/python`

### Multi-Environment Setup

For testing different Python versions:

```bash
# Python 3.9
python3.9 -m venv venv39
source venv39/bin/activate
pip install -r requirements.txt

# Python 3.11
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
```

---

## Performance Optimization

### Pre-cache Common Data

```bash
# Create cache_data.py
cat > cache_data.py << EOF
from yfinance_cache import download_cached

tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
for ticker in tickers:
    print(f"Caching {ticker}...")
    download_cached(ticker, '2024-01-01', '2025-12-31')
print("Caching complete!")
EOF

python cache_data.py
```

### Parallel Backtesting Setup

```bash
# Install multiprocessing support (usually included)
pip install multiprocess
```

### GPU Acceleration (Advanced)

```bash
# For CUDA-enabled GPUs
pip install cupy-cuda11x  # Replace 11x with your CUDA version

# For AMD GPUs
pip install pyopencl
```

---

## Development Setup

For contributing to the framework:

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black *.py

# Lint code
flake8 *.py
```

---

## Support & Resources

### Get Help

1. Check [DOCUMENTATION.md](DOCUMENTATION.md) for API reference
2. Review [README.md](README.md) for examples
3. Search existing GitHub issues
4. Create new issue with:
   - Python version: `python --version`
   - OS: Windows/macOS/Linux
   - Error message (full traceback)
   - Minimal code to reproduce

### Useful Links

- [Python Official Docs](https://docs.python.org/3/)
- [YFinance Documentation](https://pypi.org/project/yfinance/)
- [Plotly Documentation](https://plotly.com/python/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## Next Steps

After successful setup:

1. Run [example_stock_backtest.py](example_stock_backtest.py)
2. Run [example_options_backtest.py](example_options_backtest.py)
3. Run [example_greeks_analysis.py](example_greeks_analysis.py)
4. Review [EXPERIMENTAL_IDEAS.md](EXPERIMENTAL_IDEAS.md) for advanced topics
5. Start building your own strategies!

---

**Setup complete! Happy backtesting! 📈**
