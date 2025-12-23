.PHONY: all clean setup

PYTHON := /Users/atawfeek/GitHub\ Repos/trading/venv/bin/python
START_DATE := 2025-11-01
END_DATE := 2025-12-01
TICK_SIZE := 30m
LOG_DIR := logs

TICKERS := TSLA INTC AAPL NVDA MU NFLX AMZN NKE META CCL

all: setup $(TICKERS)

$(TICKERS): setup
	@echo "Running $@ parameter search..."
	@$(PYTHON) -u -c "from experiments import fourier_options_parameter_search; \
	fourier_options_parameter_search('$@', '$(START_DATE)', '$(END_DATE)', tick_size='$(TICK_SIZE)')" \
	> $(LOG_DIR)/$@_$(shell date +%Y%m%d_%H%M%S).log 2>&1
	@echo "$@ complete!"

setup:
	@mkdir -p $(LOG_DIR)

clean:
	@rm -f $(LOG_DIR)/*.log

# #################################



# # Run all tickers sequentially (AAPL → NVDA → SPY → TSLA → MSFT)
# make all

# # Run specific ticker
# make AAPL

# # Run multiple specific tickers
# make AAPL NVDA SPY

# # Run all in parallel (5 at once - faster but resource-intensive)
# make -j5 all

# # Customize parameters on-the-fly
# make START_DATE=2025-10-01 END_DATE=2025-11-15 TICK_SIZE=1h all


# #################################


# # View all logs
# ls -lh logs/

# # Watch latest log in real-time
# tail -f logs/AAPL_*.log

# # Clean all logs
# make clean

# # View specific ticker's latest log
# ls -t logs/AAPL_*.log | head -1 | xargs cat


# ###########################


# # Create new session named 'trading'
# tmux new -s trading

# # Navigate to your directory
# cd /Users/atawfeek/GitHub\ Repos/trading

# # Run your makefile task
# make all

# # Detach and let it run: Press Ctrl+b then d

# #############################

# # List all sessions
# tmux ls

# # Attach to existing session
# tmux attach -t trading

# # Create new window in session (Ctrl+b then c)
# # Switch between windows (Ctrl+b then n or p)

# # Split panes horizontally (Ctrl+b then ")
# # Split panes vertically (Ctrl+b then %)

# # Kill session when done
# tmux kill-session -t trading


# #####################

# ## Pro workflow,, starting with split panes...

# tmux new -s trading
# cd /Users/atawfeek/GitHub\ Repos/trading

# # Split horizontally (Ctrl+b then ")
# # Top pane: run make
# make all

# # Bottom pane: monitor logs
# tail -f logs/*.log


# ##########################

# Quick reference:

# Run all experiments	make all
# Run in parallel	make -j5 all
# Run in tmux background	tmux new -s trading -d 'make all'
# Monitor progress	tail -f logs/*.log
# Attach to tmux	tmux attach -t trading
# Detach from tmux	Ctrl+b then d
# Kill tmux session	tmux kill-session -t trading
# Clean logs	make clean

####################################


# Live Fourier Model environment
.PHONY: run-model setup-fourier clean-fourier

FOURIER_LOG_DIR := fourier_logs

# Model Parameters (adjustable via command line)
MODEL_TICKER ?= AAPL
N_HARMONICS ?= 18
SMOOTHING_SIGMA ?= 0.0
OVERBOUGHT_THRESHOLD ?= 9.0
OVERSOLD_THRESHOLD ?= -8.0
TICK_SIZE ?= 30m
LOOKBACK_PERIOD ?= 1mo
UPDATE_INTERVAL ?= 60
CONTRACTS_PER_TRADE ?= 1
MAX_POSITIONS ?= 2
STOPLOSS_PERCENT ?= 50.0
TAKEPROFIT_PERCENT ?= 50.0
DAYS_TO_EXPIRY ?= 30
OTM_PERCENT ?= 2.0
INITIAL_CAPITAL ?= 10000.0

setup-fourier:
	@mkdir -p $(FOURIER_LOG_DIR)

run-model: setup-fourier
	@echo "Starting Live Fourier Model for $(MODEL_TICKER)..."
	@echo "Logs will be saved to $(FOURIER_LOG_DIR)/$(MODEL_TICKER)_$(shell date +%Y%m%d_%H%M%S).log"
	@$(PYTHON) -u -c "from LiveFourier_Model import run_model; \
	run_model( \
		ticker='$(MODEL_TICKER)', \
		n_harmonics=$(N_HARMONICS), \
		smoothing_sigma=$(SMOOTHING_SIGMA), \
		overbought_threshold=$(OVERBOUGHT_THRESHOLD), \
		oversold_threshold=$(OVERSOLD_THRESHOLD), \
		tick_size='$(TICK_SIZE)', \
		lookback_period='$(LOOKBACK_PERIOD)', \
		update_interval=$(UPDATE_INTERVAL), \
		contracts_per_trade=$(CONTRACTS_PER_TRADE), \
		max_positions=$(MAX_POSITIONS), \
		stoploss_percent=$(STOPLOSS_PERCENT), \
		takeprofit_percent=$(TAKEPROFIT_PERCENT), \
		days_to_expiry=$(DAYS_TO_EXPIRY), \
		otm_percent=$(OTM_PERCENT), \
		initial_capital=$(INITIAL_CAPITAL) \
	)" > $(FOURIER_LOG_DIR)/$(MODEL_TICKER)_$(shell date +%Y%m%d_%H%M%S).log 2>&1

clean-fourier:
	@rm -rf $(FOURIER_LOG_DIR)/*.log
	@echo "Fourier logs cleaned!"


# source venv/bin/activate
# make MODEL_TICKER=NFLX N_HARMONICS=4 OVERBOUGHT_THRESHOLD=4 OVERSOLD_THRESHOLD=-3 STOPLOSS_PERCENT=10 TAKEPROFIT_PERCENT=10 run-model


# #################################

# # Run with all default parameters (AAPL with defaults)
# make run-model

# # Run with custom ticker
# make MODEL_TICKER=TSLA run-model

# # Customize multiple parameters
# make MODEL_TICKER=NVDA N_HARMONICS=25 INITIAL_CAPITAL=20000 run-model

# # More aggressive trading settings
# make MODEL_TICKER=AAPL \
#      MAX_POSITIONS=5 \
#      STOPLOSS_PERCENT=30 \
#      TAKEPROFIT_PERCENT=75 \
#      CONTRACTS_PER_TRADE=2 \
#      run-model

# # Higher frequency monitoring
# make MODEL_TICKER=TSLA \
#      TICK_SIZE=5m \
#      LOOKBACK_PERIOD=1d \
#      UPDATE_INTERVAL=30 \
#      run-model

# # Conservative settings with more harmonics
# make MODEL_TICKER=META \
#      N_HARMONICS=30 \
#      SMOOTHING_SIGMA=1.5 \
#      OVERBOUGHT_THRESHOLD=12 \
#      OVERSOLD_THRESHOLD=-10 \
#      run-model

# # Short-term options strategy
# make MODEL_TICKER=AAPL \
#      DAYS_TO_EXPIRY=7 \
#      OTM_PERCENT=1.0 \
#      run-model


# .PHONY: all clean setup

# // ...existing code...

# # Second set of commands - add new .PHONY
# .PHONY: backtest analyze report

# backtest:
#     @echo "Running backtest..."
#     @$(PYTHON) -u -c "from experiments import run_backtest; run_backtest()"

# analyze:
#     @echo "Analyzing results..."
#     @$(PYTHON) -u -c "from experiments import analyze_results; analyze_results()"

# report:
#     @echo "Generating report..."
#     @$(PYTHON) -u -c "from experiments import generate_report; generate_report()"



