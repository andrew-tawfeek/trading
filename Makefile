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