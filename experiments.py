from fourier import run_fourier_options_backtest


def fourier_options_parameter_search(ticker: str,
                                     start_date: str,
                                     end_date: str,
                                     tick_size: str = '1d',
                                     n_harmonics_range: list = list(range(1, 25)),
                                     smoothing_options: list = [0, 2, 4],
                                     overbought_range: list = list(range(2, 10)),
                                     oversold_range: list = list(range(-10, -2)),
                                     otm_percent_options: list =[2.0, 5.0, 10.0],
                                     stoploss_options: list =[10, 20, 30, 40, 50],
                                     takeprofit_options: list =[10, 20, 30, 40, 50]):

    best_return = -100
    best_params = None

    for n_harmonics in n_harmonics_range:
        for smoothing in smoothing_options:
            for overbought in overbought_range:
                for oversold in oversold_range:
                    for otm_percent in otm_percent_options:
                        for stoploss in stoploss_options:
                            for takeprofit in takeprofit_options:
                                results = run_fourier_options_backtest(
                                    ticker=ticker,
                                    start_date=start_date,
                                    end_date=end_date,
                                    n_harmonics=n_harmonics,
                                    smoothing_sigma=smoothing,
                                    overbought_threshold=overbought,
                                    oversold_threshold=oversold,
                                    contracts_per_trade=3,
                                    max_positions=20,
                                    tick_size=tick_size,
                                    stoploss_percent=stoploss,
                                    takeprofit_percent=takeprofit,
                                    days_to_expiry=30,
                                    otm_percent=otm_percent,
                                    initial_capital=10000,
                                    verbose = False
                                )

                                total_return = results['total_return']

                                if total_return > best_return:
                                    best_return = total_return
                                    best_params = {
                                        'n_harmonics': n_harmonics,
                                        'smoothing_sigma': smoothing,
                                        'overbought': overbought,
                                        'oversold': oversold,
                                        'otm_percent': otm_percent,
                                        'stoploss_percent': stoploss,
                                        'takeprofit_percent': takeprofit
                                    }
                                    print(f"New best: {best_return:.2f}% with {best_params}")