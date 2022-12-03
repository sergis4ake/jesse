import os
from typing import List, Dict
from multiprocessing import cpu_count

from jesse.modes.optimize_mode import Optimizer

os.environ['NUMEXPR_MAX_THREADS'] = str(cpu_count())


def optimize(
        user_config: dict,
        routes: List[Dict[str, str]],
        extra_routes: List[Dict[str, str]],
        start_date: str,
        finish_date: str,
        optimal_total: int,
        training_candles: dict,
        testing_candles: dict,
        csv: bool = False,
        json: bool = False,
        debug_mode: bool = False,
) -> None:
    """
    An isolated optimize() function which is perfect for using in research, and AI training
    such as our own optimization mode. Because of it being a pure function, it can be used
    in Python's multiprocessing without worrying about pickling issues.

    Example `config`:
    {
        'starting_balance': 5_000,
        'fee': 0.001,
        'type': 'futures',
        'futures_leverage': 3,
        'futures_leverage_mode': 'cross',
        'exchange': 'Binance',
        'warm_up_candles': 100
    }

    Example `route`:
    [{'exchange': 'Bybit USDT Perpetual', 'strategy': 'A1', 'symbol': 'BTC-USDT', 'timeframe': '1m'}]

    Example `extra_route`:
    [{'exchange': 'Bybit USDT Perpetual', 'symbol': 'BTC-USDT', 'timeframe': '3m'}]

    Example `candles`:
    {
        'Binance-BTC-USDT': {
            'exchange': 'Binance',
            'symbol': 'BTC-USDT',
            'candles': np.array([]),
        },
    }
    """
    _isolated_optimize(
        user_config=user_config,
        routes=routes,
        extra_routes=extra_routes,
        start_date=start_date,
        finish_date=finish_date,
        optimal_total=optimal_total,
        training_candles=training_candles,
        testing_candles=testing_candles,
        csv=csv,
        json=json,
        debug_mode=debug_mode,
    )


def _isolated_optimize(
        user_config: dict,
        routes: List[Dict[str, str]],
        extra_routes: List[Dict[str, str]],
        start_date: str,
        finish_date: str,
        optimal_total: int,
        training_candles: dict,
        testing_candles: dict,
        csv: bool = False,
        json: bool = False,
        debug_mode: bool = False,
) -> None:
    from jesse.services.validators import validate_routes
    from jesse.config import config as jesse_config, reset_config
    from jesse.routes import router
    from jesse.config import set_config
    import jesse.helpers as jh

    jesse_config['app']['trading_mode'] = 'optimize'
    jesse_config['app']['debug_mode'] = debug_mode
    cpu_cores = int(user_config['cpu_cores'])

    # inject (formatted) configuration values
    user_config = _format_config(user_config)
    set_config(user_config)

    # set routes
    router.initiate(routes, extra_routes)

    validate_routes(router)

    optimizer = Optimizer(
        training_candles=training_candles,  # type: ignore # (Jesse's type hinting is wrong)
        testing_candles=testing_candles,    # type: ignore # (Jesse's type hinting is wrong)
        optimal_total=optimal_total,
        cpu_cores=cpu_cores,
        csv=csv,
        export_json=json,
        start_date=start_date,
        finish_date=finish_date,
        user_config=user_config,
    )

    # start the process
    optimizer.run()


def _format_config(config: dict) -> dict:
    """
    Jesse's required format for user_config is different from what this function accepts. Hence,
    we need to format the config before passing it to Jesse.
    """
    exchange_config = {
        'balance': config['starting_balance'],
        'fee': config['fee'],
        'type': config['type'],
        'name': config['exchange'],
    }
    # futures exchange has different config, so:
    if exchange_config['type'] == 'futures':
        exchange_config['futures_leverage'] = config['futures_leverage']
        exchange_config['futures_leverage_mode'] = config['futures_leverage_mode']

    return {
        'exchange': exchange_config,
        'logging': {
            'balance_update': True,
            'order_cancellation': True,
            'order_execution': True,
            'order_submission': True,
            'position_closed': True,
            'position_increased': True,
            'position_opened': True,
            'position_reduced': True,
            'shorter_period_candles': False,
            'trading_candles': True
        },
        'warm_up_candles': config['warm_up_candles'],
        'warmup_candles_num': config['warm_up_candles'],
        'ratio': config['ratio'],
    }
