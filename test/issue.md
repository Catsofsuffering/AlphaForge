## 🐛 Bug Description

An issue has been identified in the parallel processing module where the `'_backend_args'` attribute is missing. The specific error occurs at:

```python
self._backend_args["maxtasksperchild"] = maxtasksperchild  # Error line
```

The `ParallelExt` class attempts to access the non-existent `_backend_args` attribute during initialization, leading to an `AttributeError`.

## 🕵️ To Reproduce

Reproduce the issue using the following steps (based on `stock_data_poc.py`):

```python

def test_stock_data():
    # Initialize QLib environment
    provider_uri = "./data/qlib_data/cn_data_rolling"
    qlib.init(provider_uri=provider_uri, region="cn")

    # Create StockData instance
    data = StockData(
        instrument="csi500",
        start_time="2010-01-01",
        end_time="2020-12-31",
        qlib_path=provider_uri
    )
```

Steps to reproduce:

1.  Configure the China A-share CSI500 dataset.
2.  Initialize QLib using the rolling data mode.
3.  Call `StockData` to load feature data.
4.  The attribute error occurs when parallel computation is triggered.

## 🧐 Expected Behavior

The correct behavior should be to inherit the backend parameter handling mechanism from the parent class `joblib.Parallel`, using the `_backend_kwargs` dictionary:

```python
self._backend_kwargs["maxtasksperchild"] = maxtasksperchild  # After fix
```

## 💡 Additional Notes

When I change the code, it has worked, but I don't know that is OK or not:

```diff
# paral.py line 24 modification
-               self._backend_args["maxtasksperchild"] = maxtasksperchild
+               self._backend_kwargs["maxtasksperchild"] = maxtasksperchild
```

Complete fixed code snippet:

```python
from joblib.parallel import Parallel, MultiprocessingBackend # Assuming MultiprocessingBackend is imported

class ParallelExt(Parallel):
    def __init__(self, *args, **kwargs):
        maxtasksperchild = kwargs.pop("maxtasksperchild", None)
        super(ParallelExt, self).__init__(*args, **kwargs)
        if isinstance(self._backend, MultiprocessingBackend) and maxtasksperchild is not None:
            self._backend_kwargs["maxtasksperchild"] = maxtasksperchild  # Fixed line
```

The stock_data.py is a script in the repo [AlphaForge](https://github.com/DulyHao/AlphaForge)

```python
from typing import List, Union, Optional, Tuple, Dict
from enum import IntEnum
import numpy as np
import pandas as pd
import torch

class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5
    
def change_to_raw_min(features):
    result = []
    for feature in features:
        if feature in ['$vwap']:
            result.append(f"$money/$volume")
        elif feature in ['$volume']:
            result.append(f"{feature}/100000")
            # result.append('$close')
        else:
            result.append(feature)
    return result

def change_to_raw(features):
    result = []
    for feature in features:
        if feature in ['$open','$close','$high','$low','$vwap']:
            result.append(f"{feature}*$factor")
        elif feature in ['$volume']:
            result.append(f"{feature}/$factor/1000000")
            # result.append('$close')
        else:
            raise ValueError(f"feature {feature} not supported")
    return result

class StockData:
    _qlib_initialized: bool = False

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda:0'),
                 raw:bool = False,
                 qlib_path:Union[str,Dict] = "",
                 freq:str = 'day',
                 ) -> None:
        self._init_qlib(qlib_path)
        self.df_bak = None
        self.raw = raw
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.freq = freq
        self.data, self._dates, self._stock_ids = self._get_data()


    @classmethod
    def _init_qlib(cls,qlib_path) -> None:
        if cls._qlib_initialized:
            return
        import qlib
        from qlib.config import REG_CN
        qlib.init(provider_uri=qlib_path, region=REG_CN)
        cls._qlib_initialized = True

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        cal: np.ndarray = D.calendar(freq=self.freq)
        start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
        end_index = cal.searchsorted(pd.Timestamp(self._end_time))  # type: ignore
        real_start_time = cal[start_index - self.max_backtrack_days]
        if cal[end_index] != pd.Timestamp(self._end_time):
            end_index -= 1
        # real_end_time = cal[min(end_index + self.max_future_days,len(cal)-1)]
        real_end_time = cal[end_index + self.max_future_days]
        result =  (QlibDataLoader(config=exprs,freq=self.freq)  # type: ignore
                .load(self._instrument, real_start_time, real_end_time))
        return result
    
    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        if self.raw and self.freq == 'day':
            features = change_to_raw(features)
        elif self.raw:
            features = change_to_raw_min(features)
        df = self._load_exprs(features)
        self.df_bak = df
        # print(df)
        df = df.stack().unstack(level=1)
        dates = df.index.levels[0]                                      # type: ignore
        stock_ids = df.columns
        values = df.values
        values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore
        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def add_data(self,data:torch.Tensor,dates:pd.Index):
        data = data.to(self.device)
        self.data = torch.cat([self.data,data],dim=0)
        self._dates = pd.Index(self._dates.append(dates))


    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
    
    
```

## Suggested Next Steps

1.  Apply the code modification using Code mode.
2.  Run `stock_data_poc.py` to verify the fix.
3.  Update QLib dependencies to ensure long-term compatibility.

-----

## Screenshot

No Screenshot

## Environment

**Note**: User could run `cd scripts && python collect_info.py all` under project directory to get system information
and paste them here directly.

 - Qlib version: 0.9.6
 - Python version: 3.10.17
 - OS (`Windows`, `Linux`, `MacOS`): Windows
 - joblib version: 1.5.0

## Additional Notes

<!-- Add any other information about the problem here. -->
