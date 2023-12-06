from typing import List, Any, Union, Optional

import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def to_tensor(input: Any, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long) -> Tensor:
    """Convert input to torch tensor
    
    Parameters
    ----------
    input : list of int, for batch of int sequence
        Sequence or batch of token ids
    padding_value : int, optional
        Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    dtype : :class:`torch.dtype`, optional
        :class:`torch.dtype` of output tensor

    Returns
    -------

    """
    if torch.jit.isinstance(input, List[Union[int, float]]): # noqa
        return torch.tensor(input, dtype=torch.long)
    elif torch.jit.isinstance(input, List[List[Union[int, float]]]): # noqa
        if padding_value is None:
            output = torch.tensor(input, dtype=dtype)
            return output
        else:
            output = pad_sequence(
                [torch.tensor(ids, dtype=dtype) for ids in input],
                batch_first=True,
                padding_value=float(padding_value)
            )
            return output
    else:
        raise TypeError("Input type not supported")
    
def sample_subset(df: pd.DataFrame, size:int, label_col:str='label'):
    """Generate subset from given dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        entire data to be sub-generate
    size : int
        total size for subset. Resulted dataframe will have the inferior closest to this size
    label_col : str, optional, default = 'label'
        name of label column
    Returns
    -------
    subset : pd.DataFrame
        balanced subset
    """
    if size < 0 :
        return df
    
    nb_class = df[label_col].unique().size
    return df.groupby(label_col).apply(lambda x: x.sample(size//nb_class)).reset_index(drop=True)

def convert_utf8(df: pd.DataFrame):
    """Auto convert numpy columns into list columns

    Parameters
    ----------
    df : pd.DataFrame
        entire data

    Returns
    -------
    df : pd.DataFrame
        formatted data
    """
    for c in df.columns:
        if isinstance(df.loc[0, c], bytes):
            df[c] = df[c].str.decode('utf-8')
    return df