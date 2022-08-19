import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from otlang.sdk.syntax import Keyword, Positional, OTLType
from pp_exec_env.base_command import BaseCommand, Syntax


class FftCommand(BaseCommand):
    """
    Compute the 1-D discrete Fourier Transform.

    | fft signal fs=100 n=100
    """
    syntax = Syntax(
        [
            Positional("signal", required=True, otl_type=OTLType.TEXT),
            Keyword("fs", required=True, otl_type=OTLType.NUMERIC),
            Keyword("n", required=False, otl_type=OTLType.INTEGER),
        ],
    )
    use_timewindow = False  # Does not require time window arguments
    idempotent = True  # Does not invalidate cache

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_progress('Start fft command')
        column_name = self.get_arg("signal").value
        fs = self.get_arg("fs").value
        n = self.get_arg("n").value or df[column_name].values.shape[0]

        signal_f = fft(df[column_name].values, n=n)
        amplitude = np.abs(signal_f[0:n//2])
        freq = fftfreq(len(signal_f), 1 / fs)[:n//2]

        df = pd.DataFrame({"amplitude": amplitude,
                           "frequency": freq})

        self.log_progress('First part is complete.', stage=1, total_stages=1)
        return df
