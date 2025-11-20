import pandas as pd


class BBHExemplar:
    def __init__(self, data_df: pd.DataFrame, num_samples: int | None = None) -> None:
        if isinstance(num_samples, int) and num_samples > len(data_df):
            raise ValueError(
                f"num_samples ({num_samples}) cannot be greater than the number of samples in data_df ({len(data_df)})."
            )

        self._data_df = data_df
        self._num_samples = num_samples or min(len(data_df), 5)

    def __call__(self) -> str:
        return self._format()

    def _format(self) -> str:
        output_str = ""
        df_sample = self._data_df.sample(self._num_samples)
        for i, (_, row) in enumerate(df_sample.iterrows()):
            output_str += f"\nExample: {i + 1}\n{row['input']}\n"
        return output_str
