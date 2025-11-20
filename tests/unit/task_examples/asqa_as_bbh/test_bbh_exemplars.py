from textwrap import dedent

import pandas as pd
import pytest

from task_examples.asqa_as_bbh.bbh_exemplar import BBHExemplar


def test__bbh_exemplar__raises__when_num_samples_larger_than_df():
    data_df = pd.DataFrame({"input": ["Hey", "Hello"]})
    with pytest.raises(ValueError):
        _ = BBHExemplar(data_df, num_samples=5)


def test__bbh_exemplar__formats_samples():
    data_df = pd.DataFrame({"input": ["Hey", "Hello"]})

    exemplar_formatter = BBHExemplar(data_df)
    exemplars = exemplar_formatter()

    exemplars_template = dedent("""
    Example: 1
    {GREET_1}
    
    Example: 2
    {GREET_2}
    """)

    assert (exemplars == exemplars_template.format(GREET_1="Hello", GREET_2="Hey")) or (
        exemplars == exemplars_template.format(GREET_1="Hey", GREET_2="Hello")
    )
