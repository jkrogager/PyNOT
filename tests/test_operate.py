from dataclasses import dataclass
from typing import Any
import numpy as np
import pytest
import os

from pynot.images import FitsImage
from pynot.operations import perform_operation

@dataclass
class InputPars:
    sequence: str
    variables: Any
    result: Any = None
    raises: Any = None


def basic_test_image(imsize=(100, 100), data=0, error=0, mask=False):
    data_img = np.zeros(imsize) + data
    err_img = np.zeros(imsize) + error
    if mask:
        mask_img = np.ones(imsize, dtype=bool)
    else:
        mask_img = np.zeros(imsize, dtype=bool)
    return FitsImage(data=data_img, error=err_img, mask=mask_img)


operate_inputs = [
    InputPars(sequence='import sys',
              variables={},
              result=None),
    InputPars(sequence='4 + 6*2',
              variables={},
              result=16),
    InputPars(sequence='a + 1',
              variables={},
              result=None, raises=Exception),
    InputPars(sequence='a + 1',
              variables={'a': 5},
              result=6),
    InputPars(sequence='a + 1',
              variables={'a': basic_test_image()},
              result={'data': 1, 'error': 0}),
    InputPars(sequence='a + b',
              variables={'a': basic_test_image(), 'b': basic_test_image()},
              result={'data': 0, 'error': 0}),
    InputPars(sequence='a + b',
              variables={'a': basic_test_image(error=1), 'b': basic_test_image(error=1)},
              result={'data': 0, 'error': np.sqrt(2)}),
    InputPars(sequence='a + b',
              variables={'a': basic_test_image(error=1), 'b': basic_test_image(imsize=(200, 200))},
              raises=ValueError),
    InputPars(sequence='a + b',
              variables={'a': basic_test_image(mask=True), 'b': basic_test_image(mask=False)},
              result={'data': 0, 'mask': 1}),
]

@pytest.mark.parametrize("pars", operate_inputs)
def test_operation(pars):
    if pars.raises:
        with pytest.raises(pars.raises):
            perform_operation(pars.sequence, pars.variables, output='output.fits')
    else:
        result = perform_operation(pars.sequence, pars.variables, output='output.fits')
        if isinstance(pars.result, dict):
            for attribute, value in pars.result.items():
                np.testing.assert_approx_equal(
                    np.mean(result.__getattribute__(attribute)),
                    value)
            assert os.path.exists('output.fits')
        else:
            assert result == pars.result
    if os.path.exists('output.fits'):
        os.remove('output.fits')


def test_prepare_variables():
    # (args_list)
    pass
