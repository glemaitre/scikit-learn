import pytest
from joblib import Parallel

from numpy.testing import assert_array_equal

from sklearn._config import config_context, get_config
from sklearn.utils.fixes import delayed


def get_working_memory():
    return get_config()["working_memory"]


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("backend", ["loky", "threading", "multiprocessing"])
def test_configuration_passes_through_to_joblib(n_jobs, backend):
    # Tests that the global global configuration is passed to joblib jobs

    with config_context(working_memory=123):
        config = get_config()
        results = Parallel(n_jobs=n_jobs, backend=backend, pre_dispatch=1)(
            delayed(get_working_memory, config=config)() for _ in range(10)
        )

    assert_array_equal(results, [123] * 10)
