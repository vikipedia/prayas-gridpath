import availability
import pandas as pd
import numpy as np


def get_timepoint():
    return np.arange(200, 300)


def exogenous_results(*args):
    return pd.DataFrame({
        "timepoint": get_timepoint(),
        "availability_derate": np.ones(100),
        "project": ["test"]*100,
        "exogenous_availability_scenario_id": [1]*100,
        "stage_id": [1]*100,
        "hyb_stor_cap_availability_derate": np.zeros(100)
    }), pd.DataFrame()


def get_forced_outage():
    f = np.ones(100)
    f[:50] = 0
    print(np.sum(f), "X"*10)
    fo = pd.DataFrame({"timepoint": np.arange(299, 199, -1),
                       "fo": f})
    return fo.set_index('timepoint')['fo']


def test_get_exogenous_results(monkeypatch):
    monkeypatch.setattr(
        availability, "get_exogenous_results_", exogenous_results)

    fo = get_forced_outage()
    results, _ = availability.get_exogenous_results("sc1",
                                                    "sc2",
                                                    "sc3",
                                                    fo,
                                                    "project",
                                                    "db_path",
                                                    "mapfile")
    assert results['availability_derate'].sum() == 50
    assert np.sum(results['availability_derate'].values[:50]) == 50
    assert np.sum(results['availability_derate'].values[50:]) == 0
    results, _ = availability.get_exogenous_results("sc1",
                                                    "sc2",
                                                    "sc3",
                                                    None,
                                                    "project",
                                                    "db_path",
                                                    "mapfile")
    assert results['availability_derate'].sum() == 100
