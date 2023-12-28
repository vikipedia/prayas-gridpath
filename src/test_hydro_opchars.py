import pytest
import hydro_opchars
import common
import pandas as pd
import os

def get_dataframe():
    header = "balancing_type_project horizon period average_power_fraction min_power_fraction max_power_fraction"
    columns = header.split()
    btp =  ["month"]*12
    horizon = list(range(202404, 202413)) + list(range(202401, 202404))
    period = [2025]*12
    average_power_fraction = [0.5]*12
    min_power_fraction = [0]*12
    max_power_fraction = [1]*12
    data = [btp,
            horizon,
            period,
            average_power_fraction,
            min_power_fraction,
            max_power_fraction]
    return pd.DataFrame(dict(zip(columns, data)))


def jitter(df, value=0.06):
    df = df.copy()
    for c in hydro_opchars.get_power_fraction_cols():
        df[c] = df[c] + value

    return df


def test_swap_original_cols_from_file(monkeypatch, tmp_path):
    allcols = ["balancing_type_project", "horizon", "period",
            "average_power_fraction", "min_power_fraction", "max_power_fraction"]
    filecols = hydro_opchars.get_power_fraction_file_cols()
    cols = hydro_opchars.get_power_fraction_cols()

    csvpath = tmp_path / "filedata.csv"
    monkeypatch.setattr(common,
                        "get_subscenario_csvpath", lambda *args: csvpath)
    dbdf = get_dataframe()
    # CASE 1 file was not there initially
    with pytest.raises(Exception):
        df = hydro_opchars.swap_original_cols_from_file(dbdf, "project", 1, "month", "csv_location", "description")
        

    # CASE 3 file was there but no reuired columns were present or columns were empty
    dbdf = get_dataframe()
    csvpath = tmp_path / "filedata1.csv"
    dbdf.to_csv(csvpath, index=False)
    df2 = hydro_opchars.swap_original_cols_from_file(dbdf, "project", 1, "month", "csv_location", "description")
    filedf = pd.read_csv(csvpath)
    
    for d_c, f_c in zip(cols, filecols):
        assert f_c in filedf.columns
        assert filedf[f_c].values == pytest.approx(dbdf[d_c].values)
    for d_c, f_c in zip(cols, filecols):
        assert f_c in filedf.columns
        assert filedf[d_c].values == pytest.approx(filedf[f_c].values)
    for c in allcols:
        assert c in filedf.columns
        
    for c in allcols:
        assert c in filedf.columns
    
    dbdf_jittered = jitter(dbdf)
    # CASE 2 file was there but it had the required columns
    df1 = hydro_opchars.swap_original_cols_from_file(dbdf_jittered, "project", 1, "month", "csv_location", "description") 
    for d_c, f_c in zip(cols, filecols):
        assert df1[d_c].values == pytest.approx(filedf[f_c].values)

    filedf = pd.read_csv(csvpath)
    for d_c, f_c in zip(cols, filecols):
        ## this loop makes sure that calling swap twice will not change values of filecols
        assert f_c in filedf.columns
        assert filedf[f_c].values == pytest.approx(dbdf[d_c].values)
