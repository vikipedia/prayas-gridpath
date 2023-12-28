import common
import pandas as pd

def test_override_dataframe():
    df1 = pd.DataFrame(dict(zip("abco",[[1]*10,
                                        [2]*10,
                                        [3]*10,
                                        list("1234567890")])))
    df2 = pd.DataFrame(dict(zip("abo",[[4]*5,
                                       [5]*5,
                                       list("12345")])))
    
    df3 = common.override_dataframe(df1, df2, ["o"])
    df4 = pd.DataFrame(dict(zip("abco", [[4]*5+[1]*5,
                                         [5]*5+[2]*5,
                                         [3]*10,
                                         list("1234567890")])))
    assert df4.equals(df3)
