import web
import pandas as pd
import numpy as np
import os
import subprocess

class NoEntriesError(Exception):
    pass


def get_field(webdb, table, field, **conds):
    r = webdb.where(table, **conds).first()

    if not r:
        raise NoEntriesError(f"Field {field} not found in {table} for {conds}")
    return r[field]


def get_table_dataframe(webdb:web.db.SqliteDB, table:str)->pd.DataFrame:
    """
    read table as dataframe. 
    """
    return pd.DataFrame(webdb.select(table).list())


def get_database(db_path):
    db = web.database("sqlite:///" + db_path)
    db.printing = False
    return db


def get_master_csv_path(csv_location):
    return os.path.join(csv_location, "csv_data_master.csv")


def get_subscenario_path(csv_location, subscenario):
    csvmasterpath = get_master_csv_path(csv_location)
    csvmaster = pd.read_csv(csvmasterpath)
    row = csvmaster[csvmaster.subscenario == subscenario].iloc[0]
    return os.path.join(csv_location, row['path'])


def get_subscenario_csvpath(project, subscenario, subscenario_id, csv_location):
    path = get_subscenario_path(csv_location, subscenario)
    csvs = [f for f in os.listdir(path) if f.startswith(f"{project}-{subscenario_id}")]
    if csvs:
        return os.path.join(path, csvs[0])
    else:
        raise Exception(f"CSV not found for {project}-{subscenario_id}")    

    
def create_command(subscenario, subscenario_id, project, csv_location,
                   db_path, gridpath_rep):
    script = os.path.join(gridpath_rep, "db", "utilities", "port_csvs_to_db.py")
    args = f"--database {db_path} --csv_location {csv_location}" \
        f" --project {project} --delete --subscenario {subscenario} " \
        f" --subscenario_id {subscenario_id}"
    return " ".join(["python", script, args])

    
def update_subscenario_via_gridpath(subscenario,
                                    subscenario_id,
                                    project,
                                    csv_location,
                                    db_path,
                                    gridpath_rep):
    
    cmd = create_command(subscenario, subscenario_id, project,
                         csv_location, db_path, gridpath_rep)
    print(cmd)

    p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate("y".encode())
    print(stdout.decode())
    print(stderr.decode())


def merge_in_csv(results,
                 csvpath, cols, on):
    csvresults = results.loc[:, cols]
    index = csvresults[on]
    csvresults.set_index(index, inplace=True)

    allcsv = pd.read_csv(csvpath, index_col=on)
    try:
        allcsv.loc[index.min():index.max()] = csvresults
    except ValueError as v:
        print("Failed to merge {on} {}-{}".format(index.min(),
                                                        index.max()))
        print("Continuing ....")
    allcsv[on] = allcsv.index
    print(f"Merging results to {csvpath}")
    allcsv.to_csv(csvpath, index=False, columns=cols)

