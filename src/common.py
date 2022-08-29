import web
import pandas as pd
import numpy as np
import os
import subprocess

class NoEntriesError(Exception):
    pass


def get_field(webdb, table, field, **conds):
    r = webdb.where(table, **conds).first()

    if not r or field not in r:
        raise NoEntriesError(f"Field {field} not found in {table} for {conds}")
    if not r[field]:
        print(f"Warning: {field} from {table} for {conds} is empty or None")
    return r[field]


def get_table_dataframe(webdb:web.db.SqliteDB, table:str)->pd.DataFrame:
    """
    read table as dataframe. 
    """
    t = pd.DataFrame(webdb.select(table).list())
    if t.shape[0]>0:
        return t
    else:
        raise NoEntriesError(f"Table {table} is empty")


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


def get_subscenario_csvpath(project,
                            subscenario,
                            subscenario_id,
                            csv_location, description="description"):
    path = get_subscenario_path(csv_location, subscenario)
    csvs = [f for f in os.listdir(path) if f.startswith(f"{project}-{subscenario_id}")]
    if csvs:
        return os.path.join(path, csvs[0])
    else:
        print(f"CSV not found for {project}-{subscenario_id}")
        filename = f"{project}-{subscenario_id}-{description}.csv"
        print(f"Creating  {filename}")
        fpath = os.path.join(path, filename)
        with open(fpath, "w+") as f:
            f.write("stage_id,timepoint,availability_derate")
        return fpath

def update_scenario_via_gridpath(scenario,
                                 csv_location,
                                 db_path,
                                 gridpath_rep):
    csv_path = os.path.join(csv_location, "scenarios.csv")
    script = os.path.join(gridpath_rep, "db", "utilities", "scenario.py")
    args = f"--database {db_path} --csv_path {csv_path} " \
        f"--scenario {scenario}"

    cmd =  " ".join(["python", script, args])
    print(cmd)
    p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate("y".encode())
    print(stdout.decode())
    print(stderr.decode())
    #out_bytes = subprocess.check_output(cmd, shell=True)
    #print(out_bytes.decode())


def run_scenario(scenario,
                 csv_location,
                 db_path):                 
                                                                                                  
    scen_loca = '../../../scenarios/toy_run_seq1'
    cmd = "gridpath_run_e2e --scenario %s --database %s --log --scenario_location %s"%(scenario, db_path, scen_loca)
                                                                                                                                                                               
    print(cmd)

    out_bytes = subprocess.run(cmd, shell=True) #- "check_output" needs zero output
    conn = get_database(db_path)
    table = get_table_dataframe(conn, "scenarios")
    run_id = table[table.scenario_name == scenario]['run_status_id'].squeeze()
    if run_id != 2:
        return False
    return True
    

def create_command(subscenario,
                   subscenario_id,
                   project,
                   csv_location,
                   db_path,
                   gridpath_rep,
                   delete=True):
    script = os.path.join(gridpath_rep, "db", "utilities", "port_csvs_to_db.py")
    args = f"--database {db_path} --csv_location {csv_location} " \
        f"--subscenario {subscenario} --subscenario_id {subscenario_id} --delete"
    if project:
        args = args + f" --project {project}"
        
    return " ".join(["python", script, args])

    
def update_subscenario_via_gridpath(subscenario,
                                    subscenario_id,
                                    project,
                                    csv_location,
                                    db_path,
                                    gridpath_rep):
    
    cmd = create_command(subscenario, subscenario_id, project,
                         csv_location, db_path, gridpath_rep)
    
    p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate("y".encode())
    print(stdout.decode())
    print(stderr.decode())

#import logging
#logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def merge_in_csv(results,
                 csvpath, cols, on):
    #logging.info("Entry")
    csvresults = results.loc[:, cols]
    index = csvresults[on]
    csvresults.set_index(index, inplace=True)
    
    allcsv = pd.read_csv(csvpath, index_col=on)

    try:                                              
        allcsv.loc[index.min():index.max()] = csvresults
    except ValueError as v:
        print("Failed to merge {} {}-{}".format(on, index.min(),
                                                        index.max()))
        print("Continuing ....by appending")
        allcsv = pd.concat([allcsv, csvresults])

    allcsv[on] = allcsv.index
    print(f"Merging results to {csvpath}")
    allcsv.to_csv(csvpath, index=False, columns=cols)
    #logging.info("Exit")
