import functools
import numpy as np
import pandas as pd
import web
import json
import os
import subprocess
import click

DB_PATH = "/home/vikrant/programming/work/publicgit/gridpath/toy.db"


class NoEntriesError(Exception):
    pass


def execute(conn, query, params):
    print(query, params)
    result = conn.cursor().execute(query, params)
    conn.commit()
    r = result.fetchall()
    return r


def get_field(webdb, table, field, conds):
    r = webdb.where(table, **conds).first()

    if not r:
        raise NoEntriesError(f"Field {field} not found in {table} for {cond}"+str(params))
    return r[field]


def get_table_dataframe(webdb, table):
    """
    read table as dataframe. 
    """
    return pd.read_json(json.dumps(webdb.select(table).list()))


def update_column(webdb, table, df, column):
    df_dict = df.to_dict(orient='split')
    rows = df_dict['data']
    cols = df_dict['columns']
    index = cols.index(column)
    cols = [cols[i] for i in range(len(cols)) if i!=index]
    with webdb.transaction():
        for row in rows:
            webdb.update(table,
                         **{column:row[index],
                            "where":" and ".join([f"{c}=${c}" for c in cols]),
                            "vars":dict(zip(cols, row))})


def update_table(webdb, table, df, delcondcols):
    """FIXME...not working"""
    df_dict = df.to_dict(orient='records')
    with webdb.transaction():
        webdb.delete(table,
                     where=" and ".join([f"{c}=${c}" for c in delcondcols]),
                     vars=delcondcols)
        webdb.multiple_insert(table, values=df_dict)


def get_database(db_path=DB_PATH):
    db = web.database("sqlite:///" + db_path)
    db.printing = False
    return db


def read_availabilty_results(conn, scenario_name, project):
    """
    read results_project_availability_endogenous table
    for given scenario and project
    """
    table = get_table_dataframe(conn, "results_project_availability_endogenous")
    scenario_id = get_scenario_id(conn, scenario_name)
    return table[(table.project == project) & table.scenario_id == scenario_id]


@functools.lru_cache(maxsize=None)
def get_exogenous_avail_id(conn, scenario, project):
    proj_avail_id = get_field(conn,
                              "scenarios",
                              "project_availability_scenario_id",
                              {"scenario_name": scenario})
    return get_field(conn,
                      "inputs_project_availability",
                      "exogenous_availability_scenario_id",
                      {"project_availability_scenario_id": proj_avail_id,
                       "project": project}),


@functools.lru_cache(maxsize=None)
def get_temporal_senario_id(conn, scenario):
    return get_field(conn,
                          "scenarios",
                          "temporal_scenario_id",
                          {"scenario_name": scenario})


@functools.lru_cache(maxsize=None)
def get_scenario_id(conn, scenario_name):
    return get_field(conn,
                          table="scenarios",
                          field="scenario_id",
                          conds={"scenario_name": scenario_name})


class TemporalSpecsMisMatch(Exception):
    pass


def find_timepoints(conn, horizon, scenario):
    temp_scena_id = get_temporal_senario_id(conn, scenario)
    df = get_table_dataframe(conn,
                             "inputs_temporal_horizon_timepoints_start_end")
    df = df[(df.horizon == horizon) &
            (df.temporal_scenario_id == temp_scena_id)]
    r, c = df.shape
    if not r:
        msg = f"Timepoints of previous senario do not match with horizons of {scenario}"
        raise TemporalSpecsMisMatch(msg)

    start = df.iloc[0]['tmp_start']
    end = df.iloc[0]['tmp_end']
    return np.arange(start, end+1)


def get_generic_col(size, dtype, value):
    col = np.empty(size, dtype=dtype)
    col[:] = value
    return col


def create_table_for_horizon(conn,
                             horizon, monthly, scenario2,
                             project, colnames,
                             exo_id_value):
    row = monthly[monthly.timepoint == horizon].iloc[0]
    timepoints = find_timepoints(conn, horizon, scenario2)
    size = len(timepoints)
    project_ = get_generic_col(size, object, project)
    exo_id = get_generic_col(size, int, exo_id_value)
    stage_id = get_generic_col(size, int, row['stage_id'])
    availability_derate = get_generic_col(size, float,
                                          row['availability_derate'])
    #availability_derate = get_generic_col(size, float, 1.1)
    colsdata = [project_, exo_id, stage_id, timepoints, availability_derate]

    df = pd.DataFrame(dict(zip(colnames,
                               colsdata)))
    return df


def get_exogenous_results(conn, scenario1, scenario2, project):
    monthly = read_availabilty_results(conn, scenario1, project)
    colnames = ["project",
                "exogenous_availability_scenario_id",
                "stage_id",
                "timepoint",
                "availability_derate"]
    exo_id_value = get_exogenous_avail_id(conn, scenario2, project)
    horizons = list(monthly['timepoint'])
    params = (monthly, scenario2, project, colnames, exo_id_value)
    total_df = create_table_for_horizon(conn, horizons[0], *params)
    for horizon in horizons[1:]:
        df = create_table_for_horizon(conn, horizon, *params)
        total_df = pd.concat([total_df, df])
    return total_df


def dbwrite_endogenous_monthly_exogenous_input_daily(
                                                  scenario1,
                                                  scenario2,
                                                  project):
    """
    creates a pipeline from results_project_availability_endogenous to
    inputs_project_availability_exogenous for given project
    """
    
    conn = get_database()
    results = get_exogenous_results(conn, scenario1, scenario2, project)
    exo_id = results.exogenous_availability_scenario_id.unique()[0]
    deletecond = {"project": project,
                  "exogenous_availability_scenario_id": exo_id}
        
    update_column(conn, "inputs_project_availability_exogenous",
                       results, "availability_derate")

    #update_table(conn, 'inputs_project_availability_exogenous',
    #                  results, deletecond)


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


def write_exogenous_results_csv(scenario1, scenario2, project, csv_location,
                                db_path=DB_PATH):
    conn = get_database(db_path)
    r = get_exogenous_results(conn, scenario1, scenario2, project)
    subscenario = 'exogenous_availability_scenario_id'
    subscenario_id = r.iloc[0][subscenario]
    csvpath = get_subscenario_csvpath(project, subscenario,
                                      subscenario_id, csv_location)
    csvresults = r.loc[:, ['stage_id', 'timepoint', 'availability_derate']]
    print(f"Writing results to {csvpath}")
    csvresults.to_csv(csvpath, index=False)
    return subscenario, subscenario_id


def create_command(subscenario, subscenario_id, project, csv_location,
                   db_path, gridpath_rep):
    script = os.path.join(gridpath_rep, "db", "utilities", "port_csvs_to_db.py")
    args = f"--database {db_path} --csv_location {csv_location}" \
        f" --project {project} --delete --subscenario {subscenario} " \
        f" --subscenario_id {subscenario_id}"
    return " ".join(["python", script, args])


def write_exogenous_via_gridpath_script(scenario1, scenario2, project,
                                        csv_location,
                                        gridpath_rep,
                                        db_path=DB_PATH):
    subscenario, subscenario_id = write_exogenous_results_csv(scenario1,
                                                              scenario2,
                                                              project,
                                                              csv_location)
    
    cmd = create_command(subscenario, subscenario_id, project,
                         csv_location, db_path, gridpath_rep)
    
    p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate("y".encode())
    print(stdout.decode())
    print(stderr.decode())


    
def find_binary_projects(scenario1, scenario2, db_path):
    webdb = get_database(db_path)
    availability_id1 = get_field(webdb,
                                 "scenarios",
                                 "project_availability_scenario_id",
                                 {"scenario_name": scenario1})
    
    rows = webdb.where("inputs_project_availability",
                       project_availability_scenario_id=availability_id1,
                       availability_type="binary").list()
    return [r['project'] for r in rows]


@click.command()
@click.option("--scenario1", default="toy1_pass1", help="Name of scenario1")
@click.option("--scenario2", default="toy1_pass2", help="Name of scenario2")
@click.option("--csv_location", default="csvs_toy", help="Path to folder where csvs are")
@click.option("--database", default="../toy.db", help="Path to database")
@click.option("--gridpath_rep", default="../", help="Path of gridpath source repository")
def endogenous_to_exogenous(scenario1:str,
                            scenario2:str,
                            csv_location:str,
                            database:str,
                            gridpath_rep:str):
    """
    Usage: python endogenous_exogenous.py [OPTIONS]
    this is a script to copy endogenous output from scenario1
    to exogenous input of scenario2. to run this script, gridpath
    virtual environment must be active.

    Options:

      --scenario1 TEXT     default -> toy1_pass1

      --scenario2 TEXT     default -> toy1_pass2

      --csv_path TEXT      default -> csvs_toy

      --database TEXT      default -> ../toy.db

      --gridpath_rep TEXT  default-> ../

    """

    projs = find_binary_projects(scenario1, scenario2, database)
    for project in projs:
        print(f"Starting {project}...")
        write_exogenous_via_gridpath_script(scenario1,
                                            scenario2,
                                            project,
                                            csv_location,
                                            gridpath_rep,
                                            db_path=database)

if __name__ == "__main__":
    endogenous_to_exogenous()
