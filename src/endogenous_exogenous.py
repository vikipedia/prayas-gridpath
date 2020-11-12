import functools
import numpy as np
import pandas as pd
import json
import os
import subprocess
import click
import forced_outage
import web
import common


def read_availabilty_results(conn, scenario_name, project):
    """
    read results_project_availability_endogenous table
    for given scenario and project
    """
    table = common.get_table_dataframe(conn, "results_project_availability_endogenous")
    scenario_id = get_scenario_id(conn, scenario_name)
    return table[(table.project == project) & (table.scenario_id == scenario_id)]


@functools.lru_cache(maxsize=None)
def get_exogenous_avail_id(conn, scenario, project):
    proj_avail_id = common.get_field(conn,
                                     "scenarios",
                                     "project_availability_scenario_id",
                                     scenario_name= scenario)
    return common.get_field(conn,
                            "inputs_project_availability",
                            "exogenous_availability_scenario_id",
                            project_availability_scenario_id= proj_avail_id,
                            project= project),


@functools.lru_cache(maxsize=None)
def get_temporal_scenario_id(conn, scenario):
    return common.get_field(conn,
                            "scenarios",
                            "temporal_scenario_id",
                            scenario_name= scenario)


@functools.lru_cache(maxsize=None)
def get_scenario_id(conn, scenario_name):
    return common.get_field(conn,
                            table="scenarios",
                            field="scenario_id",
                            scenario_name= scenario_name)


class TemporalSpecsMisMatch(Exception):
    pass

def get_start_end_(df, horizon, temporal_scenario_id):
    df = df[(df.horizon == str(horizon)) &
            (df.temporal_scenario_id == temporal_scenario_id)]
    
    return df.iloc[0]['tmp_start'], df.iloc[0]['tmp_end']


def get_start_end(df, horizon, temp_scena_id):
    if isinstance(horizon, int):
        return get_start_end_(df, horizon, temp_scena_id)
    else:
        return [get_start_end_(df, h, temp_scena_id) for h in horizon]

def create_array(start, end):
    return np.arange(start, end+1)

    
def find_timepoints(conn, horizon, scenario):
    temp_scena_id = get_temporal_scenario_id(conn, scenario)
    df = common.get_table_dataframe(conn,
                                    "inputs_temporal_horizon_timepoints_start_end")
    try:
        st = get_start_end(df, horizon, temp_scena_id)
        if isinstance(st, tuple):
            return create_array(*st)
        else:
            return np.hstack([create_array(*se) for se in st])
    except IndexError as e:
        msg = f"Timepoints of previous senario do not match with horizons of {scenario}"
        raise TemporalSpecsMisMatch(msg, e)


def get_generic_col(size, dtype, value):
    col = np.empty(size, dtype=dtype)
    col[:] = value
    return col


def create_table_for_horizon(conn,
                             horizon, monthly, scenario2, scenario3,
                             project, colnames,
                             exo_id_value):
    HORIZON = list(horizon.keys())[0]
    row = monthly[monthly.timepoint == HORIZON].iloc[0]
    if scenario3:
        timepoints = find_timepoints(conn, horizon[HORIZON], scenario3)
    else:
        timepoints = find_timepoints(conn, horizon[HORIZON], scenario2)
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


def get_temporal_start_end_table(conn, scenario):
    temporal_id = get_temporal_scenario_id(conn, scenario)
    temporal = conn.where("inputs_temporal_horizon_timepoints_start_end",
                          temporal_scenario_id=temporal_id).list()
    return temporal


def get_horizons(conn, timepoints , scenario1, scenario2, scenario3):
    """
    Very complicated way to handle horizons matching of 
    scenario1 -> scenario2 -> scenario3 with only
    results from scenario1
    """
    if not scenario3:
        return {t:{t:t} for t in timepoints}
    
    tmp1 = get_temporal_start_end_table(conn, scenario1)
    tmp2 = get_temporal_start_end_table(conn, scenario2)
    tmp3 = get_temporal_start_end_table(conn, scenario3)

    timepoints2 = []
    for item in tmp2:
        timepoints2.extend(list(range(item['tmp_start'], item['tmp_end']+1)))
    
    horizons3 = [int(item['horizon']) for item in tmp3]
    horizons3.sort()
    timepoints2.sort()

    if horizons3 == timepoints2:
        return {int(r['horizon']):{int(r['horizon']):range(r['tmp_start'], r['tmp_end']+1)} for r in tmp2}


def get_exogenous_results_(conn,
                           scenario1,
                           scenario2,
                           scenario3=None,
                           project=None):
    monthly = read_availabilty_results(conn, scenario1, project)
    colnames = ["project",
                "exogenous_availability_scenario_id",
                "stage_id",
                "timepoint",
                "availability_derate"]
    if scenario3:
        exo_id_value = get_exogenous_avail_id(conn, scenario3, project)
    else:
        exo_id_value = get_exogenous_avail_id(conn, scenario2, project)
    hdict = get_horizons(conn, monthly['timepoint'],
                         scenario1, scenario2, scenario3)
    params = (monthly, scenario2, scenario3, project, colnames, exo_id_value)
    horizons = list(hdict.keys())
    total_df = create_table_for_horizon(conn,hdict[horizons[0]], *params)
    for horizon in horizons[1:]:
        df = create_table_for_horizon(conn, hdict[horizon], *params)
        total_df = pd.concat([total_df, df])
    total_df.reset_index(drop=True, inplace=True)
    return total_df, monthly


def dbwrite_endogenous_monthly_exogenous_input_daily(
        scenario1,
        scenario2,
        scenario3,
        project,
        dbpath):
    """
    creates a pipeline from results_project_availability_endogenous to
    inputs_project_availability_exogenous for given project
    """
    
    conn = common.get_database(dbpath)
    results = get_exogenous_results(conn,
                                    scenario1,
                                    scenario2,
                                    scenario3,
                                    project)
    exo_id = results.exogenous_availability_scenario_id.unique()[0]
    deletecond = {"project": project,
                  "exogenous_availability_scenario_id": exo_id}
        
    update_column(conn, "inputs_project_availability_exogenous",
                       results, "availability_derate")


def get_exogenous_results(scenario1, scenario2, scenario3,fo,
                          project, db_path):
    conn = common.get_database(db_path)
    results,monthly = get_exogenous_results_(conn,
                                             scenario1,
                                             scenario2,
                                             scenario3,
                                             project)
    if scenario3 and isinstance(fo, pd.DataFrame):
        derate = combine_forced_outage(results, fo, project)
        results['availability_derate'] = derate

    return results, monthly


def merge_in_csv(results,
                 csvpath):
    cols = ['stage_id', 'timepoint', 'availability_derate']
    on = 'timepoint'
    common.merge_in_csv(results, csvpath, cols, on)

    
def write_exogenous_results_csv(results,
                                project,
                                csv_location):
    subscenario = 'exogenous_availability_scenario_id'
    subscenario_id = results.iloc[0][subscenario]
    csvpath = common.get_subscenario_csvpath(project, subscenario,
                                             subscenario_id, csv_location)
    merge_in_csv(results, csvpath)
    return subscenario, subscenario_id


def combine_forced_outage(maintenance, fofull, project):
    m = maintenance['availability_derate'].copy()
    fo = fofull[project]
    return forced_outage.combine_fo_m(m, fo)
    
def write_exogenous_via_gridpath_script(scenario1,
                                        scenario2,
                                        scenario3,
                                        fo,
                                        project,
                                        csv_location,
                                        gridpath_rep,
                                        db_path):
    results,monthly = get_exogenous_results(scenario1,
                                            scenario2,
                                            scenario3,
                                            fo,
                                            project,
                                            db_path)
    subscenario, subscenario_id = write_exogenous_results_csv(results,
                                                              project,
                                                              csv_location)
    if not scenario3:
        csvpath = common.get_subscenario_csvpath(project,
                                                 subscenario,
                                                 subscenario_id,
                                                 csv_location)
        merge_in_csv(monthly, csvpath)

    common.update_subscenario_via_gridpath(subscenario, subscenario_id,
                                           project, csv_location, db_path,
                                           gridpath_rep)

    
def find_projects(scenario1, type_, webdb):
    availability_id1 = common.get_field(webdb,
                                        "scenarios",
                                        "project_availability_scenario_id",
                                        scenario_name= scenario1)
    
    rows = webdb.where("inputs_project_availability",
                       project_availability_scenario_id=availability_id1,
                       availability_type=type_).list()
    
    return set(r['project'] for r in rows)


def find_projects_to_copy(scenario1, scenario2, db_path):
    webdb = common.get_database(db_path)
    projects1 = find_projects(scenario1, "binary", webdb)
    projects2 = find_projects(scenario2, "exogenous", webdb)
    return sorted(projects1 & projects2)


def endogenous_to_exogenous(scenario1:str,
                            scenario2:str,
                            scenario3:str,
                            fo:str,
                            csv_location:str,
                            database:str,
                            gridpath_rep:str,
                            skip_scenario2:bool,
                            dev:bool):

    projs = find_projects_to_copy(scenario1, scenario2, database)
    if dev:
        projs = projs[:1]
    
    if not skip_scenario2:
        for project in projs:
            print(f"Starting {project} for {scenario2} ...")
            write_exogenous_via_gridpath_script(scenario1,
                                                scenario2,
                                                scenario3=None,
                                                fo=None,
                                                project=project,
                                                csv_location=csv_location,
                                                gridpath_rep= gridpath_rep,
                                                db_path=database)

    if scenario3:
        if fo:
            print("Reading forced outage excel workbook")
            fo = pd.read_excel(fo,
                               sheet_name="gridpath-input",
                               nrows=35041,
                               usecols=projs)
        
        for project in projs:
            print(f"Starting {project} for {scenario3} ...")
            write_exogenous_via_gridpath_script(scenario1,
                                                scenario2,
                                                scenario3,
                                                fo,
                                                project,
                                                csv_location,
                                                gridpath_rep,
                                                db_path=database)


@click.command()
@click.option("-s1", "--scenario1", default="toy1_pass1", help="Name of scenario1")
@click.option("-s2", "--scenario2", default="toy1_pass2", help="Name of scenario2")
@click.option("-s3", "--scenario3", default=None, help="Name of scenario3")
@click.option("-f", "--fo", default=None, help="Excel filepath, containing forced outage information")
@click.option("-c", "--csv_location", default="csvs_toy", help="Path to folder where csvs are")
@click.option("-d", "--database", default="../toy.db", help="Path to database")
@click.option("-g", "--gridpath_rep", default="../", help="Path of gridpath source repository")
@click.option("--skip_scenario2/--no-skip_scenario2", default=False, help="skip copying for senario2")
@click.option("--dev/--no-dev", default=False, help="Run only for one project")
def main(scenario1:str,
         scenario2:str,
         scenario3:str,
         fo:str,
         csv_location:str,
         database:str,
         gridpath_rep:str,
         skip_scenario2:bool,
         dev:bool):

    """
    Usage: python endogenous_exogenous.py [OPTIONS]
    this is a script to copy endogenous output from scenario1
    to exogenous input of scenario2. to run this script, gridpath
    virtual environment must be active.

    Options:

      --scenario1 TEXT     default -> toy1_pass1

      --scenario2 TEXT     default -> toy1_pass2

      --scenario3 TEXT     default -> None

      --fo TEXT  default -> None     

      --csv_location TEXT      default -> csvs_toy

      --database TEXT      default -> ../toy.db

      --gridpath_rep TEXT  default-> ../

    """
    return endogenous_to_exogenous(
        scenario1,
        scenario2,
        scenario3,
        fo,
        csv_location,
        database,
        gridpath_rep,
        skip_scenario2,
        dev
    )
    
if __name__ == "__main__":
    main()
