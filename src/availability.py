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
    r = table[(table.project == project) & (table.scenario_id == scenario_id)]
    if r.shape[0] >0:
        return r
    else:
        raise common.NoEntriesError(f"Availability results for {project} and {scenario_name} are not there in table results_project_availability_endogenous")


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

webdb = common.get_database("/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/mh.db")

def filtered_table(webdb, table, **conds):
    rows = webdb.where(table, **conds).list()
    if rows:
        return pd.DataFrame(rows)
    else:
        raise common.NoEntriesError(f"No entries in {table} for {conds}")

def find_timepoints_(conn, horizon, scenario):
    temp_scena_id = get_temporal_scenario_id(conn, scenario)
    horizon_sub_problem_timepoints = filtered_table(conn,
                                                    "inputs_temporal_horizon_timepoints",
                                                    temporal_scenario_id=temp_scena_id,
                                                    horizon=horizon
                                                    )
    subproblem_id = horizon_sub_problem_timepoints.groupby("subproblem_id").max().index[0]
    spinup_subproblem_timepoints = filtered_table(conn,
                                                  "inputs_temporal",
                                                  temporal_scenario_id=temp_scena_id,
                                                  spinup_or_lookahead=0,
                                                  subproblem_id=str(subproblem_id))
    return spinup_subproblem_timepoints['timepoint']
    


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
        timepoints = find_timepoints_(conn, horizon[HORIZON], scenario3)
    else:
        timepoints = find_timepoints_(conn, horizon[HORIZON], scenario2)
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

    if len(timepoints)==len(timepoints2):
        return {int(r['horizon']):{int(r['horizon']):int(r['horizon'])} for r in tmp3}
        
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
        r, c = monthly.shape
        
        if r==365:
            result = monthly[['project', 'stage_id', 'timepoint', 'availability_derate']]
            exid  = np.empty_like(result['stage_id'])
            exid[:] = exo_id_value
            result['exogenous_availability_scenario_id'] = exid
            return result, monthly
         

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
                                csv_location,
                                description):
    subscenario = 'exogenous_availability_scenario_id'
    subscenario_id = results.iloc[0][subscenario]
    csvpath = common.get_subscenario_csvpath(project, subscenario,
                                             subscenario_id, csv_location,
                                             description)
    merge_in_csv(results, csvpath)
    return subscenario, subscenario_id


def combine_forced_outage(maintenance, fofull, project):
    m = maintenance['availability_derate'].copy()
    fo = fofull[project]
    return forced_outage.combine_fo_m(m, fo)

def sanity_check(csvpath):
    df = pd.read_csv(csvpath)
    if df.shape[0] == len(df.timepoint.unique()):
        return True
    else:
        print(f"Sanity check failed on {csvpath}, updating database skipped")
    

def write_exogenous_via_gridpath_script(scenario1,
                                        scenario2,
                                        scenario3,
                                        fo,
                                        project,
                                        csv_location,
                                        gridpath_repo,
                                        db_path,
                                        description,
                                        update_database):
    results,monthly = get_exogenous_results(scenario1,
                                            scenario2,
                                            scenario3,
                                            fo,
                                            project,
                                            db_path)
    subscenario, subscenario_id = write_exogenous_results_csv(results,
                                                              project,
                                                              csv_location,
                                                              description)

    csvpath = common.get_subscenario_csvpath(project,
                                             subscenario,
                                             subscenario_id,
                                             csv_location,
                                             description)
    if not scenario3:
        merge_in_csv(monthly, csvpath)

    if update_database and sanity_check(csvpath):
        common.update_subscenario_via_gridpath(subscenario, subscenario_id,
                                               project, csv_location, db_path,
                                               gridpath_repo)

    
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
                            gridpath_repo:str,
                            skip_scenario2:bool,
                            project:str,
                            name:str,
                            update_database:bool):

    projs = find_projects_to_copy(scenario1, scenario2, database)
    if project:
        projs = [project]
    
    if not skip_scenario2:
        for project_ in projs:
            print(f"Starting {project_} for {scenario2} ...")        
            write_exogenous_via_gridpath_script(scenario1,
                                                scenario2,
                                                scenario3=None,
                                                fo=None,
                                                project=project_,
                                                csv_location=csv_location,
                                                gridpath_repo= gridpath_repo,
                                                db_path=database,
                                                description=name,
                                                update_database=update_database)

    if scenario3:
        projs = find_projects_to_copy(scenario1, scenario2, database)
        if fo:
            print("Reading forced outage excel workbook")
            fo_all = pd.read_excel(fo,
                               sheet_name="gridpath-input",
                               nrows=35041,
                               engine='openpyxl')

            fo = pd.read_excel(fo,
                               sheet_name="gridpath-input",
                               nrows=35041,
                               usecols=projs,
                               engine='openpyxl')
        
            df_ava, df_monthly = get_exogenous_results(scenario1,
                                            scenario2,
                                            scenario3,
                                            fo,
                                            projs[0],
                                            database)

            conn = common.get_database(database)
            table = common.get_table_dataframe(conn, "inputs_project_availability")
            table.dropna(subset = ['exogenous_availability_scenario_id'], inplace = True)

            exo_prj = [x for x in table.project if x not in projs]

            for prj in exo_prj:
                df_ava['availability_derate'] = fo_all[prj]
                subscenario, subscenario_id = write_exogenous_results_csv(df_ava,
                                                              prj,
                                                              csv_location,
                                                              name)
                if update_database:
                    common.update_subscenario_via_gridpath(subscenario,
                                                           subscenario_id,
                                                           prj, csv_location,
                                                           database,
                                                           gridpath_repo)

        for project_ in projs:
            print(f"Starting {project_} for {scenario3} ...")
            write_exogenous_via_gridpath_script(scenario1,
                                                scenario2,
                                                scenario3,
                                                fo,
                                                project_,
                                                csv_location,
                                                gridpath_repo,
                                                db_path=database,
                                                description=name,
                                                update_database=update_database)


@click.command()
@click.option("-d", "--database", default="dispatch.db", help="Path to database (default: dispatch.db")
@click.option("-c", "--csv_location", default="csvs", help="Path to folder where csvs are (default: csvs)")
@click.option("-g", "--gridpath_repo", default="..", help="Path of gridpath source repository (default: ..)")
@click.option("-s1", "--scenario1", default="pass1", help="Name of scenario1 (default: pass1)")
@click.option("-s2", "--scenario2", default="pass2", help="Name of scenario2 (default: pass2)")
@click.option("-s3", "--scenario3", default=None, help="Name of scenario3 (default: None)")
@click.option("-f", "--fo", default=None, help="Excel filepath, containing forced outage information (default: None)")
@click.option("--skip_scenario2/--no-skip_scenario2", default=False, help="skip copying for senario2 (default: no-skip)")
@click.option("--project", default=None, help="Run only for one project (default: None")
@click.option("-n", "--name", default="all", help="Description in name of csv files (default: all)")
@click.option("--update_database/--no-update_database", default=False, help="Update database only if this flag is True (default: no-update)")
def main(scenario1:str,
         scenario2:str,
         scenario3:str,
         fo:str,
         csv_location:str,
         database:str,
         gridpath_repo:str,
         skip_scenario2:bool,
         project:str,
         name:str,
         update_database:bool):

    """
    Usage: python availability.py [OPTIONS]
    this is a script to copy endogenous output from scenario1
    to exogenous input of scenario2. to run this script, gridpath
    virtual environment must be active.

    Options:

      --database TEXT       default -> ../toy.db

      --csv_location TEXT   default -> csvs_toy

      --gridpath_repo TEXT  default-> ../

      --scenario1 TEXT      default -> toy1_pass1

      --scenario2 TEXT      default -> toy1_pass2

      --scenario3 TEXT      default -> None

      --fo TEXT             default -> None     

      --skip_scenario2      default -> no-skip_scenario2

      --project TEXT        default -> None

      --name TEXT           default -> all

      --update_database     default -> no-update_database

    """
    return endogenous_to_exogenous(
        scenario1,
        scenario2,
        scenario3,
        fo,
        csv_location,
        database,
        gridpath_repo,
        skip_scenario2,
        project,
        name,
        update_database
    )
    
if __name__ == "__main__":
    main()
