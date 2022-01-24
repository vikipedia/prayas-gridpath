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
    table = common.get_table_dataframe(
        conn, "results_project_availability_endogenous")
    scenario_id = get_scenario_id(conn, scenario_name)
    r = table[(table.project == project) & (table.scenario_id == scenario_id)]
    if r.shape[0] > 0:
        return r
    else:
        raise common.NoEntriesError(
            f"Availability results for {project} and {scenario_name} are not there in table results_project_availability_endogenous")


@functools.lru_cache(maxsize=None)
def get_exogenous_avail_id(conn, scenario, project):
    proj_avail_id = common.get_field(conn,
                                     "scenarios",
                                     "project_availability_scenario_id",
                                     scenario_name=scenario)
    return common.get_field(conn,
                            "inputs_project_availability",
                            "exogenous_availability_scenario_id",
                            project_availability_scenario_id=proj_avail_id,
                            project=project),


@functools.lru_cache(maxsize=None)
def get_temporal_scenario_id(conn, scenario):
    return common.get_field(conn,
                            "scenarios",
                            "temporal_scenario_id",
                            scenario_name=scenario)


@functools.lru_cache(maxsize=None)
def get_scenario_id(conn, scenario_name):
    return common.get_field(conn,
                            table="scenarios",
                            field="scenario_id",
                            scenario_name=scenario_name)


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


webdb = common.get_database(
    "/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/mh.db")


def filtered_table(webdb, table, **conds):
    rows = webdb.where(table, **conds).list()
    if rows:
        return pd.DataFrame(rows)
    else:
        raise common.NoEntriesError(f"No entries in {table} for {conds}")


def get_generic_col(size, dtype, value):
    col = np.empty(size, dtype=dtype)
    col[:] = value
    return col


def create_table_for_results(availability_derate,
                             project,
                             exo_id_value,
                             stage_id):
    size = len(availability_derate)
    project_ = get_generic_col(size, object, project)
    exo_id = get_generic_col(size, int, exo_id_value)
    r = availability_derate
    r['project'] = project
    r['exogenous_availability_scenario_id'] = exo_id
    r['stage_id'] = stage_id.reset_index()['stage_id']
    return r


def get_exogenous_results_(conn,
                           scenario1,
                           scenario2,
                           scenario3=None,
                           project=None,
                           mapfile=None):
    pass1_results = read_availabilty_results(conn, scenario1, project)
    colnames = ["project",
                "exogenous_availability_scenario_id",
                "stage_id",
                "timepoint",
                "availability_derate"]

    timepoint_map = pd.read_excel(mapfile,
                                  sheet_name="map",
                                  skiprows=3,
                                  engine='openpyxl')

    source_timepoints = [
        c for c in timepoint_map.columns if c.startswith("pass1_timepoint_")][0]

    if scenario3:
        exo_id_value = get_exogenous_avail_id(conn, scenario3, project)
        target_timepoints = [
            c for c in timepoint_map.columns if c.startswith("pass3_timepoint_")][0]
    else:
        exo_id_value = get_exogenous_avail_id(conn, scenario2, project)
        target_timepoints = [
            c for c in timepoint_map.columns if c.startswith("pass2_timepoint_")][0]
    params = (pass1_results, scenario2, scenario3,
              project, colnames, exo_id_value)

    timepoints = np.array(pass1_results['timepoint'].unique())
    if len(timepoints) > len(timepoint_map[target_timepoints].unique()):
        t_map_g = timepoint_map.groupby(source_timepoints)
        t_map = t_map_g.first()
        weight = t_map_g['number_of_hours_in_timepoint'].sum()
    else:
        t_map = timepoint_map.set_index(source_timepoints)
        weight = t_map['number_of_hours_in_timepoint']

    r = t_map.join(pass1_results.set_index(
        'timepoint'), lsuffix="", rsuffix="_other")
    r['availability_derate'] = r['availability_derate'] * weight
    r['weight'] = weight
    g = r.reset_index().groupby(target_timepoints)
    gsum = g.sum()
    derate = gsum['availability_derate']/gsum['weight']
    derate = derate.reset_index().rename(
        columns={target_timepoints: 'timepoint', 0: "availability_derate"})
    stage_id = g.first()

    r = create_table_for_results(derate,
                                 project,
                                 exo_id_value,
                                 stage_id)

    return r, pass1_results


def get_exogenous_results(scenario1, scenario2, scenario3, fo,
                          project, db_path, mapfile):
    conn = common.get_database(db_path)
    results, monthly = get_exogenous_results_(conn,
                                              scenario1,
                                              scenario2,
                                              scenario3,
                                              project,
                                              mapfile)
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
                                        mapfile,
                                        description,
                                        update_database):
    results, monthly = get_exogenous_results(scenario1,
                                             scenario2,
                                             scenario3,
                                             fo,
                                             project,
                                             db_path,
                                             mapfile)
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
                                        scenario_name=scenario1)

    portfolio_id = common.get_field(webdb,
                                    "scenarios",
                                    "project_portfolio_scenario_id",
                                    scenario_name=scenario1)

    rows = webdb.where("inputs_project_availability",
                       project_availability_scenario_id=availability_id1,
                       availability_type=type_).list()

    portfolio_rows = webdb.where("inputs_project_portfolios",
                                 project_portfolio_scenario_id=portfolio_id).list()

    return set(r['project'] for r in rows) & set(r['project'] for r in portfolio_rows)


def find_projects_to_copy(scenario1, scenario2, db_path):
    webdb = common.get_database(db_path)
    projects1 = find_projects(scenario1, "binary", webdb)
    projects2 = find_projects(scenario2, "exogenous", webdb)
    return sorted(projects1 & projects2)


def endogenous_to_exogenous(scenario1: str,
                            scenario2: str,
                            scenario3: str,
                            fo: str,
                            csv_location: str,
                            database: str,
                            mapfile: str,
                            gridpath_repo: str,
                            skip_scenario2: bool,
                            project: str,
                            name: str,
                            update_database: bool):

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
                                                gridpath_repo=gridpath_repo,
                                                db_path=database,
                                                mapfile=mapfile,
                                                description=name,
                                                update_database=update_database)

    if scenario3:
        projs = find_projects_to_copy(scenario1, scenario3, database)
        if project:
            projs = [project]

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
                                                       database,
                                                       mapfile)

            conn = common.get_database(database)
            table = common.get_table_dataframe(
                conn, "inputs_project_availability")
            table.dropna(
                subset=['exogenous_availability_scenario_id'], inplace=True)

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
                                                mapfile=mapfile,
                                                description=name,
                                                update_database=update_database)


@click.command()
@click.option("-d", "--database", default="dispatch.db", help="Path to database (default: dispatch.db")
@click.option("-t", "--timepoint_map", default="timepoint_map.xlsx", help="Path to timepoints mapfile")
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
def main(scenario1: str,
         scenario2: str,
         scenario3: str,
         fo: str,
         csv_location: str,
         database: str,
         timepoint_map: str,
         gridpath_repo: str,
         skip_scenario2: bool,
         project: str,
         name: str,
         update_database: bool):
    """
    Usage: python availability.py [OPTIONS]
    this is a script to copy endogenous output from scenario1
    to exogenous input of scenario2. to run this script, gridpath
    virtual environment must be active.

    Options:

      --database TEXT       default -> ../toy.db

      --timepoint_map TEXT        default -> timepoint_map.xlsx

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
        timepoint_map,
        gridpath_repo,
        skip_scenario2,
        project,
        name,
        update_database
    )


if __name__ == "__main__":
    main()