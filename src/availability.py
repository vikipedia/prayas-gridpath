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
import sqlite3
import logging
from logger import init as init_logger


logger = logging.getLogger(__name__)


def read_availabilty_results(conn, scenario_name, project):
    """
    read results_project_availability_endogenous table
    for given scenario and project
    """
    scenario_id = get_scenario_id(conn, scenario_name)
    try:  # inputs_project_availability_exogenous
        r = common.filtered_table(conn,
                                  "results_project_availability_endogenous",
                                  project=project,
                                  scenario_id=scenario_id)
        logger.info(
            f"read data from results_project_availability_endogenous for {scenario_name}")
    except sqlite3.OperationalError as oe:
        r = common.filtered_table(conn,
                                  "results_project_timepoint",
                                  project=project,
                                  scenario_id=scenario_id)
        logger.info(
            f"read data from results_project_timepoint for {scenario_name}")
    if r.shape[0] > 0:
        return r
    else:
        exo_id = get_exogenous_avail_id(conn, scenario_name, project)
        r = common.filtered_table(conn,
                                  "inputs_project_availability_exogenous",
                                  project=project,
                                  exogenous_availability_scenario_id=exo_id)
        logger.info(
            f"read data from inputs_project_availability_exogenous for {scenario_name}")
        if r.shape[0] > 0:
            return r
        else:
            raise common.NoEntriesError(
                f"Availability results for {project} and {scenario_name} are neither available in table results_project_timepoint/results_project_availability_endogenous nor in inputs_project_availability_exogenous ")


def availability_precision_correction(availability, decimals=1):
    r = availability
    if (r.availability_derate < 0).sum() < 0:
        logger.warning(
            "Some values of availability_derate were negative, focefully setting them to 0")
    if (r.availability_derate > 1).sum() > 1:
        logger.warning(
            "Some values of availability_derate were more than 1, focefully setting them to 1")
    r['availability_derate'] = np.where(r.availability_derate < 0,
                                        0,
                                        r.availability_derate)
    r['availability_derate'] = np.where(r.availability_derate > 1,
                                        1,
                                        r.availability_derate)
    r['availability_derate'] = np.round(r.availability_derate,
                                        decimals=decimals)

    return r


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
                            project=project)


@functools.lru_cache(maxsize=None)
def get_temporal_scenario_id(conn, scenario):
    return common.get_field(conn,
                            "scenarios",
                            "temporal_scenario_id",
                            scenario_name=scenario)


def num_hrs_in_tp(conn, scenario):
    tmp_id = get_temporal_scenario_id(conn, scenario)
    return conn.where("inputs_temporal",
                      "number_of_hours_in_timepoint",
                      temporal_scenario_id=tmp_id)


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
    r['hyb_stor_cap_availability_derate'] = None
    return r


def get_exogenous_results_(conn,
                           scenario1,
                           scenario2,
                           scenario3=None,
                           project=None,
                           mapfile=None):
    pass1_results = read_availabilty_results(conn, scenario1, project)
    pass1_results = availability_precision_correction(pass1_results)
    colnames = ["project",
                "exogenous_availability_scenario_id",
                "stage_id",
                "timepoint",
                "availability_derate",
                "hyb_stor_cap_availability_derate"]

    timepoint_map = pd.read_excel(mapfile,
                                  sheet_name="map",
                                  skiprows=2,
                                  engine='openpyxl')

    if scenario3:
        exo_id_value = get_exogenous_avail_id(conn, scenario3, project)
        target_timepoints = [
            c for c in timepoint_map.columns if c.startswith("pass3_timepoint")][0]
    else:
        exo_id_value = get_exogenous_avail_id(conn, scenario2, project)
        target_timepoints = [
            c for c in timepoint_map.columns if c.startswith("pass2_timepoint")][0]
    params = (pass1_results, scenario2, scenario3,
              project, colnames, exo_id_value)

    r = combine_with_timepoint_map(conn,
                                   scenario1,
                                   target_timepoints,
                                   timepoint_map,
                                   pass1_results)
    g = r.reset_index().groupby(target_timepoints)
    gsum = g.sum(numeric_only=True)
    derate = gsum['availability_derate']/gsum['weight']
    derate = derate.reset_index().rename(
        columns={target_timepoints: 'timepoint', 0: "availability_derate"})
    stage_id = g.first()

    r = create_table_for_results(derate,
                                 project,
                                 exo_id_value,
                                 stage_id)

    return r, pass1_results


def combine_with_timepoint_map(conn,
                               scenario1,
                               target_timepoints,
                               timepoint_map,
                               pass1_results):
    source_timepoints = [
        c for c in timepoint_map.columns if c == "pass1_timepoint"][0]

    timepoints = np.array(pass1_results['timepoint'].unique())
    if 'number_of_hours_in_timepoint' not in timepoint_map.columns:
        df_tmp = common.get_table_dataframe(conn, 'inputs_temporal')
        tmp_sc_id = get_temporal_scenario_id(conn, scenario1)
        df_tmp_sc1 = df_tmp[df_tmp.temporal_scenario_id ==
                            tmp_sc_id].reset_index(drop=True)
        col_tp = [c for c in timepoint_map.columns if c.startswith(
            "pass1_timepoint")][0]
        timepoint_map = timepoint_map.set_index(col_tp, drop=False).join(
            df_tmp_sc1[['timepoint', 'number_of_hours_in_timepoint']].set_index('timepoint'))
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
    if r['availability_derate'].isnull().sum() == len(r):
        raise Exception("Wrong timepoint map provided")

    return r


def get_exogenous_results(scenario1, scenario2, scenario3, fo,
                          project, db_path, mapfile):
    conn = common.get_database(db_path)
    results, pass1_results = get_exogenous_results_(conn,
                                                    scenario1,
                                                    scenario2,
                                                    scenario3,
                                                    project,
                                                    mapfile)

    if scenario3 and fo is not None:
        results = combine_fo(results, fo)
    return results, pass1_results


def combine_fo(results, fo):
    logger.info("Combining forced outage...")
    results_ = results.set_index('timepoint')
    fo_ = fo.copy()
    fo_.name = 'fo'
    results_ = results_.join(fo_)  # fo has timepoint as index
    results = results_.reset_index()

    if results.fo.isnull().sum() != 0:
        logger.error("Timepoint in forced ouage is incorrect")
        raise Exception("Timepoint in forced ouage is incorrect")
    m = results['availability_derate'].copy()
    f = results['fo']
    del results['fo']
    derate = forced_outage.combine_fo_m(m, f)
    results['availability_derate'] = derate
    return results


def merge_in_csv(results,
                 csvpath):
    cols = ['stage_id', 'timepoint', 'availability_derate',
            'hyb_stor_cap_availability_derate']
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


def sanity_check(csvpath):
    df = pd.read_csv(csvpath)
    if df.shape[0] == len(df.timepoint.unique()):
        return True
    else:
        logger.info(
            f"Sanity check failed on {csvpath}, updating database skipped")


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
    results, pass1_results = get_exogenous_results(scenario1,
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
    # if not scenario3:
    #     merge_in_csv(pass1_results, csvpath)

    if update_database and sanity_check(csvpath):
        common.update_subscenario_via_gridpath(subscenario, subscenario_id,
                                               project, csv_location, db_path,
                                               gridpath_repo)


def get_type(project, scenario, webdb):
    availability_id1 = common.get_field(webdb,
                                        "scenarios",
                                        "project_availability_scenario_id",
                                        scenario_name=scenario)
    return common.get_field(webdb,
                            "inputs_project_availability",
                            "availability_type",
                            project_availability_scenario_id=availability_id1,
                            project=project)


def find_projects(scenario1, type_=None, webdb=None):
    availability_id1 = common.get_field(webdb,
                                        "scenarios",
                                        "project_availability_scenario_id",
                                        scenario_name=scenario1)
    
    portfolio_id = common.get_field(webdb,
                                    "scenarios",
                                    "project_portfolio_scenario_id",
                                    scenario_name=scenario1)
    
    params = {"project_availability_scenario_id": availability_id1}

    if type_:
        params["availability_type"] = type_

    rows = webdb.where("inputs_project_availability",
                       **params).list()
    portfolio_rows = webdb.where("inputs_project_portfolios",
                                 project_portfolio_scenario_id=portfolio_id).list()

    return set(r['project'] for r in rows) & set(r['project'] for r in portfolio_rows)


def find_projects_to_copy(scenario1, scenario2, db_path):
    webdb = common.get_database(db_path)
    projects1 = set(p for p in find_projects(
        scenario1, webdb=webdb) if not (get_type(p, scenario1, webdb) == "exogenous" and get_exogenous_avail_id(webdb, scenario1, p) is None))
    # for scenario 1
    # filter out projects that are exogenous and do not have exo_sc_id provided

    #projects2 = find_projects(scenario2, "exogenous", webdb)
    projects2 = set(p for p in find_projects(scenario2, "exogenous", webdb) if get_exogenous_avail_id(webdb, scenario2, p) is not None)

    if projects1 - projects2:
        logger.warning(f"Some binary projects from {scenario1} are not listed\
 in exogenous projects of {scenario2}")
    return sorted(projects1 & projects2)


def handle_exogenous_only_projects(scenario1,
                                   scenario2,
                                   scenario3,
                                   fo_df,
                                   projs,
                                   database,
                                   mapfile,
                                   update_database,
                                   csv_location,
                                   name,
                                   gridpath_repo):
    """Handles projects which have valid exogenous_availability_scenario_id
    but do not have data to be copied from previous scenario
    """
    conn = common.get_database(database)
    table = common.get_table_dataframe(
        conn, "inputs_project_availability")

    table.dropna(subset=['exogenous_availability_scenario_id'],
                 inplace=True)

    table_scenario = common.get_table_dataframe(conn, "scenarios")
    table_pf = common.get_table_dataframe(
        conn, "inputs_project_portfolios")
    pf_id = table_scenario[table_scenario.scenario_name ==
                           scenario3]['project_portfolio_scenario_id'].squeeze()
    filterd_pf_id = table_pf[table_pf.project_portfolio_scenario_id == pf_id]
    table_pf = filterd_pf_id.reset_index(drop=True)

    exo_prj = list(set([x for x in table.project if (
        (x not in projs) and (x in list(table_pf.project)))]))

    if exo_prj and any(prj in fo_df.columns for prj in exo_prj):
        # just getting dummy dataframe to fill in
        df_ava, df_monthly = get_exogenous_results_(conn,
                                                    scenario1,
                                                    scenario2,
                                                    scenario3,
                                                    projs[0],
                                                    mapfile)

    for prj in exo_prj:
        if prj in fo_df.columns:
            logger.info(f"Updating derate for exogenous project {prj}, from \
{scenario3}")
            del df_ava['availability_derate']
            derate = fo_df[prj].copy()
            print(fo_df[prj])
            derate.name = 'availability_derate'
            df_ava = df_ava.join(derate)
            df_ava['project'] = prj
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
        else:
            logger.warning(
                f"{prj} has valid exogenous_availability_scenario_id but no forced outage is given for the project")


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

    # TODO: Make this an if-else
    projs = find_projects_to_copy(scenario1, scenario2, database)
    if project:
        projs = [project]

    if not skip_scenario2:
        for project_ in projs:
            logger.info(f"Starting {project_} for {scenario2} ...")
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
        # TODO: No need to do this since it is done outside the loop?
        projs = find_projects_to_copy(
            scenario1, scenario3, database)  # this will change

        if fo:
            logger.info("Reading forced outage csv")
            fo_df = pd.read_csv(fo, dtype={'timepoint': np.int64})
            fo_df = fo_df.set_index('timepoint')
            handle_exogenous_only_projects(scenario1, scenario2, scenario3,
                                           fo_df, projs, database, mapfile,
                                           update_database, csv_location, name,
                                           gridpath_repo)
        if project:
            projs = [project]

        for project_ in projs:
            logger.info(f"Starting {project_} for {scenario3} ...")
            if fo:
                project_fo = fo_df[project_] if project_ in fo_df.columns else None
            else:
                project_fo = None

            write_exogenous_via_gridpath_script(scenario1,
                                                scenario2,
                                                scenario3,
                                                project_fo,
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
@click.option("-l", "--loglevel", default="INFO", help="Loglevel one of INFO,WARN,DEBUG,ERROR")
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
         update_database: bool,
         loglevel: str):
    global logger
    init_logger("availability.log", loglevel)
    logger = logging.getLogger("availability")

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
