import web
import pandas as pd
import numpy as np
import common
import os
import click
import availability
import pytest
import sqlite3
import logging
from logger import init as init_logger

logger = logging.getLogger(__name__)


def read_exogenous_availabilty_results(webdb, scenario, project):
    exo_avail_id = availability.get_exogenous_avail_id(
        webdb, scenario, project)
    table = "inputs_project_availability_exogenous"
    return common.filtered_table(webdb,
                                 table,
                                 project=project,
                                 exogenous_availability_scenario_id=exo_avail_id)


def get_period(webdb, scenario):
    temporal_scenario_id = availability.get_temporal_scenario_id(
        webdb, scenario)
    return common.get_field(webdb, 'inputs_temporal_periods', 'period',
                            temporal_scenario_id=temporal_scenario_id)


def read_inputs_temporal(webdb, scenario):
    temporal_scenario_id = availability.get_temporal_scenario_id(
        webdb, scenario)
    return common.filtered_table(webdb,
                                 "inputs_temporal",
                                 temporal_scenario_id=temporal_scenario_id)


def read_inputs_temporal_horizon_timepoints(webdb, scenario):
    temporal_scenario_id = availability.get_temporal_scenario_id(
        webdb, scenario)
    return common.filtered_table(webdb,
                                 "inputs_temporal_horizon_timepoints",
                                 temporal_scenario_id=temporal_scenario_id)


def compute_availability(availability_data,
                         inputs_temporal,
                         timepoint_horizon_map):

    a = availability_data.merge(
        inputs_temporal[(inputs_temporal.spinup_or_lookahead == 0) | (inputs_temporal.spinup_or_lookahead.isna())], on="timepoint")

    a['weights'] = a.timepoint_weight * a.number_of_hours_in_timepoint
    a.availability_derate = a.availability_derate * a.weights
    a = a.merge(timepoint_horizon_map, on='timepoint')
    g = a.groupby('horizon', sort=False)[
        ['availability_derate', 'weights']].sum()
    derate = (g['availability_derate']/g['weights'])
    derate.name = "availability_derate"
    derate = derate.reset_index()
    derate['horizon'] = pd.to_numeric(derate['horizon'])
    return derate.set_index('horizon')['availability_derate']


def hydro_op_chars_inputs_(webdb: web.db.SqliteDB, project,
                           hydro_op_chars_sid,
                           balancing_type_project):
    othercols = ['horizon',
                 'period',
                 'balancing_type_project']
    cols = ",".join(othercols + get_power_fraction_cols())
    rows = webdb.where("inputs_project_hydro_operational_chars",
                       what=cols,
                       project=project,
                       hydro_operational_chars_scenario_id=hydro_op_chars_sid,
                       balancing_type_project=balancing_type_project).list()
    if rows:
        return pd.DataFrame(rows)
    else:
        raise common.NoEntriesError(
            f"Table inputs_project_hydro_operational_chars has no entries for project={project}, hydro_op_chars_scenario_id={hydro_op_chars_sid}, balancing_type_project={balancing_type_project}")


def copy_cols(src, dest, srccols, destcols):
    """copies filecols from src to dest
    """
    destdf = dest.copy()
    for dest_c, src_c in zip(destcols, srccols):
        destdf[dest_c] = src[src_c]
    return destdf


def get_power_fraction_cols():
    return ['average_power_fraction',
            'min_power_fraction',
            'max_power_fraction']


def get_power_fraction_file_cols():
    cols = get_power_fraction_cols()
    return ["_".join(["orig", c]) for c in cols]


def swap_original_cols_from_file(dbdf,
                                 project,
                                 hydro_op_chars_scenario_id,
                                 balancing_type_project,
                                 csv_location,
                                 description):
    def merge_with_db(filedf, dbdf):
        filedf = filedf.copy()
        dbdf = dbdf.copy()

        for c in cols:
            del dbdf[c]
            del filedf[c]

        del filedf['balancing_type_project']
        del filedf['period']
        dbdf = dbdf.merge(filedf, on="horizon")
        return dbdf.rename(columns=dict(zip(filecols, cols)))

    dbdf = dbdf.copy()
    csvpath = common.get_subscenario_csvpath(project,
                                             "hydro_operational_chars_scenario_id",
                                             hydro_op_chars_scenario_id,
                                             csv_location,
                                             description)
    if csvpath and os.path.exists(csvpath):
        filedf = pd.read_csv(csvpath)
    else:
        raise Exception(
            f"For subscenario, hydro_operational_chars_scenario_id, {hydro_op_chars_scenario_id} file is missing")

    cols = get_power_fraction_cols()
    filecols = get_power_fraction_file_cols()

    if filecols[0] in filedf.columns:
        # columns exists
        return merge_with_db(filedf, dbdf)
    else:
        # columns does not exists
        filedf = copy_cols(filedf, filedf, cols, filecols)
        allcols = ["balancing_type_project", "horizon", "period"]\
            + get_power_fraction_cols()\
            + get_power_fraction_file_cols()
        filedf.to_csv(csvpath, index=False, columns=allcols)
        filedf = filedf[filedf.balancing_type_project ==
                        balancing_type_project]
        dbdf = merge_with_db(filedf, dbdf)
    return dbdf


def hydro_op_chars_inputs(webdb, scenario, project, csv_location, description):
    hydro_op_chars_scenario_id = get_hydro_ops_chars_scenario_id(webdb,
                                                                 scenario,
                                                                 project)
    balancing_type_project = get_balancing_type(webdb, scenario)
    dbdf = hydro_op_chars_inputs_(webdb, project,
                                  hydro_op_chars_scenario_id,
                                  balancing_type_project)

    dbdf = swap_original_cols_from_file(
        dbdf, project, hydro_op_chars_scenario_id, balancing_type_project, csv_location, description)
    return dbdf


def get_capacity(webdb,
                 scenario,
                 project):
    capacity_scenario_id = get_project_specified_capacity_scenario_id(webdb,
                                                                      scenario)
    period = get_period(webdb, scenario)
    return common.get_field(webdb, "inputs_project_specified_capacity",
                            "specified_capacity_mw",
                            project=project,
                            period=period,
                            project_specified_capacity_scenario_id=capacity_scenario_id)


def get_project_specified_capacity_scenario_id(webdb, scenario):
    return common.get_field(webdb,
                            "scenarios",
                            "project_specified_capacity_scenario_id",
                            scenario_name=scenario)


def get_temporal_scenario_id(webdb, scenario):
    return common.get_field(webdb,
                            "scenarios",
                            "temporal_scenario_id",
                            scenario_name=scenario)


def get_balancing_type(webdb, scenario):
    temporal_scenario_id = get_temporal_scenario_id(webdb, scenario)
    return common.get_field(webdb, "inputs_temporal_horizons",
                            "balancing_type_horizon",
                            temporal_scenario_id=temporal_scenario_id)


def get_gross_power_mw_dataset(webdb: web.db.SqliteDB, scenario, project):
    scenario_id = common.get_field(webdb,
                                   'scenarios',
                                   "scenario_id",
                                   scenario_name=scenario)
    try:
        cols = ",".join(['period',
                         'timepoint',
                         'timepoint_weight',
                         'number_of_hours_in_timepoint',
                         'gross_power_mw'])
        rows = webdb.where("results_project_dispatch",
                           what=cols,
                           scenario_id=scenario_id,
                           project=project,
                           operational_type='gen_hydro').list()
    except sqlite3.OperationalError as oe:
        rows = webdb.where('results_project_timepoint',
                           what=cols,
                           scenario_id=scenario_id,
                           project=project,
                           operational_type='gen_hydro').list()

    return pd.DataFrame(rows)


def adjust_mean_const(b, min_, max_, force=False):
    """
    adjusts values in b such that original average of b remains as it is
    but every value of b lies between corresponding min_ and max_
    """
    def adjust(c):
        c1 = c.copy()
        less, more = c < min_, c > max_

        if less.sum() and more.sum():
            # logger.info("+-"*5)
            c1[less] += (c1[more] - max_[more]).sum()/less.sum()
            c1[more] = max_[more]
        elif more.sum():
            # logger.info("+"*5)
            can_be_increased = c < max_
            can_be_increased_count = can_be_increased.sum()
            if (can_be_increased_count > 0):
                c1[can_be_increased] += (c1[more] -
                                         max_[more]).sum()/can_be_increased_count
            else:
                logger.warning(
                    "Input data is such that its mean cannot be maintained, while adjusting between min and max limits; mean will decrease")
            c1[more] = max_[more]
        elif less.sum():
            # logger.info("-"*5)
            can_be_decreased = c > min_
            can_be_decreased_count = can_be_decreased.sum()
            if (can_be_decreased_count > 0):
                c1[can_be_decreased] -= (min_[less] -
                                         c1[less]).sum()/can_be_decreased_count
            else:
                logger.warning(
                    "Input data is such that its mean cannot be maintained, while adjusting between min and max; mean will increase")
            c1[less] = min_[less]

        # logger.info(c.mean(), c1.mean())
        return c1

    c1 = adjust(b)
    # printcols(c1, min_, max_)
    n = 0
    N = 2 * len(b)
    while n < N and not np.all((c1 >= min_) & (c1 <= max_)):
        # logger.info(f"iteration {n}..")
        c1 = adjust(c1)
        # printcols(c1, min_, max_)
        n += 1
    logger.info(f"Function adjust_mean_const, end value of counter n : {n}")
    if not np.all((c1 >= min_) & (c1 <= max_)):
        logger.info("adjust_mean_const: Failed to converge")
        if force:
            less = c1 < min_
            more = c1 > max_
            if less.any():
                logger.info("Setting forcibly some values to min")
                c1[less] = min_[less]
            if more.any():
                logger.info("Setting forcibly some values to max")
                c1[more] = max_[more]

            logger.info("Original average: {}".format(b.mean()))
            logger.info("After forcible adjustment: {}".format(c1.mean()))
    return c1


def test_adjust_mean_const():
    def generate_data(N, scale, center):
        b = np.array([center + np.random.normal(scale=scale)
                     for i in range(N)])
        logger.info(str(b.mean()))
        min_ = np.array([center - scale + np.random.normal(scale=scale/20)
                         for i in range(N)])
        max_ = np.array([center + scale + np.random.normal(scale=scale/20)
                         for i in range(N)])
        return b, min_, max_
    b, min_, max_ = generate_data(365, 5, 100.0)
    b1 = adjust_mean_const(b, min_, max_)
    assert b.mean() == pytest.approx(b1.mean())

    b, min_, max_ = generate_data(12, 5, 100.0)
    b1 = adjust_mean_const(b, min_, max_)
    assert b.mean() == pytest.approx(b1.mean())

    b, min_, max_ = generate_data(365, 0.01, 0.5)
    b1 = adjust_mean_const(b, min_, max_)
    assert b.mean() == pytest.approx(b1.mean())


def printcols(*cols):
    for i, args in enumerate(zip(*cols)):
        logger.info(f"{i:3d}" + " " + " ".join([f"{arg:.8f}" for arg in args]))


def get_projects(webdb, scenario):
    proj_ops_char_sc_id = common.get_field(webdb,
                                           "scenarios",
                                           "project_operational_chars_scenario_id",
                                           scenario_name=scenario
                                           )
    rows = webdb.where("inputs_project_operational_chars",
                       project_operational_chars_scenario_id=proj_ops_char_sc_id,
                       operational_type="gen_hydro")
    p1 = set(row['project'] for row in rows)
    p2 = set(get_portfolio_projects(webdb, scenario))
    return sorted(p1 & p2)


def get_portfolio_projects(webdb, scenario):
    project_portfolio_scenario_id = common.get_field(webdb,
                                                     "scenarios",
                                                     "project_portfolio_scenario_id",
                                                     scenario_name=scenario)
    rows = webdb.where("inputs_project_portfolios",
                       project_portfolio_scenario_id=project_portfolio_scenario_id)
    return (row['project'] for row in rows)


def match_horizons(gross_power_mw_df, hydro_opchar, timepoint_map, scenario1, scenario2):
    pass1_name = "pass1" if "pass1" in scenario1 else "pass2"
    pass1 = [c for c in timepoint_map.columns if c.startswith(
        f"{pass1_name}_timepoint")][0]
    t_map = timepoint_map.groupby(pass1).first()

    pass_name = "pass2" if "pass2" in scenario2 else "pass3"
    pass2_horizon = [
        c for c in timepoint_map.columns if c.startswith(f"{pass_name}_horizon_")][0]

    cols = [c for c in gross_power_mw_df.columns]
    rsuffix = "_other"
    dfnew = gross_power_mw_df.set_index("timepoint").join(t_map, rsuffix=rsuffix)
    if dfnew[pass2_horizon].isnull().sum() > 0:
        raise Exception("Possibly supplied timepoint map is wrong.")

    dfnew = dfnew.reset_index().set_index(pass2_horizon)
    return hydro_opchar.join(dfnew, lsuffix="_X").rename(columns={'horizon': "horizon_"})


def reduce_size(df, scenario1, scenario2, timepoint_map):
    """
    """
    pass1_name = "pass1" if "pass1" in scenario1 else "pass2"
    pass1 = [c for c in timepoint_map.columns if c.startswith(
        f"{pass1_name}_timepoint")][0]
    t_map = timepoint_map.groupby(pass1).first()

    pass_name = "pass2" if "pass2" in scenario2 else "pass3"
    pass2_horizon = [
        c for c in timepoint_map.columns if c.startswith(f"{pass_name}_horizon_")][0]

    cols = [c for c in df.columns]
    rsuffix = "_other"
    dfnew = df.set_index("timepoint").join(t_map, rsuffix=rsuffix)

    weight = dfnew["number_of_hours_in_timepoint"]
    dfnew['gross_power_mw_x'] = dfnew['gross_power_mw'] * weight
    dfnew = dfnew.reset_index()
    if dfnew[pass2_horizon].isnull().sum() > 0:
        raise Exception("Possibly supplied timepoint map is wrong.")
    grouped = dfnew.groupby(pass2_horizon).sum()
    grouped['gross_power_mw'] = grouped['gross_power_mw_x'] /\
        grouped["number_of_hours_in_timepoint"]

    # logger.info(grouped)
    return grouped.reset_index().rename(columns={"horizon": "horizon_", pass2_horizon: "horizon"})


def compute_adjusted_min_max(derate, avg, min_, max_):
    derate = np.fmax(np.fmin(derate, 1), 0)

    max_ = np.fmax(np.fmin(derate, max_), 0)
    min_ = np.fmax(np.fmin(min_, max_), 0)

    avg = np.fmax(np.fmin(avg, 1), 0)

    return derate, avg, min_, max_


def compute_adjusted_variables(derate, avg, min_, max_):
    min_ = np.where(derate <= 1e-6, min_, np.fmin(min_/derate, 1))
    max_ = np.where(derate <= 1e-6, max_, np.fmin(max_/derate, 1))
    avg = np.fmax(np.fmin(np.where(derate <= 1e-6, 0, np.fmin(avg/derate, 1)),
                          max_),
                  min_)
    return avg, min_, max_


def test_compute_adjusted_variables():
    avg = pd.Series([0.5, 0.8, 0.8, 0.7, 0.1])
    min_ = pd.Series([0.2, 0, 0, 0.2, 0.2])
    max_ = pd.Series([0.6, 1, 1, 0.6, 0.6])

    derate = pd.Series([0.75, 0.75, 0, 0.75, 0.75])

    avg, min_, max_ = compute_adjusted_variables(derate,
                                                 avg,
                                                 min_,
                                                 max_)
    assert pytest.approx(avg) == [0.666666666666667,
                                  1, 0, 0.8, 0.266666666666667]
    assert pytest.approx(min_) == [
        0.266666666666667, 0, 0, 0.266666666666667, 0.266666666666667]
    assert pytest.approx(max_) == [0.8, 1, 1, 0.8, 0.8]


def get_timepoint_horizon_map(scenario, timepoint_map):
    pass_name = "pass2" if "pass2" in scenario else "pass3"

    timepoint_col_name = [c for c in timepoint_map.columns
                          if c.startswith(f"{pass_name}_timepoint")][0]
    horizon_col_name = [c for c in timepoint_map.columns
                        if c.startswith(f"{pass_name}_horizon_")][0]
    timepoint_horizon_map = timepoint_map[[
        timepoint_col_name, horizon_col_name]].drop_duplicates()

    timepoint_horizon_map.rename(columns={timepoint_col_name: "timepoint", horizon_col_name: "horizon"},
                                 inplace=True)
    return timepoint_horizon_map


def availability_adjustment(webdb, scenario, project, hydro_op, timepoint_map):
    # if availability inputs are not available then return avg, min_, max_ as it is
    try:
        a = read_exogenous_availabilty_results(webdb, scenario, project)
    except common.NoEntriesError as e:
        logger.warning(
            f"Availabiity inputs not available for {scenario}/{project}")
        logger.warning(f"Skipping availability adjustment for {project}")
        df = hydro_op.reset_index()
        df['availability_derate'] = 1
        return compute_adjusted_min_max(df['availability_derate'],
                                        df['cuf'],
                                        df['min_power_fraction'],
                                        df['max_power_fraction'])
    it = read_inputs_temporal(webdb, scenario)
    itht = get_timepoint_horizon_map(scenario, timepoint_map)
    derate = compute_availability(a, it, itht)
    df = pd.merge(hydro_op.reset_index(), derate.reset_index())
    return compute_adjusted_min_max(df['availability_derate'],
                                    df['cuf'],
                                    df['min_power_fraction'],
                                    df['max_power_fraction'])


def organise_results(hydro_op, cols, avg, min_, max_):
    results = hydro_op.reset_index()[cols].copy()
    results['average_power_fraction'] = avg
    results['min_power_fraction'] = min_
    results['max_power_fraction'] = max_
    return results


def get_horizon_count_dict(scenario1, scenario2, timepoint_map):
    pass1_name = "pass1" if "pass1" in scenario1 else "pass2"
    pass2_name = "pass2" if "pass2" in scenario2 else "pass3"

    horizon1_col_name = [c for c in timepoint_map.columns
                         if c.startswith(f"{pass1_name}_horizon_")][0]
    horizon2_col_name = [c for c in timepoint_map.columns
                         if c.startswith(f"{pass2_name}_horizon_")][0]

    horizon1_horizon2_map = timepoint_map[[
        horizon1_col_name, horizon2_col_name]].drop_duplicates()

    horizon1_count_df = horizon1_horizon2_map.groupby(
        horizon1_col_name).count()

    horizon1_count_dict = horizon1_count_df.to_dict()[horizon2_col_name]
    return horizon1_count_dict


def adjusted_mean_results(webdb,
                          scenario1,
                          scenario2,
                          project,
                          mapfile,
                          csv_location,
                          description):
    """adjusting of mean happens based on

    Assumption: The columns min,max,avg(cuf) are assumed to be in
    the same order (ascending/descending) of horizon. If it is different,
    then we need to adjust the order befor doing calculation!
    """
    cols = ["balancing_type_project", "horizon", "period",
            "average_power_fraction", "min_power_fraction", "max_power_fraction"]
    df0 = hydro_op_chars_inputs(webdb, scenario2, project,
                                csv_location, description)
    gross_power_mw_df = get_gross_power_mw_dataset(webdb, scenario1, project)
    capacity = get_capacity(webdb, scenario1, project)
    if abs(capacity) <= 1e-8:
        raise Exception("Capacity is zero or very small!")
    cuf = gross_power_mw_df['gross_power_mw']/capacity
    weight = gross_power_mw_df['number_of_hours_in_timepoint']
    prd = gross_power_mw_df['period'].unique()[0]
    hydro_op = df0[df0.period == prd].reset_index(
        drop=True).set_index("horizon")
    min_, max_ = [hydro_op[c] for c in cols[-2:]]
    timepoint_map = pd.read_excel(mapfile,
                                  sheet_name="map",
                                  skiprows=2,
                                  engine="openpyxl")

    if len(cuf) > len(min_):
        gross_power_mw_df = reduce_size(
            gross_power_mw_df, scenario1, scenario2, timepoint_map)
        gross_power_mw_df = gross_power_mw_df.set_index("horizon")
        hydro_op = hydro_op.join(gross_power_mw_df, rsuffix="right")
    elif len(cuf) < len(min_):
        Exception("gross_power_mw needs to expand in size. Code does not handle it!")
    else:
        hydro_op = match_horizons(
            gross_power_mw_df, hydro_op, timepoint_map, scenario1, scenario2)

    min_, max_ = [hydro_op[c] for c in cols[-2:]]
    hydro_op['cuf'] = hydro_op['gross_power_mw']/capacity
    weight = hydro_op.reset_index()['number_of_hours_in_timepoint']

    derate, avg, min_, max_ = availability_adjustment(
        webdb, scenario2, project, hydro_op, timepoint_map)

    horizon_count_dict = get_horizon_count_dict(
        scenario1, scenario2, timepoint_map)

    prev = 0
    for horizon, count in horizon_count_dict.items():
        start_index = prev
        stop_index = start_index + count

        avg[start_index: stop_index] = adjust_mean_const(avg[start_index: stop_index] * weight[start_index: stop_index],
                                                         min_[
                                                             start_index: stop_index] * weight[start_index: stop_index],
                                                         max_[
                                                             start_index: stop_index] * weight[start_index: stop_index],
                                                         force=True) / weight[start_index: stop_index]
        prev = stop_index

    avg, min_, max_ = compute_adjusted_variables(derate, avg, min_, max_)
    results = organise_results(hydro_op, cols, avg, min_, max_)
    return results


def get_hydro_ops_chars_scenario_id(webdb, scenario, project):
    pocs_id = common.get_field(webdb,
                               "scenarios",
                               "project_operational_chars_scenario_id",
                               scenario_name=scenario)

    return common.get_field(webdb,
                            "inputs_project_operational_chars",
                            "hydro_operational_chars_scenario_id",
                            project_operational_chars_scenario_id=pocs_id,
                            project=project)


def write_results_csv(results,
                      project,
                      subscenario,
                      subscenario_id,
                      csv_location,
                      description):
    csvpath = common.get_subscenario_csvpath(project, subscenario,
                                             subscenario_id, csv_location, description)
    cols = ["balancing_type_project", "horizon", "period"]
    cols += get_power_fraction_cols()

    on = 'horizon'

    common.merge_in_csv(results, csvpath, cols, on)
    return subscenario, subscenario_id


def hydro_op_chars(database,
                   csv_location,
                   gridpath_repo,
                   scenario1,
                   scenario2,
                   description,
                   project,
                   mapfile,
                   update_database):
    webdb = common.get_database(database)
    projects = get_projects(webdb, scenario1)
    if project:
        projects = [project]  # projects[:1]

    subscenario = "hydro_operational_chars_scenario_id"
    for project_ in projects:
        logger.info(f"Computing data for {project_}")
        subscenario_id = get_hydro_ops_chars_scenario_id(
            webdb, scenario2, project_)

        results = adjusted_mean_results(webdb,
                                        scenario1,
                                        scenario2,
                                        project_,
                                        mapfile,
                                        csv_location,
                                        description)

        write_results_csv(results,
                          project_,
                          subscenario,
                          subscenario_id,
                          csv_location,
                          description)
        if update_database:
            common.update_subscenario_via_gridpath(subscenario,
                                                   subscenario_id,
                                                   project_,
                                                   csv_location,
                                                   database,
                                                   gridpath_repo)


@click.command()
@click.option("-d", "--database", default="../toy.db", help="Path to database")
@click.option("-c", "--csv_location", default="csvs_toy", help="Path to folder where csvs are")
@click.option("-g", "--gridpath_repo", default="../", help="Path of gridpath source repository")
@click.option("-s1", "--scenario1", default="toy1_pass1", help="Name of scenario1")
@click.option("-s2", "--scenario2", default="toy1_pass2", help="Name of scenario2")
@click.option("-m", "--description", default="rpo50S3_all", help="Description for csv files.")
@click.option("--project", default=None, help="Run for only one project")
@click.option("-t", "--timepoint_map", default="timepoint_map.xlsx", help="Excel file of timepoint map")
@click.option("--update_database/--no-update_database", default=False, help="Update database only if this flag is True")
@click.option("-l", "--loglevel", default="INFO", help="Loglevel one of INFO,WARN,DEBUG,ERROR")
def main(database,
         csv_location,
         gridpath_repo,
         scenario1,
         scenario2,
         description,
         project,
         timepoint_map,
         update_database,
         loglevel
         ):
    global logger

    init_logger("hydro.log", loglevel)
    logger = logging.getLogger("hydro_opchars")

    hydro_op_chars(database,
                   csv_location,
                   gridpath_repo,
                   scenario1,
                   scenario2,
                   description,
                   project,
                   timepoint_map,
                   update_database)


if __name__ == "__main__":
    main()
