import web
import pandas as pd
import numpy as np
import common
import os
import click
import availability
import pytest


def read_exogenous_availabilty_results(webdb, scenario, project):
    exo_avail_id = availability.get_exogenous_avail_id(
        webdb, scenario, project)
    table = "inputs_project_availability_exogenous"
    return common.filtered_table(webdb,
                                 table,
                                 project=project,
                                 exogenous_availability_scenario_id=exo_avail_id)


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
    a = availability_data.merge(inputs_temporal[inputs_temporal["spinup_or_lookahead"] == 0], on="timepoint")
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


def hydro_op_chars_inputs_(webdb, project,
                           hydro_op_chars_sid,
                           balancing_type_project):
    rows = webdb.where("inputs_project_hydro_operational_chars",
                       project=project,
                       hydro_operational_chars_scenario_id=hydro_op_chars_sid,
                       balancing_type_project=balancing_type_project).list()
    if rows:
        return pd.DataFrame(rows)
    else:
        raise common.NoEntriesError(
            f"Table inputs_project_hydro_operational_chars has no entries for project={project}, hydro_op_chars_scenario_id={hydro_op_chars_sid}, balancing_type_project={balancing_type_project}")


def hydro_op_chars_inputs(webdb, scenario, project):
    hydro_op_chars_scenario_id = get_hydro_ops_chars_scenario_id(webdb,
                                                                 scenario, project)
    balancing_type_project = get_balancing_type(webdb, scenario)
    return hydro_op_chars_inputs_(webdb, project,
                                  hydro_op_chars_scenario_id,
                                  balancing_type_project)


def get_capacity(webdb,
                 scenario,
                 project):
    capacity_scenario_id = get_project_specified_capacity_scenario_id(webdb,
                                                                      scenario)
    return common.get_field(webdb, "inputs_project_specified_capacity",
                            "specified_capacity_mw",
                            project=project,
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


def get_power_mw_dataset(webdb, scenario, project):
    scenario_id = common.get_field(webdb,
                                   'scenarios',
                                   "scenario_id",
                                   scenario_name=scenario)
    rows = webdb.where("results_project_dispatch",
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
        less, more, between = c < min_, c > max_, (c >= min_) & (c <= max_)

        if less.sum() and more.sum():
            # print("+-"*5)
            c1[less] += (c1[more] - max_[more]).sum()/less.sum()
            c1[more] = max_[more]
        elif more.sum():
            # print("+"*5)
            c1[between] += (c1[more] - max_[more]).sum()/between.sum()
            c1[more] = max_[more]
        elif less.sum():
            # print("-"*5)
            c1[between] -= (min_[less] - c1[less]).sum()/between.sum()
            c1[less] = min_[less]

        # print(c.mean(), c1.mean())
        return c1

    c1 = adjust(b)
    # printcols(c1, min_, max_)
    n = 0
    N = 400
    while n < N and not np.all((c1 >= min_) & (c1 <= max_)):
        # print(f"iteration {n}..")
        c1 = adjust(c1)
        # printcols(c1, min_, max_)
        n += 1
    if n == N:
        print("adjust_mean_const: Failed to converge")
        if force:
            less = c1 < min_
            more = c1 > max_
            if less.any():
                print("Setting focibly some values to min")
                c1[less] = min_[less]
            if more.any():
                print("Setting focibly some values to min")
                c1[more] = max_[more]

            print("Original average:", b.mean())
            print("After forcible adjustment:", c1.mean())
    return c1


def printcols(*cols):
    for i, args in enumerate(zip(*cols)):
        print(f"{i:3d}", " ".join([f"{arg:.8f}" for arg in args]))


def get_projects(webdb, scenario):
    proj_ops_char_sc_id = common.get_field(webdb,
                                           "scenarios",
                                           "project_operational_chars_scenario_id",
                                           scenario_name=scenario
                                           )
    rows = webdb.where("inputs_project_operational_chars",
                       project_operational_chars_scenario_id=proj_ops_char_sc_id,
                       operational_type="gen_hydro")
    return [row['project'] for row in rows]


def match_horizons(power_mw_df, hydro_opchar, timepoint_map, scenario1, scenario2):
    pass1_name = "pass1" if "pass1" in scenario1 else "pass2"
    pass1 = [c for c in timepoint_map.columns if c.startswith(
        f"{pass1_name}_timepoint")][0]
    t_map = timepoint_map.groupby(pass1).first()

    pass_name = "pass2" if "pass2" in scenario2 else "pass3"
    pass2_horizon = [
        c for c in timepoint_map.columns if c.startswith(f"{pass_name}_horizon_")][0]

    cols = [c for c in power_mw_df.columns]
    rsuffix = "_other"
    dfnew = power_mw_df.set_index("timepoint").join(t_map, rsuffix=rsuffix)
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
    dfnew['power_mw_x'] = dfnew['power_mw'] * weight
    dfnew = dfnew.reset_index()
    if dfnew[pass2_horizon].isnull().sum() > 0:
        raise Exception("Possibly supplied timepoint map is wrong.")
    grouped = dfnew.groupby(pass2_horizon).sum()
    grouped['power_mw'] = grouped['power_mw_x'] / \
        grouped["number_of_hours_in_timepoint"]

    # print(grouped)
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
    timepoint_horizon_map = timepoint_map[[timepoint_col_name, horizon_col_name]].drop_duplicates()

    timepoint_horizon_map.rename(columns = {timepoint_col_name: "timepoint", horizon_col_name: "horizon"},
                                 inplace = True)    
    return timepoint_horizon_map


def availability_adjustment(webdb, scenario, project, hydro_op, timepoint_map):
    # if availability inputs are not available then return avg, min_, max_ as it is
    try:
        a = read_exogenous_availabilty_results(webdb, scenario, project)
    except common.NoEntriesError as e:
        print(
            f"Warning: Availabiity inputs not available for {scenario}/{project}")
        print(f"Skipping availability adjustment for {project}")
        df = hydro_op.reset_index()
        return df['cuf'], df['min_power_fraction'], df['max_power_fraction']
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


def adjusted_mean_results(webdb, scenario1, scenario2, project, mapfile):
    """adjusting of mean happens based on 

    Assumption: The columns min,max,avg(cuf) are assumed to be in 
    the same order (ascending/descending) of horizon. If it is different,
    then we need to adjust the order befor doing calculation!
    """
    cols = ["balancing_type_project", "horizon", "period",
            "average_power_fraction", "min_power_fraction", "max_power_fraction"]
    df0 = hydro_op_chars_inputs(webdb, scenario2, project)
    power_mw_df = get_power_mw_dataset(webdb, scenario1, project)
    capacity = get_capacity(webdb, scenario1, project)
    cuf = power_mw_df['power_mw']/capacity
    weight = power_mw_df['number_of_hours_in_timepoint']
    prd = power_mw_df['period'].unique()[0]
    hydro_op = df0[df0.period == prd].reset_index(
        drop=True).set_index("horizon")
    min_, max_ = [hydro_op[c] for c in cols[-2:]]
    timepoint_map = pd.read_excel(mapfile,
                                  sheet_name="map",
                                  skiprows=2,
                                  engine="openpyxl")

    if len(cuf) > len(min_):
        power_mw_df = reduce_size(
            power_mw_df, scenario1, scenario2, timepoint_map)
        power_mw_df = power_mw_df.set_index("horizon")
        hydro_op = hydro_op.join(power_mw_df, rsuffix="right")
    elif len(cuf) < len(min_):
        Exception("power_mw needs to expand in size. Code does not handle it!")
    else:
        hydro_op = match_horizons(
            power_mw_df, hydro_op, timepoint_map, scenario1, scenario2)

    min_, max_ = [hydro_op[c] for c in cols[-2:]]
    hydro_op['cuf'] = hydro_op['power_mw']/capacity
    weight = hydro_op.reset_index()['number_of_hours_in_timepoint']

    derate, avg, min_, max_ = availability_adjustment(
        webdb, scenario2, project, hydro_op, timepoint_map)
    avg = adjust_mean_const(avg*weight, min_*weight, max_*weight)/weight
    avg = adjust_mean_const(avg*weight, min_*weight, max_*weight, force=True)/weight
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
    cols = ["balancing_type_project", "horizon", "period",
            "average_power_fraction", "min_power_fraction", "max_power_fraction"]
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
        print(f"Computing data for {project_}")
        subscenario_id = get_hydro_ops_chars_scenario_id(
            webdb, scenario2, project_)

        results = adjusted_mean_results(
            webdb, scenario1, scenario2, project_, mapfile)

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
def main(database,
         csv_location,
         gridpath_repo,
         scenario1,
         scenario2,
         description,
         project,
         timepoint_map,
         update_database
         ):

    hydro_op_chars(database,
                   csv_location,
                   gridpath_repo,
                   scenario1,
                   scenario2,
                   description,
                   project,
                   timepoint_map,
                   update_database)


def dbtest():
    # TODO: remove references to local path
    webdb = common.get_database(
        "/home/vikrant/programming/work/publicgit/gridpath/db/toy2.db")
    scenario1 = "FY40_RE80_pass3_auto_pass1"
    scenario2 = 'FY40_RE80_pass3_auto_pass2'
    project = 'Bhira'
    timepoint_map = "/home/vikrant/programming/work/publicgit/gridpath/db/timepoint_map_2040.xlsx"
    return adjusted_mean_results(webdb, scenario1, scenario2, project, timepoint_map)


def test_1():
    datapath = "/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/project/opchar/hydro_opchar/hydro-daily-limits-rpo30.xlsx"
    project = 'Koyna_Stage_1'
    hydro_dispatch = pd.read_excel(
        datapath, sheet_name=project, nrows=365, engine="openpyxl")
    # hydro_dispatch = hydro_dispatch.dropna(axis=0)
    b = hydro_dispatch['avg']
    min_ = hydro_dispatch['min']
    max_ = hydro_dispatch['max']
    b1 = adjust_mean_const(b, min_, max_)
    printcols(b, b1)
    assert np.all(abs(b - b1) <= 0.001)


def test_compare_with_db():

    def get_db_results():
        subscenario_id = get_hydro_ops_chars_scenario_id(webdb,
                                                         compare_scenario,
                                                         project)
        rows = webdb.where("inputs_project_hydro_operational_chars",
                           project=project,
                           hydro_operational_chars_scenario_id=subscenario_id).list()
        return pd.DataFrame(rows)

    gridpath = "/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath"
    database = os.path.join(gridpath, "mh.db")
    csv_location = os.path.join(gridpath, "db", "csvs_mh")

    scenario1 = "rpo50S3_pass1_2"
    scenario2 = "rpo50S3_pass2"
    compare_scenario = "rpo50S3_pass2"
    project = "Sardar_Sarovar_CHPH"
    timepoint_map = "/home/vikrant/Downloads/timepoint_map.xlsx"

    webdb = common.get_database(database)
    subscenario = "hydro_operational_chars_scenario_id"
    print(f"Computing data for {project}")
    subscenario_id = get_hydro_ops_chars_scenario_id(webdb, scenario2, project)
    results = adjusted_mean_results(
        webdb, scenario1, scenario2, project, timepoint_map)
    write_results_csv(results,
                      project,
                      subscenario,
                      subscenario_id,
                      csv_location,
                      "rpo50S3_all")

    csvpath = common.get_subscenario_csvpath(project, subscenario,
                                             subscenario_id, csv_location, "rpo50S3_all")
    filedata = pd.read_csv(csvpath)
    filedata.set_index("horizon", inplace=True)
    dailyfile = filedata[filedata.balancing_type_project == "month"]
    b1 = dailyfile['average_power_fraction']

    dbdata = get_db_results()
    dbdata.set_index('horizon', inplace=True)
    dailydb = dbdata[dbdata.balancing_type_project == "month"]
    b2 = dailydb['average_power_fraction']

    printcols(b1, b2, b1.index, b2.index)
    diff = abs(b1-b2) >= 0.001
    printcols(b1[diff], b2[diff], b1.index[diff], b2.index[diff])
    assert np.all(abs(b1-b2) <= 0.001)


# test_compare_with_db()
if __name__ == "__main__":
    main()
