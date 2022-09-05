import web
import pandas as pd
import numpy as np
import common
import os
import click


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

        #print(c.mean(), c1.mean())
        return c1

    c1 = adjust(b)
    #printcols(c1, min_, max_)
    n = 0
    N = 30
    while n < N and not np.all((c1 >= min_) & (c1 <= max_)):
        #print(f"iteration {n}..")
        c1 = adjust(c1)
        #printcols(c1, min_, max_)
        n += 1
    if n == N:
        print("adjust_mean_const: Failed to converge")
        if force:
            less = c1 < min_
            more = c1 > max_
            if less:
                print("Setting focibly some values to min")
                c1[less] = min_[less]
            if more:
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


def reduce_size(webdb, df, scenario, mapfile):
    """
    """
    timepoint_map = pd.read_excel(mapfile,
                                  sheet_name="map",
                                  skiprows=2,
                                  engine="openpyxl")
    pass1 = [c for c in timepoint_map.columns if c.startswith(
        "pass1_timepoint_")][0]
    t_map = timepoint_map.groupby(pass1).first()

    pass_name = "pass2" if "pass2" in scenario else "pass3"
    pass2_horizon = [
        c for c in timepoint_map.columns if c.startswith(f"{pass_name}_horizon_")][0]

    cols = [c for c in df.columns]
    rsuffix = "_other"
    dfnew = df.set_index("timepoint").join(t_map, rsuffix=rsuffix)
    
    weight = dfnew["number_of_hours_in_timepoint"]
    dfnew['power_mw_x'] = dfnew['power_mw'] * weight
    dfnew = dfnew.reset_index()
    grouped = dfnew.groupby(pass2_horizon).sum()
    grouped['power_mw'] = grouped['power_mw_x'] / \
        grouped["number_of_hours_in_timepoint"]

    #print(grouped)                 
    return grouped.reset_index(drop=True)


def adjusted_mean_results(webdb, scenario1, scenario2, project, mapfile):
    cols = ["balancing_type_project", "horizon", "period",
            "average_power_fraction", "min_power_fraction", "max_power_fraction"]
    df0 = hydro_op_chars_inputs(webdb, scenario2, project)
    power_mw_df = get_power_mw_dataset(webdb, scenario1, project)
    capacity = get_capacity(webdb, scenario1, project)
    cuf = power_mw_df['power_mw']/capacity
    weight = power_mw_df['number_of_hours_in_timepoint']
    prd = power_mw_df['period'].unique()[0]
    df = df0[df0.period == prd].reset_index(drop = True)
    min_, max_ = [df[c] for c in cols[-2:]]

    if len(cuf) > len(min_):
        power_mw_df = reduce_size(webdb, power_mw_df, scenario2, mapfile)
        cuf = power_mw_df['power_mw']/capacity
        weight = power_mw_df['number_of_hours_in_timepoint']


    avg = adjust_mean_const(cuf*weight, min_*weight, max_*weight)/weight
    avg = adjust_mean_const(avg, min_, max_, force=True)    
    results = df[cols].copy()

    del results['average_power_fraction']
    results['average_power_fraction'] = avg

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
    webdb = common.get_database("/home/vikrant/programming/work/publicgit/gridpath/db/toy2.db")
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
    #hydro_dispatch = hydro_dispatch.dropna(axis=0)
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
