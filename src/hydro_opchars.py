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
        raise common.NoEntriesError(f"Table inputs_project_hydro_operational_chars has no entries for project={project}, hydro_op_chars_scenario_id={hydro_op_chars_sid}, balancing_type_project={balancing_type_project}")


def hydro_op_chars_inputs(webdb, scenario, project):
    hydro_op_chars_scenario_id = get_hydro_ops_chars_sceanario_id(webdb,
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
    return common.get_field(webdb , "inputs_project_specified_capacity",
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


def get_temporal_start_end_table(conn, scenario):
    temporal_id = get_temporal_scenario_id(conn, scenario)
    temporal = conn.where("inputs_temporal_horizon_timepoints_start_end",
                          temporal_scenario_id=temporal_id).list()
    return temporal




def get_power_mw_dataset(webdb, scenario, project):
    scenario_id = common.get_field(webdb,
                                   'scenarios',
                                   "scenario_id",
                                   scenario_name = scenario)
    rows = webdb.where("results_project_dispatch",
                       scenario_id=scenario_id,
                       project=project,
                       operational_type='gen_hydro').list()
    
    return pd.DataFrame(rows)


def adjust_mean_const(b, min_, max_):
    """
    adjusts values in b such that original average of b remains as it is
    but every value of b lied between corresponding min_ and max_
    """
    def adjust(c):
        c1 = c.copy()
        less, more, between = c < min_, c > max_, (c >= min_) & (c <= max_)

        if less.sum() and more.sum():
            #print("+-"*5)
            c1[less] += (c1[more]- max_[more]).sum()/less.sum()
            c1[more] = max_[more]
        elif more.sum():
            #print("+"*5)
            c1[between] += (c1[more] - max_[more]).sum()/between.sum()
            c1[more] = max_[more]
        elif less.sum():
            #print("-"*5)
            c1[between] -= (min_[less] - c1[less]).sum()/between.sum()
            c1[less] = min_[less]
        

        #print(c.mean(), c1.mean())
        return c1
    
    c1 = adjust(b)
    #printcols(c1, min_, max_)
    n = 0
    while n <20 and not np.all((c1 >= min_) & (c1 <= max_)):
        #print(f"iteration {n}..")
        c1 = adjust(c1)
        #printcols(c1, min_, max_)
        n += 1
    if n ==20:
        print("Failed to adjust mean")
    
    #print(b.mean(), c1.mean())
    return c1


def printcols(*cols):
    for i, args in enumerate(zip(*cols)):
        print(f"{i:3d}", " ".join([f"{arg:5f}" for arg in args]))


def get_projects(webdb, scenario):
    proj_ops_char_sc_id = common.get_field(webdb,
                                    "scenarios",
                                    "project_operational_chars_scenario_id",
                                    scenario_name=scenario
                                    )
    rows =  webdb.where("inputs_project_operational_chars",
                        project_operational_chars_scenario_id=proj_ops_char_sc_id,
                        operational_type="gen_hydro")
    return [row['project'] for row in rows]




def reduce_size(webdb, df, scenario):
    tmp1 = get_temporal_start_end_table(webdb, scenario)

    horizon = [] 
    for row in df.to_dict(orient="records"):
        x = row['timepoint']
        horizon.append([p['horizon'] for p in tmp1 if x >= p['tmp_start'] and x <= p['tmp_end']][0])
    df['horizon'] = horizon
    
    grouped = df.groupby('horizon').mean()
    grouped.reset_index(inplace=True, drop=True)
    return grouped

def adjusted_mean_results(webdb, scenario1, scenario2, project):
    cols = ["balancing_type_project", "horizon", "period",
            "average_power_fraction","min_power_fraction", "max_power_fraction"]
    df = hydro_op_chars_inputs(webdb, scenario2, project)
    power_mw_df = get_power_mw_dataset(webdb, scenario1, project)
    capacity = get_capacity(webdb, scenario1, project)
    cuf =  power_mw_df['power_mw']/capacity
    min_, max_ = [df[c] for c in cols[-2:]]

    if len(cuf) > len(min_):
        power_mw_df = reduce_size(webdb, power_mw_df, scenario2)
        cuf = power_mw_df['power_mw']/capacity
        
    avg = adjust_mean_const(cuf, min_, max_)
    results = df[cols]
    del results['average_power_fraction']
    results['average_power_fraction'] = avg
    return results


def get_hydro_ops_chars_sceanario_id(webdb, scenario, project):
    pocs_id = common.get_field(webdb,
                               "scenarios",
                               "project_operational_chars_scenario_id",
                               scenario_name=scenario)
    
    return common.get_field(webdb,
                            "inputs_project_operational_chars",
                            "hydro_operational_chars_scenario_id",
                            project_operational_chars_scenario_id=pocs_id,
                            project = project)


def write_results_csv(results,
                      project,
                      subscenario,
                      subscenario_id,
                      csv_location,
                      description):
    csvpath = common.get_subscenario_csvpath(project, subscenario,
                                             subscenario_id, csv_location, description)
    cols = ["balancing_type_project", "horizon", "period",
            "average_power_fraction","min_power_fraction", "max_power_fraction"]
    on = 'horizon'

    common.merge_in_csv(results, csvpath, cols, on)
    return subscenario, subscenario_id


def hydro_op_chars(scenario1,
                   scenario2,
                   csv_location,
                   database,
                   gridpath_rep,
                   project,
                   update_database,
                   description):
    webdb = common.get_database(database)
    projects = get_projects(webdb, scenario1)
    if project:
        projects = [project]#projects[:1]

    subscenario = "hydro_operational_chars_scenario_id"
    for project_ in projects:
        print(f"Computing data for {project_}")
        subscenario_id = get_hydro_ops_chars_sceanario_id(webdb, scenario2, project_)

        results = adjusted_mean_results(webdb, scenario1, scenario2, project_)
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
                                                   gridpath_rep)
        
        
@click.command()
@click.option("-s1", "--scenario1", default="toy1_pass1", help="Name of scenario1")
@click.option("-s2", "--scenario2", default="toy1_pass2", help="Name of scenario2")
@click.option("-c", "--csv_location", default="csvs_toy", help="Path to folder where csvs are")
@click.option("-d", "--database", default="../toy.db", help="Path to database")
@click.option("-g", "--gridpath_rep", default="../", help="Path of gridpath source repository")
@click.option("--project", default=None, help="Run for only one project")
@click.option("--update_database/--no-update_database", default=False, help="Update database only if this flag is True")
@click.option("-m", "--description", default="rpo50S3_all", help="Description for csv files.")
def main(scenario1,
         scenario2,
         csv_location,
         database,
         gridpath_rep,
         project,
         update_database,
         description
         ):

    hydro_op_chars(scenario1,
                   scenario2,
                   csv_location,
                   database,
                   gridpath_rep,
                   project,
                   update_database,
                   description)

def dbtest():    
    webdb = common.get_database("/home/vikrant/programming/work/publicgit/gridpath/mh.db")
    scenario1 = "rpo30_pass1"
    scenario2 = 'rpo30_pass2'
    project = 'Koyna_Stage_3'
    adjusted_mean_results(webdb, scenario1, scenario2, project)


    
def test_1():
    datapath = "/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/project/opchar/hydro_opchar/hydro-daily-limits-rpo30.xlsx"
    project = 'Koyna_Stage_1'
    hydro_dispatch = pd.read_excel(datapath, sheet_name=project, nrows=365)
    #hydro_dispatch = hydro_dispatch.dropna(axis=0)
    b = hydro_dispatch['avg']
    min_ = hydro_dispatch['min']
    max_ = hydro_dispatch['max']
    b1 = adjust_mean_const(b, min_, max_)
    printcols(b, b1)
    assert np.all(abs(b - b1) <= 0.001)

    
def test_compare_with_db():

    def get_db_results():
        subscenario_id = get_hydro_ops_chars_sceanario_id(webdb,
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

    webdb = common.get_database(database)
    subscenario = "hydro_operational_chars_scenario_id"
    print(f"Computing data for {project}")
    subscenario_id = get_hydro_ops_chars_sceanario_id(webdb, scenario2, project)
    results = adjusted_mean_results(webdb, scenario1, scenario2, project)
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
    dailyfile = filedata[filedata.balancing_type_project=="month"]
    b1 = dailyfile['average_power_fraction']
    
    dbdata = get_db_results()
    dbdata.set_index('horizon', inplace=True)
    dailydb = dbdata[dbdata.balancing_type_project=="month"]
    b2 = dailydb['average_power_fraction']
    
    printcols(b1, b2, b1.index, b2.index)
    diff = abs(b1-b2)>= 0.001
    printcols(b1[diff], b2[diff], b1.index[diff], b2.index[diff])
    assert np.all(abs(b1-b2)<=0.001)

    
    
    
#test_compare_with_db()

if __name__ == "__main__":
    main()
