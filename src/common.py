import web
import pandas as pd
import numpy as np
import os
import subprocess
import logging

logger = logging.getLogger(__name__)


class NoEntriesError(Exception):
    pass


def get_field(webdb, table, field, **conds):
    r = webdb.where(table, **conds).first()

    if not r or field not in r:
        raise NoEntriesError(f"Field {field} not found in {table} for {conds}")
    if not r[field]:
        logger.warn(
            f"{field} from {table} for {conds} is empty or None")
    return r[field]


def get_table_dataframe(webdb: web.db.SqliteDB, table: str) -> pd.DataFrame:
    """
    read table as dataframe. 
    """
    t = pd.DataFrame(webdb.select(table).list())
    if t.shape[0] > 0:
        return t
    else:
        raise NoEntriesError(f"Table {table} is empty")


def filtered_table(webdb, table, **conds):
    rows = webdb.where(table, **conds).list()
    if rows:
        return pd.DataFrame(rows)
    else:
        raise NoEntriesError(f"No entries in {table} for {conds}")


def get_database(db_path):
    db = web.database("sqlite:///" + db_path)
    db.printing = False
    return db


def get_master_csv_path(csv_location, filename="csv_data_master.csv"):
    return os.path.join(csv_location, filename)


def get_subscenario_path(csv_location, subscenario):
    csvmasterpath = get_master_csv_path(csv_location)
    if not os.path.exists(csvmasterpath):
        csvmasterpath = get_master_csv_path(csv_location, "csv_structure.csv")
    csvmaster = pd.read_csv(csvmasterpath)
    row = csvmaster[csvmaster.subscenario == subscenario].iloc[0]
    return os.path.join(csv_location, row['path'])


def get_subscenario_csvpath(project,
                            subscenario,
                            subscenario_id,
                            csv_location, description="description"):
    path = get_subscenario_path(csv_location, subscenario)
    csvs = [f for f in os.listdir(path) if f.startswith(
        f"{project}-{subscenario_id}")]
    if csvs:
        return os.path.join(path, csvs[0])
    else:
        # TODO : This is not clean.
        if subscenario == 'exogenous_availability_scenario_id':
            logger.info(f"CSV not found for {project}-{subscenario_id}")
            filename = f"{project}-{subscenario_id}-{description}.csv"
            logger.info(f"Creating  {filename}")
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

    cmd = " ".join(["python", script, args])
    logger.info(cmd)
    p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate("y".encode())
    logger.info(stdout.decode())
    e = stderr.decode().strip()
    if e:
        logger.error(e)
    # out_bytes = subprocess.check_output(cmd, shell=True)
    # logger.info(out_bytes.decode())


def run_scenario(scenario,
                 scenario_location,
                 db_path):
    cmd = "gridpath_run_e2e --scenario %s --database %s --log --scenario_location %s" % (
        scenario, db_path, scenario_location)

    logger.info(cmd)

    # - "check_output" needs zero output

    completed_process = subprocess.run(cmd, shell=True)
    logger.info(completed_process.returncode)

    conn = get_database(db_path)
    table = get_table_dataframe(conn, "scenarios")
    run_id = table[table.scenario_name == scenario]['run_status_id'].squeeze()
    status = True
    if run_id != 2:
        status = False
    return status


def create_command(subscenario,
                   subscenario_id,
                   project,
                   csv_location,
                   db_path,
                   gridpath_rep,
                   delete=True):
    script = os.path.join(gridpath_rep, "db",
                          "utilities", "port_csvs_to_db.py")
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

    logger.info(cmd)
    p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate("y".encode())
    logger.info(stdout.decode())
    e = stderr.decode().strip()
    if e:
        logger.error(e)

# import logging
# logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def override_dataframe(dataframe1, dataframe2, index_cols):
    """override data from dataframe2 in dataframe1 using
    index_cols as a key to compare. 
    """
    dx = dataframe1.to_dict(orient="records")
    dy = dataframe2.to_dict(orient="records")
    ddx = {tuple(r[c] for c in index_cols): r for r in dx}
    ddy = {tuple(r[c] for c in index_cols): r for r in dy}
    commonkeys = ddx.keys() & ddy.keys()
    extrakeys = ddy.keys() - ddx.keys()
    [ddx[k].update(ddy[k]) for k in commonkeys]
    ddx.update({k: ddy[k] for k in extrakeys})
    return pd.DataFrame(ddx.values())


def merge_in_csv(results,
                 csvpath, cols, on):
    # logging.info("Entry")
    csvpartial = results.loc[:, cols]
    allcsv = pd.read_csv(csvpath)
    df = override_dataframe(allcsv, csvpartial, [on])
    logger.info(f"Merging results to {csvpath}")
    df.to_csv(csvpath, index=False)
    # logging.info("Exit")
