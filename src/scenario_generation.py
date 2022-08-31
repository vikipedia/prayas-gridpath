import click
import os
import csv
import re
import functools
import pandas as pd
import numpy as np
import datetime
import common
import shutil


BALANCING_TYPE = {"year": 1,
                  "month": 12,
                  "day": 365}

GRAN = {1: "yearly",
        12: "monthly",
        365: "daily",
        365*4: "6hr",
        365*24: "hourly",
        365*96: "15min"}


class InvalidSubscenario(Exception):
    pass


class CSVLocation(object):
    """Documentation for CSVLocation
    class which acts as wrapper over folder, csv_location
    """

    def __init__(self, csv_location):
        """
        param csv_location - a path where all csvs are strored in gridpath format
        """
        self.csv_location = csv_location

    def get_scenarios_csv(self):
        return os.path.join(self.csv_location, "scenarios.csv")

    def get_csv_data_master(self):
        return os.path.join(self.csv_location, "csv_data_master.csv")


class Scenario(CSVLocation):
    """Documentation for Scenario
    it stores all subscenarios in given scenario
    """

    def __init__(self, csv_location, scenario_name):
        super().__init__(csv_location)
        scenarios_csv = self.get_scenarios_csv()
        self.scenario_name = scenario_name

        self.subscenarios = {}
        with open(scenarios_csv) as f:
            csvf = csv.DictReader(f)
            for row in csvf:
                subscenario_name = row['optional_feature_or_subscenarios']
                subscenario_id = row[scenario_name]
                if subscenario_id.strip() != "":
                    self.subscenarios[subscenario_name] = int(subscenario_id)
                    setattr(self, subscenario_name, int(subscenario_id))

    def get_subscenarios(self):
        return [Subscenario(name, v, self.csv_location) for name, v in self.subscenarios.items()]

    def get_subscenario(self, name):
        if name in self.subscenarios:
            return Subscenario(name, self.subscenarios[name], self.csv_location)
        else:
            raise KeyError(
                f"Scenario {self.scenario_name} does not have subscenario {name}")

    def __str__(self):
        return f"Senario<{self.scenario_name}>"

    def __repr__(self):
        return str(self)


def test_scenario_class():
    rpo30 = Scenario(
        "/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh", "rpo30")

    assert rpo30.scenario_name == "rpo30"
    assert rpo30.csv_location == "/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh"
    assert rpo30.temporal_scenario_id == 5
    assert rpo30.load_zone_scenario_id == 1
    assert rpo30.load_scenario_id == 1
    assert rpo30.project_portfolio_scenario_id == 1
    assert rpo30.project_operational_chars_scenario_id == 3
    assert rpo30.project_availability_scenario_id == 3
    assert rpo30.project_load_zone_scenario_id == 1
    assert rpo30.project_specified_capacity_scenario_id == 1
    assert rpo30.project_specified_fixed_cost_scenario_id == 1
    assert rpo30.solver_options_id == 1
    assert rpo30.temporal_scenario_id == 5


class Subscenario(CSVLocation):
    """Documentation for Scenario

    """

    def __init__(self, name, id_, csv_location):
        super().__init__(csv_location)
        self.name = name
        self.id_ = id_
        try:
            self.__find_files()
        except Exception as e:
            print(e)
            print("Creating empty Subscenario")

    @functools.lru_cache(maxsize=None)
    def __getattr__(self, name):
        files = [os.path.basename(f) for f in self.files]
        attrs = [".".join(f.split(".")[:-1]) for f in files]
        if name in attrs:
            file = [f for f in self.get_files(
            ) if f.endswith(f"{name}.csv")][0]
            return pd.read_csv(file)
        elif name == "data":
            file = [f for f in self.get_files()][0]
            return pd.read_csv(file)

    def get_name(self):
        return self.name

    def get_id(self, arg):
        return self._id

    def __str__(self):
        return f"{self.name}<{self.id_}>"

    def __repr__(self):
        return str(self)

    def get_folder(self):
        master = self.get_csv_data_master()
        return self.get_subscenario_folder(self.get_csv_data_master(),
                                           self.name,
                                           self.csv_location)

    @staticmethod
    def get_subscenario_folder(master, name, csv_location):
        with open(master) as f:
            csvf = csv.DictReader(f)
            folder = [row['path']
                      for row in csvf if row['subscenario'] == name][0]
            return os.path.join(csv_location, folder)

    def __find_files(self):
        master = self.get_csv_data_master()
        p = re.compile(f"{self.id_}_.*")
        p_ = re.compile(f".*-{self.id_}.*.csv")
        with open(master) as f:
            csvf = csv.DictReader(f)
            rows = [row for row in csvf if row['subscenario'] == self.name]
            sub_types = [row['subscenario_type'] for row in rows]
            project_input = [row['project_input'] for row in rows]
            filenames = [r['filename'] for r in rows if r['filename']]
            files = []
            if "dir_subsc_only" in sub_types:
                self.sub_type = "dir_subsc_only"
                subfolders = [f for f in os.listdir(
                    self.get_folder()) if p.match(f)]
                path = os.path.join(self.get_folder(), subfolders[0])
                files = [os.path.join(path, f) for f in filenames]
                self.files = files
            elif "simple" in sub_types:
                self.sub_type = "simple"
                path = self.get_folder()
                if '1' in project_input:
                    p = p_

                files = [os.path.join(path, f) for f in os.listdir(
                    self.get_folder()) if p.match(f)]
            else:
                raise InvalidSubscenario(f"Invalid subscenario {self.name}")
            self.files = files

    def get_files(self):
        return self.files

    def writedata(self, subfolder, **kwargs):  # FIXME
        def checkexisting(folder):

            return os.path.exists(folder) and self.files

        def writefile(**data):
            for name, value in data.items():
                path = os.path.join(scid_folder, name+".csv")
                if os.path.exists(path):
                    print(f"File {name}.csv exists, skipping!")
                else:
                    value.to_csv(path, index=False,
                                 date_format='%d-%m-%Y %H:%M')

        if subfolder:
            folder = self.get_folder()
            scid_folder = os.path.join(folder, subfolder)
        else:
            scid_folder = self.get_folder()

        try:
            os.makedirs(scid_folder)
        except Exception as e:
            print(e)
            print("Not creating folder, probably it exists")

        for k, v in kwargs.items():
            writefile(**{k: v})
        self.__find_files()

    def mergedata(self, merge_on: str, **kwargs):
        """
        only for those subscenarios for which data merging is possbible.
        for example for exogenous_availability_scenario_id, data of
        different temporal settings can be stored in same file.
        """
        scid_folder = self.get_folder()
        try:
            os.makedirs(scid_folder)
        except Exception as e:
            print(e)
            print("Not creating folder, probably it exists")

        for name, value in kwargs.items():
            path = os.path.join(scid_folder, name + ".csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df = df.merge(value, on=merge_on)
            else:
                df = value

            df.to_csv(path, index=False)


def test_temporal_scenario_id_class():
    tmp5 = Subscenario(
        5, "/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh")
    s = tmp5.structure
    assert len(s.timepoint[s.spinup_or_lookahead == 0]) == 365*96


def remove_subcenario(s):
    for f in s.get_files():
        os.remove(f)
    folder = os.path.dirname(s.get_files()[0])
    os.removedirs(folder)


def test_create_temporal_subscenario():

    t1 = Subscenario('temporal_scenario_id', 1, CSV_LOCATION)

    t75 = create_temporal_subscenario(t1, "month", 'daily', 75)
    assert len(t75.structure) == 365
    assert len(t75.horizon_timepoints) == 12
    t76 = create_temporal_subscenario(t1, "year", 'daily', 76)
    assert len(t76.structure) == 365
    assert len(t76.horizon_timepoints) == 1
    remove_subcenario(t75)
    remove_subcenario(t76)


def create_temporal_subscenario(base: Subscenario,
                                balancing_type_horizon: str,
                                granularity: str,
                                id_: int,
                                map_file):

    structure, horizon_params, horizon_timepoints, period_params = create_temporal_subscenario_data(
        base, balancing_type_horizon, granularity, id_, map_file)
    steps = len(structure['subproblem_id'].unique())
    granularity = len(structure[structure.spinup_or_lookahead == 0])
    d = GRAN

    gran = d[granularity]
    subfolder = f"{id_}_{steps}steps_{gran}_timepoints"
    tscid = Subscenario(name='temporal_scenario_id',
                        id_=id_,
                        csv_location=base.csv_location)
    tscid.writedata(subfolder,
                    description=pd.DataFrame({f"{steps} steps": [],
                                              f"{gran} timepoints": []}),
                    structure=structure,
                    horizon_params=horizon_params,
                    horizon_timepoints=horizon_timepoints,
                    period_params=period_params)
    return tscid


def write_endo_project_file(filename, data, headers):
    if os.path.exists(filename):
        print(f"File {filename} exists, skipping overwrite.")
    with open(filename, "w") as f:
        csvf = csv.DictWriter(f, headers)
        csvf.writerow(data)


def get_project_filename(project, subscenario_id, subscenario_name):
    return f"{project}-{subscenario_id}-{subscenario_name}.csv"


def create_availability_subscenario(csv_location: str,
                                    availability: str,
                                    endogenous: str,
                                    description: str,
                                    id_: int
                                    ):
    """
    csv_location -> csv_location
    """
    pascid = Subscenario(name='project_availability_scenario_id',
                         id_=id_,
                         csv_location=csv_location
                         )
    data = pd.read_csv(availability)
    availdata = data[data.endogenous_availability_scenario_id.notnull()]
    if endogenous:
        agg_data = pd.read_csv(endogenous)
        for project, endoscid_ in availdata[
                ['project', 'endogenous_availability_scenario_id']].drop_duplicates().values:
            endoscid = Subscenario('endogenous_availability_scenario_id',
                                   endoscid_,
                                   csv_location)

            print(endoscid_, "**"*10)
            df = agg_data[(agg_data.subscenario_id == endoscid_)
                          & (agg_data.project == project)]
            cols = list(df.columns)
            cols_ = ['project', 'subscenario_id']  # , 'subscenario_name']
            for c in cols_:
                cols.remove(c)
            projectdata = df[cols]
            filename = common.get_subscenario_csvpath(project,
                                                      'endogenous_availability_scenario_id',
                                                      int(float(endoscid_)),
                                                      endoscid.csv_location,
                                                      "endo")
            f = os.path.basename(filename.split(".")[0])

            endoscid.writedata(None, **{f: projectdata})

    pascid.writedata(None, **{f"{id_}_availability_{description}": data})
    return pascid


def get_subset(temporal):
    s = temporal.structure
    s = s[s.spinup_or_lookahead == 0]
    s = s[['timepoint', 'timestamp']]
    s['timestamp'] = pd.to_datetime(s.timestamp, format='%d-%m-%Y %H:%M')
    s.sort_values('timestamp', inplace=True)
    s.set_index('timestamp', inplace=True)
    return s


def create_horizon_or_timepoint(base, size):
    pad = len(str(size+1))
    if size == 1:
        return np.array([int(base)])
    return np.array([int(str(base) + str(i).zfill(pad)) for i in range(1, size+1)])


def get_horizon_col_name(timepoint_map, horizon):
    col = [
        c for c in timepoint_map.columns if 'horizon' in c and c.endswith(horizon)][0]
    return col


def create_timepoints(horizon, n):
    return create_horizon_or_timepoint(horizon, n)


def create_horizon_params(base,
                          balancing_type_horizon,
                          structure,
                          ):

    grp = structure.groupby("subproblem_id", sort=False)
    period = base.period_params['period'].iloc[0]
    hname = 'horizon_'+balancing_type_horizon
    df = grp[[hname]].first().reset_index()
    n = len(df)
    subproblem_id = np.array(range(1, n+1))
    balancing_type_horizon_ = np.array([balancing_type_horizon]*n)
    boundary = np.array(["linear"]*n)
    df['balancing_type_horizon'] = balancing_type_horizon
    df['boundary'] = boundary
    df.rename(columns={hname: 'horizon'}, inplace=True)
    return df[["subproblem_id",
               "balancing_type_horizon",
               "horizon",
               "boundary"]]


def create_period_params(base):
    return base.period_params.copy()


def get_delta(granularity):
    if granularity == "15min":
        return datetime.timedelta(minutes=15)
    elif granularity == "hourly":
        return datetime.timedelta(hours=1)
    elif granularity == "6hour":
        return datetime.timedelta(hours=6)
    elif granularity == "daily":
        return datetime.timedelta(days=1)


def get_subproblem_id_index(balancing_type_horizon, s1):
    """
    s1 is dataframe with columns m, d, period
    """

    if balancing_type_horizon == 'month':
        index = pd.Index(data=s1['m'].unique(), name='m')
    elif balancing_type_horizon == 'year':
        index = pd.Index(data=s1['period'].unique(), name='period')
    elif balancing_type_horizon == 'day':
        index = pd.MultiIndex.from_arrays(
            [s1['d'], s1['m']], names=['d', 'm']).unique()
    else:
        raise ValueError(
            f"Incompatible balancing_type_horizon {balancing_type_horizon}")

    return index


def get_subproblem_id(index):
    subproblem_id = pd.Series(data=range(
        1, len(index)+1), index=index, name='subproblem_id')
    return subproblem_id


def get_groupby_cols(granularity):
    common = ['stage_id', 'period']

    if granularity == "daily":
        grpbycols = ['d', 'm']
    elif granularity == "monthly":
        grpbycols = ['m']
    elif granularity == "yearly":
        grpbycols = []
    elif granularity == "hourly":
        grpbycols = ['d', 'm', 'H']
    elif granularity == "6hr":
        grpbycols = ['d', 'm', '6HR']
    elif granularity == "15min":
        grpbycols = ['d', 'm', 'H', 'M']
    else:
        raise ValueError(f"Incompatible granularity {granularity}")

    grpbycols.extend(common)
    return grpbycols


def split_timestamp(s):
    ts = s.timestamp.str.split("-", expand=True)
    ts.columns = ['d', 'm', 'ys']
    ts1 = ts.ys.str.split(expand=True)
    del ts['ys']
    ts['y'] = ts1[0]
    ts2 = ts1[1].str.split(":", expand=True)
    ts['H'] = ts2[0]
    ts['M'] = ts2[1]
    ts['6HR'] = pd.to_numeric(ts['H'])//6
    return ts


def get_groupsby_structure(base: Subscenario, granularity):
    structure = base.structure
    s = structure[structure.spinup_or_lookahead == 0]
    ts = split_timestamp(s)
    return s[['timepoint', 'timepoint_weight', 'timestamp']].join(ts)


def subset_data(data, temporal: Subscenario):
    s = temporal.structure
    s = s[s.spinup_or_lookahead == 0]
    t = s[['timepoint']]
    return t.merge(data, on='timepoint')[data.columns]


def collapse(data,
             columns,
             basetemporal: Subscenario,
             granularity,
             subtemporal: Subscenario,
             weighted=False,
             operation='sum'):
    ts = get_groupsby_structure(basetemporal, granularity)
    col_names = [c for c in data.columns]
    grpbycols = get_groupby_cols(granularity)
    grpbycols.remove('period')
    grpbycols.remove('stage_id')

    data = data.merge(ts, on='timepoint')
    if weighted:
        for c in columns:
            data[c] = data[c]*data.timepoint_weight
        grp = data.groupby(grpbycols, sort=False)
        opgrp = grp[columns]
        weight = grp['timepoint_weight']
        r = opgrp.sum()
        for c in columns:
            r[c] = r[c]/weight.sum()
    else:
        grp = data.groupby(grpbycols, sort=False)
        opgrp = grp[columns]
        op = getattr(opgrp, operation)
        r = op()

    timestamp = grp[[
        'timestamp']+[c for c in col_names if c not in columns and c != 'timepoint']].first()
    r = r.join(timestamp)

    s = subtemporal.structure
    s = s[s.spinup_or_lookahead == 0][['timepoint', 'timestamp']]
    final_r = r.merge(s, on='timestamp')
    final_r = final_r.sort_values(by="timepoint").reset_index()
    return final_r[col_names]


def create_structure(base: Subscenario,
                     balancing_type_horizon: str,
                     granularity: str,
                     map_file):

    ns = [key for key, value in GRAN.items() if value == granularity]
    if not ns:
        raise ValueError(
            "Invalid granularity specified. valid granularity values are {}".format(GRAN.values()))

    structure = base.structure
    s = structure[structure.spinup_or_lookahead == 0]
    ts = split_timestamp(s)
    s1 = s.join(ts)

    grpbycols = get_groupby_cols(granularity)
    fcols = ['timepoint_weight', 'previous_stage_timepoint_map',
             'spinup_or_lookahead', 'linked_timepoint', 'month', 'hour_of_day', 'timestamp']
    scols = ['number_of_hours_in_timepoint']

    grp = s1.groupby(grpbycols, sort=False)

    firstcols = grp[fcols].first()
    sumcols = grp[scols].sum()
    # sumcols = sumcols.astype(int)

    index = get_subproblem_id_index(balancing_type_horizon, s1)
    subproblem_id = get_subproblem_id(index)
    s_ = firstcols.join(sumcols).join(subproblem_id)
    s_ = create_timepoint_horizon_cols(s_,
                                       balancing_type_horizon,
                                       map_file,
                                       granularity)
    s_['linked_timepoint'].iloc[:] = np.nan
    colnames = ['subproblem_id', 'stage_id', 'timepoint', 'period', 'number_of_hours_in_timepoint', 'timepoint_weight',
                'previous_stage_timepoint_map', 'spinup_or_lookahead', 'linked_timepoint', 'month', 'hour_of_day', 'timestamp', f"horizon_{balancing_type_horizon}"]
    return s_[colnames]


def create_timepoint_horizon_cols(s_, horizon, map_file, granularity):
    """
    s_ is dataframe containing columns with name timestamp in '%d-%m-%Y %H:%M' format
    make use
    of these columns to make timepoint
    """
    s_['timestamp_'] = pd.to_datetime(s_.timestamp, format='%d-%m-%Y %H:%M')

    s_.sort_values('timestamp_', inplace=True)
    s_.reset_index(inplace=True)

    timepoint_map = pd.read_excel(map_file,
                                  sheet_name="map",
                                  skiprows=2,
                                  engine='openpyxl')

    timepoint_cols = [
        c for c in timepoint_map.columns if "timepoint" in c and granularity in c]
    if not timepoint_cols:
        raise Exception(
            f"timepoint column for {granularity} not found in {map_file}")
    timepoint_col = timepoint_cols[0]
    horizon_ = get_horizon_col_name(timepoint_map, horizon)
    tf = timepoint_map[[timepoint_col, horizon_]]
    tf = tf.groupby(timepoint_col, sort=False).first().reset_index()
    tf = tf.rename(columns=dict(zip([timepoint_col, horizon_],
                                    ['timepoint', f'horizon_{horizon}'])))
    s_['timepoint'] = tf['timepoint'].iloc[:]
    s_[f'horizon_{horizon}'] = tf[f'horizon_{horizon}'].iloc[:]

    return s_


def create_horizon_timepoints(structure, balancing_type_horizon):
    grp = structure.groupby('subproblem_id', sort=False)
    cols = ['subproblem_id', 'stage_id']
    hname = "horizon_" + balancing_type_horizon
    df = grp[['stage_id', hname, 'timepoint']].first().reset_index()
    balancing_type_horizon = np.array([balancing_type_horizon]*len(df))
    tmp_end = grp['timepoint'].last().reset_index(drop=True)
    df['balancing_type_horizon'] = balancing_type_horizon
    df['tmp_end'] = tmp_end
    df.rename(columns={'timepoint': 'tmp_start',
                       hname: 'horizon'},
              inplace=True)

    return df[['subproblem_id',
               'stage_id',
               'balancing_type_horizon',
               'horizon',
               'tmp_start',
               'tmp_end']]


def create_temporal_subscenario_data(base: Subscenario,
                                     balancing_type_horizon: str,
                                     granularity: str,
                                     id_: int,
                                     map_file):
    hparams = base.horizon_params
    base_balancing_type = hparams['balancing_type_horizon'].unique()[0]
    bt = BALANCING_TYPE
    if balancing_type_horizon not in bt:
        raise ValueError("Wrong input for steps, possible steps are")
    if bt[balancing_type_horizon] > bt[base_balancing_type]:  # WHY?
        raise ValueError(
            f"New subscenario Can not have more steps than base subscenario")

    structure = create_structure(
        base, balancing_type_horizon, granularity, map_file)
    period_params = create_period_params(base)
    horizon_params = create_horizon_params(
        base, balancing_type_horizon, structure)
    horizon_timepoints = create_horizon_timepoints(
        structure, balancing_type_horizon)
    return structure, horizon_params, horizon_timepoints, period_params


CSV_LOCATION = "/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh"


def test_subscenario_class():
    rpo30 = Scenario(CSV_LOCATION, "rpo30")
    assert rpo30.get_subscenario('temporal_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/period_params.csv', '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/horizon_params.csv',
                                                                         '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/structure.csv', '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/horizon_timepoints.csv']

    assert rpo30.get_subscenario('load_zone_scenario_id').get_files() == [
        '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/system/load_zones/1_load_zone_msedcl_voll20.csv']
    assert rpo30.get_subscenario('load_scenario_id').get_files() == [
        '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/system/load/1_load_msedcl_5pc_all.csv', '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/system/load/1_load_msedcl_5pc_all.txt']
    assert rpo30.get_subscenario('project_portfolio_scenario_id').get_files() == [
        '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/portfolios/1_portfolio_msedcl_2030_rpo30.csv']
    assert rpo30.get_subscenario('project_operational_chars_scenario_id').get_files() == [
        '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/opchar/3_opchar_msedcl_rpo30_daily.csv']
    assert rpo30.get_subscenario('project_availability_scenario_id').get_files() == [
        '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/availability/3_availability_rpo30.csv']
    assert rpo30.get_subscenario('project_load_zone_scenario_id').get_files() == [
        '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/load_zones/1_project_load_zones_msedcl.csv']
    assert rpo30.get_subscenario('project_specified_capacity_scenario_id').get_files() == [
        '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/specified/capacity/1_specified_msedcl_2030_rpo30.csv']
    assert rpo30.get_subscenario('project_specified_fixed_cost_scenario_id').get_files() == [
        '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/specified/fixed_cost/1_fc_msedcl_2030_rpo30.csv']
    assert rpo30.get_subscenario('solver_options_id').get_files() == [
        '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/solver/1_cplex_mipgap_0.5.txt', '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/solver/1_cplex_mipgap_0.5.csv']
    assert len(rpo30.get_subscenario(
        'temporal_scenario_id').structure) == 365*96


def get_subscenarios(scenario, csv_location):
    scenarioscsv = os.path.join(csv_location, "scenarios.csv")
    with open(scenarioscsv) as f:
        csvf = csv.DictReader(f)
        return {r['optional_feature_or_subscenarios']: r[scenario] for r in csvf if r[scenario]}


def get_subscenario_paths(csv_location):
    def fullpath(p):
        return os.path.join(csv_location, p)

    csv_data_master = os.path.join(csv_location, "csv_data_master.csv")
    with open(csv_data_master) as f:
        csvf = csv.DictReader(f)
        return {r['subscenario']: fullpath(r['path']) for r in csvf if r['path']}


def collapse_variable_profile(vpop_scid,
                              project,
                              basetemporal: Subscenario,
                              subtemporal: Subscenario,
                              granularity):

    f = common.get_subscenario_csvpath(project,
                                       'variable_generator_profile_scenario_id',
                                       vpop_scid,
                                       basetemporal.csv_location)
    df = pd.read_csv(f)
    basedf = subset_data(df, basetemporal)
    collapsed_df = collapse(basedf,
                            ['cap_factor'],
                            basetemporal,
                            granularity,
                            subtemporal,
                            weighted='True')
    return collapsed_df


def update_variable_profile_opchar_scenario(vpop_scid,
                                            project,
                                            basetemporal: Subscenario,
                                            subtemporal: Subscenario,
                                            granularity,
                                            description,
                                            update=False):

    f = common.get_subscenario_csvpath(project,
                                       'variable_generator_profile_scenario_id',
                                       vpop_scid,
                                       basetemporal.csv_location,
                                       description=description)
    df = pd.read_csv(f)
    basedf = subset_data(df, basetemporal)
    if len(subset_data(df, subtemporal)) == 0 or update:
        collapsed_df = collapse_variable_profile(vpop_scid,
                                                 project,
                                                 basetemporal,
                                                 subtemporal,
                                                 granularity)
        common.merge_in_csv(collapsed_df,
                            f,
                            ['stage_id', 'timepoint', 'cap_factor'],
                            on='timepoint')


def create_project_operational_chars_subscenario(opchars_base: Subscenario,
                                                 id_: int,
                                                 balancing_type_project: str,
                                                 basetemporal,
                                                 subtemporal,
                                                 granularity,
                                                 desc: str,
                                                 update,
                                                 hydroopchars_dir=None):
    opcharsscid = Subscenario(opchars_base.name, id_,
                              opchars_base.csv_location)
    df = opchars_base.data
    ot = df.operational_type.str.replace("gen_commit_bin", "gen_commit_cap")
    df['operational_type'] = ot
    df.balancing_type_project = balancing_type_project
    cols = ['fuel', 'heat_rate_curves_scenario_id', 'variable_om_curves_scenario_id', 'startup_chars_scenario_id', 'startup_cost_per_mw', 'shutdown_cost_per_mw', 'startup_fuel_mmbtu_per_mw', 'startup_plus_ramp_up_rate', 'shutdown_plus_ramp_down_rate', 'ramp_up_when_on_rate', 'ramp_down_when_on_rate', 'ramp_up_violation_penalty', 'ramp_down_violation_penalty', 'min_up_time_hours', 'min_up_time_violation_penalty', 'min_down_time_hours', 'min_down_time_violation_penalty', 'discharging_capacity_multiplier',
            'minimum_duration_hours', 'maximum_duration_hours', 'aux_consumption_frac_capacity', 'last_commitment_stage', 'curtailment_cost_per_pwh', 'lf_reserves_up_derate', 'lf_reserves_down_derate', 'regulation_up_derate', 'regulation_down_derate', 'frequency_response_derate', 'spinning_reserves_derate', 'lf_reserves_up_ramp_rate', 'lf_reserves_down_ramp_rate', 'regulation_up_ramp_rate', 'regulation_down_ramp_rate', 'frequency_response_ramp_rate', 'spinning_reserves_ramp_rate']
    df[cols] = np.nan
    filename = f"{id_}_{desc}"
    opcharsscid.writedata(None, **{filename: df})

    projects = df.project[df.operational_type == "gen_var"]
    vpscid = df.variable_generator_profile_scenario_id[df.operational_type == 'gen_var']
    for vpop_scid, project in zip(vpscid, projects):
        update_variable_profile_opchar_scenario(int(vpop_scid),
                                                project,
                                                basetemporal,
                                                subtemporal,
                                                granularity,
                                                desc,
                                                update)

    update_hydro_op_chars(df,
                          subtemporal,
                          balancing_type_project,
                          hydroopchars_dir,
                          desc)

    return opcharsscid


def update_hydro_op_chars(opcharsdf,
                          subtemporal,
                          balancing_type_project,
                          hydro_dir,
                          desc):
    df = opcharsdf
    gen_hydro = df.operational_type == "gen_hydro"
    hprojects = df.project[gen_hydro]
    hopcscid = df.hydro_operational_chars_scenario_id[gen_hydro].astype(int)

    cols = ['balancing_type_project',
            'horizon',
            'period',
            'average_power_fraction',
            'min_power_fraction',
            'max_power_fraction']

    period = subtemporal.period_params.period.iloc[0]
    for hopc_scid, project in zip(hopcscid, hprojects):
        if hydro_dir:
            csvs = [f for f in os.listdir(
                hydro_dir) if f.startswith(f"{project}-{hopc_scid}")]
            if csvs:
                filename = os.path.join(hydro_dir, csvs[0])
                df = pd.read_csv(filename)
                hydro_data = df[df.balancing_type_project ==
                                balancing_type_project]
            else:
                raise Exception(f"{hydro_dir} has no csv files!")

        elif balancing_type_project == 'year':
            hydro_data = pd.DataFrame(
                [dict(zip(cols, ['year', period, period, 1.0, 0.0, 1.0]))])
        else:
            raise Exception("hydro_dir option can not be empty.")

        write_hydro_opchars_year_data(hydro_data,
                                      project,
                                      int(hopc_scid),
                                      cols,
                                      subtemporal.csv_location,
                                      desc)


def write_hydro_opchars_year_data(hydro_data,
                                  project,
                                  hopc_scid,
                                  cols,
                                  csv_location,
                                  description):
    dest = common.get_subscenario_csvpath(project,
                                          'hydro_operational_chars_scenario_id',
                                          hopc_scid,
                                          csv_location,
                                          description=description)
    print(f"Merging data to {dest}")
    common.merge_in_csv(hydro_data,
                        dest,
                        cols=cols,
                        on='horizon')


def next_available_subscenario_id(subscenario: Subscenario):
    folder = subscenario.get_folder()
    p = re.compile(r'(?P<id>\d{1,2})_.*')
    files = [f for f in os.listdir(folder) if p.match(f)]
    return max([int(p.match(f).groupdict()['id']) for f in files])+1


def get_opchars_file_desc(opchars, currentsteps):
    p = re.compile('{id}_(?P<desc>.*)_.+.csv'.format(id=opchars.id_))
    m = p.match(os.path.basename(opchars.files[0]))
    return "_".join([m.groupdict()['desc'], currentsteps+"ly"])


def update_load_scenario_id(load_scid,
                            basetemporal,
                            subtemporal,
                            granularity,
                            update):
    lscid = Subscenario('load_scenario_id',
                        load_scid, basetemporal.csv_location)

    d = lscid.data
    load_zones = d.load_zone.unique()
    for zone in load_zones:
        if len(subset_data(d, subtemporal)) == 0 or update:
            d = subset_data(d[d.load_zone == zone], basetemporal)
            cd = collapse(d,
                          ['load_mw'],
                          basetemporal,
                          granularity,
                          subtemporal,
                          weighted=True,
                          operation='mean')

            filename = lscid.get_files()[0]
            common.merge_in_csv(cd,
                                filename,
                                list(d.columns),
                                on='timepoint')


def create_new_scenario(base_scenario,
                        output_scenario,
                        csv_location,
                        steps,
                        granularity,
                        availability,
                        endogenous,
                        hydroopchars_dir,
                        db_path,
                        gridpath_rep,
                        map_file,
                        update=False):
    """
    create a new scenario using a base scenario (assumed daily).
    In the new scenario temporal definations are 
    different. accordingly all time dependent files 
    should change.
    """
    base = Scenario(csv_location, base_scenario)
    temporal_base = base.get_subscenario('temporal_scenario_id')
    t_id = next_available_subscenario_id(temporal_base)
    temporal = create_temporal_subscenario(
        temporal_base, steps, granularity, t_id, map_file)
    project_availability_base = base.get_subscenario(
        'project_availability_scenario_id')
    pa_id = next_available_subscenario_id(project_availability_base)
    project_availability = create_availability_subscenario(csv_location,
                                                           availability,
                                                           endogenous,
                                                           description=f"{output_scenario}",
                                                           id_=pa_id)
    popchars_base = base.get_subscenario(
        'project_operational_chars_scenario_id')
    popchars_id = next_available_subscenario_id(popchars_base)
    desc = get_opchars_file_desc(popchars_base, steps)
    popchars = create_project_operational_chars_subscenario(popchars_base,
                                                            popchars_id,
                                                            steps,
                                                            temporal_base,
                                                            temporal,
                                                            granularity,
                                                            desc,
                                                            update,
                                                            hydroopchars_dir
                                                            )
    update_load_scenario_id(base.load_scenario_id,
                            temporal_base,
                            temporal,
                            granularity,
                            update)

    scenarios = pd.read_csv(base.get_scenarios_csv())
    scenarios.set_index('optional_feature_or_subscenarios', inplace=True)
    c = scenarios[base_scenario].copy()
    c['temporal_scenario_id'] = temporal.id_
    c['project_operational_chars_scenario_id'] = popchars.id_
    c['project_availability_scenario_id'] = project_availability.id_
    scenarios[output_scenario] = c
    scenarios.reset_index(inplace=True)
    scenarios.to_csv(base.get_scenarios_csv(),
                     float_format="%.0f", index=False)
    update_to_database(temporal,
                       project_availability,
                       popchars,
                       db_path, gridpath_rep)
    common.update_scenario_via_gridpath(output_scenario,
                                        csv_location,
                                        db_path,
                                        gridpath_rep)


def update_to_database(temporal, project_availability, opchar_scid,
                       db_path,
                       gridpath_rep):
    common.update_subscenario_via_gridpath("temporal_scenario_id",
                                           temporal.id_,
                                           None,
                                           temporal.csv_location,
                                           db_path,
                                           gridpath_rep
                                           )
    common.update_subscenario_via_gridpath("project_availability_scenario_id",
                                           project_availability.id_,
                                           None,
                                           project_availability.csv_location,
                                           db_path,
                                           gridpath_rep
                                           )
    update_project_based_subscenarios_to_db(project_availability,
                                            'endogenous_availability_scenario_id',
                                            db_path,
                                            gridpath_rep)

    common.update_subscenario_via_gridpath("project_operational_chars_scenario_id",
                                           opchar_scid.id_,
                                           None,
                                           opchar_scid.csv_location,
                                           db_path,
                                           gridpath_rep)

    update_project_based_subscenarios_to_db(opchar_scid,
                                            'hydro_operational_chars_scenario_id',
                                            db_path,
                                            gridpath_rep
                                            )


def update_project_based_subscenarios_to_db(subscenario,
                                            sub_subscenario,
                                            db_path,
                                            gridpath_rep):
    availdf = subscenario.data
    endo = availdf[sub_subscenario]
    project = availdf.project[endo.notnull()]
    endo = endo[endo.notnull()]
    for p, e in zip(project, endo):
        common.update_subscenario_via_gridpath(sub_subscenario,
                                               int(e),
                                               p,
                                               subscenario.csv_location,
                                               db_path,
                                               gridpath_rep)


@click.command()
@click.option("-b", "--base_scenario", default="rpo30", help="Base scenario from which other scenario will be generated.")
@click.option("-o", "--output_scenario", default="rpo30_monthly", help="Name of scenario to generate")
@click.option("-s", "--steps", default="month", help="steps as one of year, month, day")
@click.option("-t", "--granularity", default="daily", help="granularity option as one of 15min, hourly, daily, monthly, yearly")
@click.option("-a", "--availability", default=None, help="path to availability data")
@click.option("-e", "--endo", default=None, help="Path to file which contains endogenous availability data in single file")
@click.option("-h", "--hydro_dir", default=None, help="Path to directory which contains data for hydro_operational_chars_scenario_id")
@click.option("-c", "--csv_location", default="csvs_mh", help="Path to folder where csvs are")
@click.option("-d", "--database", default="../mh.db", help="Path to database")
@click.option("-g", "--gridpath_rep", default="../", help="Path of gridpath source repository")
@click.option("-u", "--update/--no-update", default=False, help="Update new data in csv files even if it exists.")
@click.option("-m", "--map_file", default='../../../db/timepoint_map_2040.xlsx', help="Base scenario from which other scenario will be generated.")
def main(base_scenario,
         output_scenario,
         steps,
         granularity,
         availability,
         endo,
         hydro_dir,
         csv_location,
         database,
         gridpath_rep,
         create_new_scenario,
         map_file,
         update):
    create_new_scenario(base_scenario,
                        output_scenario,
                        csv_location,
                        steps,
                        granularity,
                        availability,
                        endo,
                        hydro_dir,
                        database,
                        gridpath_rep,
                        map_file,
                        update
                        )


if __name__ == "__main__":
    main()
