import click
import os
import csv
import re
import functools
import pandas as pd
import numpy as np
import datetime

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
                if subscenario_id.strip()!="":
                    self.subscenarios[subscenario_name] = int(subscenario_id)
                    setattr(self, subscenario_name, int(subscenario_id))

    def get_subscenarios(self):
        return [Subscenario(name, v, self.csv_location) for name, v in self.subscenarios.items()]

    def get_subscenario(self, name):
        if name in self.subscenarios:
            return Subscenario(name, self.subscenarios[name], self.csv_location)
        else:
            raise KeyError(f"Scenario {self.scenario_name} does not have subscenario {name}")
    
    def __str__(self):
        return f"Senario<{self.scenario_name}>"

    def __repr__(self):
        return str(self)


def test_scenario_class():
    rpo30 = Scenario("/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh", "rpo30")
    
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
            file = [f for f in self.get_files() if f.endswith(f"{name}.csv")][0]
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
            folder = [row['path'] for row in csvf if row['subscenario']==name][0]
            return os.path.join(csv_location, folder)

        
    def __find_files(self):
        master = self.get_csv_data_master()
        p = re.compile(f"{self.id_}_.*")
        p_ = re.compile(f".*-{self.id_}.*.csv")
        with open(master) as f:
            csvf = csv.DictReader(f)
            rows = [row for row in csvf if row['subscenario']==self.name]
            sub_types = [row['subscenario_type'] for row in rows]
            project_input = [row['project_input'] for row in rows]
            filenames = [r['filename'] for r in rows if r['filename']]
            if "dir_subsc_only" in sub_types:
                self.sub_type = "dir_subsc_only"
                subfolders = [f for f in os.listdir(self.get_folder()) if p.match(f)]
                path = os.path.join(self.get_folder(), subfolders[0])
                files = [os.path.join(path, f) for f in filenames]
            elif "simple" in sub_types:
                self.sub_type = "simple"
                path = self.get_folder()
                if '1' in project_input:
                    p = p_
                    
                files = [os.path.join(path, f) for f in os.listdir(self.get_folder()) if p.match(f)]
            self.files = files
            
           
    def get_files(self):
        return self.files


    def writedata(self, subfolder, **kwargs):##FIXME
        def checkexisting(folder):
            
            return os.path.exists(folder) and self.files

        def writefile(**data):
            for name, value in data.items():
                path = os.path.join(scid_folder,name+".csv")
                if os.path.exists(path):
                    print(f"File {name}.csv exists, skipping!")
                else:
                    value.to_csv(path, index=False, date_format='%d-%m-%Y %H:%M')

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
            writefile(**{k:v})
        self.__find_files()

    def mergedata(self, merge_on:str, **kwargs):
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
            

BALANCING_TYPE = {"year":1,
                  "month":12,
                  "day":365}
                  #"hour":365*24,
                  #"15min":365*96}
GRAN = {1:"yearly",
        12:"monthly",
        365:"daily",
        365*24:"hourly",
        365*96:"15min"}
    

def test_temporal_scenario_id_class():
    tmp5 = Subscenario(5, "/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh")
    s = tmp5.structure
    assert len(s.timepoint[s.spinup_or_lookahead==0])==365*96

def remove_subcenario(s):
    for f in s.get_files():
        os.remove(f)
    folder = os.path.dirname(s.get_files()[0])
    os.removedirs(folder)
    
def test_create_temporal_subscenario():

    t1 = Subscenario('temporal_scenario_id', 1, CSV_LOCATION)

    t75 = create_temporal_subscenario(t1, "month", 'daily', 75)
    assert len(t75.structure)==365
    assert len(t75.horizon_timepoints)==12
    t76 = create_temporal_subscenario(t1, "year", 'daily', 76)
    assert len(t76.structure)==365
    assert len(t76.horizon_timepoints)==1
    remove_subcenario(t75)
    remove_subcenario(t76)
    
def create_temporal_subscenario(base:Subscenario,
                                balancing_type_horizon:str,
                                granularity:str,
                                id_:int):
    structure,horizon_params,horizon_timepoints, period_params = create_temporal_subscenario_data(base, balancing_type_horizon, granularity, id_)
    steps = len(structure['subproblem_id'].unique())
    granularity = len(structure[structure.spinup_or_lookahead==0])
    d = GRAN
    gran = d[granularity]
    subfolder = f"{id_}_{steps}steps_{gran}_timepoints"
    tscid = Subscenario(name='temporal_scenario_id',
                       id_=id_,
                       csv_location=base.csv_location)
    tscid.writedata(subfolder,
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
    
        
def create_availability_subscenario(csv_location:str,
                                    availability:str,
                                    endogenous:str,
                                    description:str,
                                    id_:int
                                    ):
    """
    csv_location -> csv_location
    """
    pascid = Subscenario(name = 'project_availability_scenario_id',
                         id_= id_,
                         csv_location = csv_location
                         )
    data = pd.read_csv(availability)
    agg_data = pd.read_csv(endogenous)
    availdata = data[data.endogenous_availability_scenario_id.notnull()]
    if endogenous:
        for project, endoscid_ in availdata[
                ['project','endogenous_availability_scenario_id']].drop_duplicates().values:
            endoscid = Subscenario('endogenous_availability_scenario_id',
                                   endoscid_,
                                   csv_location)

            df = agg_data[(agg_data.subscenario_id == endoscid_) & (agg_data.project==project)]
            cols = list(df.columns)
            cols_ = ['project', 'subscenario_id', 'subscenario_name']
            for c in cols_:
                cols.remove(c)
            projectdata = df[cols]
            filename = get_project_filename(**df[cols_].iloc[0].to_dict())
            try:
                endoscid.writedata(None, **{filename:projectdata})
            except Exception as e:
                print("Exception in writing data", e)
            
    pascid.writedata(None, **{f"{id_}_availability_{description}":data})
    return pascid


def get_subset(temporal):
    s = temporal.structure
    s = s[s.spinup_or_lookahead==0]
    s = s[['timepoint', 'timestamp']]
    s['timestamp'] = pd.to_datetime(s.timestamp, format='%d-%m-%Y %H:%M')
    s.sort_values('timestamp', inplace=True)
    s.set_index('timestamp', inplace=True)
    return s


def create_horizon_or_timepoint(base, size):
    pad = len(str(size+1))
    if size==1:
        return np.array([int(base)])
    return np.array([int(str(base) + str(i).zfill(pad)) for i in range(1, size+1)])


def create_horizon(period, n):
    return create_horizon_or_timepoint(period, n)


def create_timepoints(horizon, n):
    return create_horizon_or_timepoint(horizon, n)


def create_horizon_params(base,
                          balancing_type_horizon,
                          structure,
                          ):

    grp = structure.groupby("subproblem_id")
    period = base.period_params['period'].iloc[0]
    hname = 'horizon_'+balancing_type_horizon
    df = grp[[hname]].first().reset_index()
    n = len(df)
    subproblem_id = np.array(range(1,n+1))
    balancing_type_horizon_ = np.array([balancing_type_horizon]*n)
    boundary = np.array(["linear"]*n)
    df['balancing_type_horizon'] = balancing_type_horizon
    df['boundary'] = boundary
    df.rename(columns={hname:'horizon'}, inplace=True)
    return df[["subproblem_id",
               "balancing_type_horizon",
               "horizon",
               "boundary"]]

def create_period_params(base):
    return base.period_params.copy()


def get_delta(granularity):
    if granularity=="15min":
        return datetime.timedelta(minutes=15)
    elif granularity=="hourly":
        return datetime.timedelta(hours=1)
    elif granularity=="daily":
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
        index = pd.MultiIndex.from_arrays([s1['d'], s1['m']], names=['d','m']).unique()
    else:
        raise ValueError(f"Incompatible balancing_type_horizon {balancing_type_horizon}")

    return index

def get_subproblem_id(index):
    subproblem_id = pd.Series(data=range(1, len(index)+1), index=index, name='subproblem_id')
    return subproblem_id


def get_groupby_cols(granularity):
    common = ['stage_id', 'period']

    if granularity == "daily":
        grpbycols = ['d', 'm']
    elif granularity == "monthly":
        grpbycols = ['m'] 
    elif granularity == "yearly":
        grpbycols = []
    elif granularity == "hourly" :
        grpbycols = ['d','m','H']
    elif granularity == "15min" :
        grpbycols = ['d','m','H','M']
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
    return ts    

def get_groupsby_structure(base:Subscenario, granularity):
    structure = base.structure
    s = structure[structure.spinup_or_lookahead==0]
    ts = split_timestamp(s)
    return s[['timepoint', 'timepoint_weight']].join(ts)


def subset_data(data, temporal:Subscenario):
    s = temporal.structure
    s = s[s.spinup_or_lookahead==0]
    t = s[['timepoint']]
    return t.merge(data, on='timepoint')


def collapse(data,
             columns,
             basetemporal:Subscenario,
             granularity,
             subtemporal:Subscenario,
             weighted=False,
             operation='sum'):
    ts = get_groupsby_structure(basetemporal, granularity)
    grpbycols = get_groupby_cols(granularity)
    grpbycols.remove('period')
    grpbycols.remove('stage_id')
    print(ts.columns)
    data = data.merge(ts, on='timepoint')
    data.sort_values('timepoint', inplace=True)
    if weighted:
        for c in columns:
            data[c] = data.c*data.timepoint_weight
    grp = data.groupby(grpbycols)[columns]
    op = getattr(grp, operation)
    r = op()
    r.reset_index(inplace=True)
    s = subtemporal.structure
    s = s[s.spinup_or_lookahead==0]['timepoint']
    s.sort_values()
    r.join(s)
    return r

def create_structure(base:Subscenario,
                     balancing_type_horizon:str,
                     granularity:str):

    ns = [key for key, value in GRAN.items() if value == granularity]
    if not ns:
        raise ValueError("Invalid granularity specified. valid granularity values are {}".format(GRAN.values()))
    
    structure = base.structure
    s = structure[structure.spinup_or_lookahead==0]
    ts = split_timestamp(s)
    s1 = s.join(ts)
    
    grpbycols = get_groupby_cols(granularity)
    fcols = ['timepoint_weight', 'previous_stage_timepoint_map','spinup_or_lookahead','linked_timepoint', 'month', 'hour_of_day', 'timestamp']
    scols = ['number_of_hours_in_timepoint']
    
    grp = s1.groupby(grpbycols)

    firstcols = grp[fcols].first()
    sumcols = grp[scols].sum()
    #sumcols = sumcols.astype(int)

    index = get_subproblem_id_index(balancing_type_horizon, s1)
    subproblem_id = get_subproblem_id(index)
    horizon = pd.Series(data=create_horizon(base.period_params['period'].iloc[0], len(index)),
                        index = index,
                        name = "horizon_" + balancing_type_horizon)
    s_ = firstcols.join(sumcols).join(subproblem_id).join(horizon)
    
    s_  = create_timepoint_col(s_, horizon)
    s_['linked_timepoint'].iloc[:] = 0
    colnames = ['subproblem_id','stage_id','timepoint','period','number_of_hours_in_timepoint','timepoint_weight','previous_stage_timepoint_map','spinup_or_lookahead','linked_timepoint','month','hour_of_day','timestamp',horizon.name]
    return s_[colnames]


def create_timepoint_col(s_, horizon):
    """
    s_ is dataframe containing columns with name timestamp in '%d-%m-%Y %H:%M' format
    make use
    of these columns to make timepoint 
    """
    s_['timestamp_'] = pd.to_datetime(s_.timestamp, format='%d-%m-%Y %H:%M')

    s_.sort_values('timestamp_', inplace=True)
    s_.reset_index(inplace=True)
    grpbyhorizon = s_.groupby(horizon.name)
    def get_time(h):
        return grpbyhorizon['timestamp_'].first()[h]
        
    lens = [len(s_[s_[horizon.name]==h]) for h in sorted(s_[horizon.name].unique(),
                                                         key=get_time)]
    data = sum([list(range(1, n+1)) for n in lens], start=[])
    timepoint = pd.Series(data = data, name='timepoint')
    s_ = s_.join(timepoint)

    df = s_[[horizon.name, 'timepoint']].astype(str)
    s_['timepoint'] = pd.to_numeric(df[horizon.name] + df['timepoint'].str.zfill(len(str(max(df['timepoint'], key=len)))))
    return s_


def create_horizon_timepoints(structure, balancing_type_horizon):
    grp = structure.groupby('subproblem_id')
    cols = ['subproblem_id', 'stage_id']
    hname = "horizon_" + balancing_type_horizon
    df  = grp[['stage_id', hname, 'timepoint']].first().reset_index()
    balancing_type_horizon = np.array([balancing_type_horizon]*len(df))
    tmp_end = grp['timepoint'].last().reset_index(drop=True)
    df['balancing_type_horizon'] = balancing_type_horizon
    df['tmp_end'] = tmp_end
    df.rename(columns={'timepoint':'tmp_start',
                       hname:'horizon'},
              inplace=True)
    
    return df[['subproblem_id',
               'stage_id',
               'balancing_type_horizon',
               'horizon',
               'tmp_start',
               'tmp_end']]

        
def create_temporal_subscenario_data(base:Subscenario,
                                     balancing_type_horizon:str,
                                     granularity:str,
                                     id_:int):
    hparams = base.horizon_params
    base_balancing_type = hparams['balancing_type_horizon'].unique()[0]
    bt = BALANCING_TYPE
    if balancing_type_horizon not in bt:
        raise ValueError("Wrong input for steps, possible steps are")
    if bt[balancing_type_horizon] > bt[base_balancing_type]:
        raise ValueError(f"New subscenario Can not have more steps than base subscenario")
    
    structure = create_structure(base, balancing_type_horizon, granularity)
    period_params = create_period_params(base)
    horizon_params = create_horizon_params(base, balancing_type_horizon, structure)
    horizon_timepoints = create_horizon_timepoints(structure, balancing_type_horizon )

    return structure,horizon_params,horizon_timepoints,period_params
    

CSV_LOCATION = "/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh"
            
def test_subscenario_class():
    rpo30 = Scenario(CSV_LOCATION, "rpo30")
    assert rpo30.get_subscenario('temporal_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/period_params.csv', '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/horizon_params.csv', '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/structure.csv', '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/horizon_timepoints.csv']
    
    assert rpo30.get_subscenario('load_zone_scenario_id').get_files()==['/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/system/load_zones/1_load_zone_msedcl_voll20.csv']
    assert rpo30.get_subscenario('load_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/system/load/1_load_msedcl_5pc_all.csv', '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/system/load/1_load_msedcl_5pc_all.txt']
    assert rpo30.get_subscenario('project_portfolio_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/portfolios/1_portfolio_msedcl_2030_rpo30.csv']
    assert rpo30.get_subscenario('project_operational_chars_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/opchar/3_opchar_msedcl_rpo30_daily.csv']
    assert rpo30.get_subscenario('project_availability_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/availability/3_availability_rpo30.csv']
    assert rpo30.get_subscenario('project_load_zone_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/load_zones/1_project_load_zones_msedcl.csv']
    assert rpo30.get_subscenario('project_specified_capacity_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/specified/capacity/1_specified_msedcl_2030_rpo30.csv']
    assert rpo30.get_subscenario('project_specified_fixed_cost_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/project/specified/fixed_cost/1_fc_msedcl_2030_rpo30.csv']
    assert rpo30.get_subscenario('solver_options_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/solver/1_cplex_mipgap_0.5.txt', '/home/vikrant/programming/work/publicgit/gridpath/db/csvs_mh/solver/1_cplex_mipgap_0.5.csv']
    assert len(rpo30.get_subscenario('temporal_scenario_id').structure)==365*96
    

def get_subscenarios(scenario, csv_location):
    scenarioscsv =  os.path.join(csv_location, "scenarios.csv")
    with open(scenarioscsv) as f:
        csvf = csv.DictReader(f)
        return {r['optional_feature_or_subscenarios']:r[scenario] for r in csvf if r[scenario]}


def get_subscenario_paths(csv_location):
    def fullpath(p):
        return os.path.join(csv_location, p)
    
    csv_data_master =  os.path.join(csv_location, "csv_data_master.csv")
    with open(csv_data_master) as f:
        csvf = csv.DictReader(f)
        return {r['subscenario']:fullpath(r['path']) for r in csvf if r['path']}


def create_project_operational_chars_subscenario(opchars_base:Subscenario,
                                                 id_:int,
                                                 balancing_type_project:str,
                                                 desc:str):
    opcharsscid = Subscenario(opchars_base.name, id_, opchars_base.csv_location)
    df = opchars_base.data
    ot = df.operational_type.str.replace("gen_commit_bin", "gen_commit_cap")
    df['operational_type'] = ot
    df.balancing_type_project = balancing_type_project
    cols = ['fuel', 'heat_rate_curves_scenario_id', 'variable_om_curves_scenario_id', 'startup_chars_scenario_id', 'startup_cost_per_mw', 'shutdown_cost_per_mw', 'startup_fuel_mmbtu_per_mw', 'startup_plus_ramp_up_rate', 'shutdown_plus_ramp_down_rate', 'ramp_up_when_on_rate', 'ramp_down_when_on_rate', 'ramp_up_violation_penalty', 'ramp_down_violation_penalty', 'min_up_time_hours', 'min_up_time_violation_penalty', 'min_down_time_hours', 'min_down_time_violation_penalty', 'discharging_capacity_multiplier', 'minimum_duration_hours', 'maximum_duration_hours', 'aux_consumption_frac_capacity', 'last_commitment_stage', 'curtailment_cost_per_pwh', 'lf_reserves_up_derate', 'lf_reserves_down_derate', 'regulation_up_derate', 'regulation_down_derate', 'frequency_response_derate', 'spinning_reserves_derate', 'lf_reserves_up_ramp_rate', 'lf_reserves_down_ramp_rate', 'regulation_up_ramp_rate', 'regulation_down_ramp_rate', 'frequency_response_ramp_rate', 'spinning_reserves_ramp_rate']
    df[cols] = np.nan
    filename = f"{id_}_{desc}"
    opcharsscid.writedata(None, **{filename:df})
    return opcharsscid

def next_available_subscenario_id(subscenario:Subscenario):
    folder = subscenario.get_folder()
    p = re.compile(r'(?P<id>\d{1,2})_.*')
    files = [f for f in os.listdir(folder) if p.match(f)]
    return max([int(p.match(f).groupdict()['id']) for f in files])+1

def get_opchars_file_desc(opchars, currentsteps):
    p = re.compile('{id}_(?P<desc>.*)_.+.csv'.format(id=opchars.id_))
    m = p.match(os.path.basename(opchars.files[0]))
    return "_".join([m.groupdict()['desc'],currentsteps+"ly"])

def create_new_scenario(base_scenario,
                        output_scenario,
                        csv_location,
                        steps,
                        granularity,
                        availability,
                        endogenous):
    """
    create a new scenario using a base scenario (assumed daily).
    In the new scenario temporal definations are 
    different. accordingly all time dependent files 
    should change.
    """
    base = Scenario(csv_location, base_scenario)
    temporal_base = base.get_subscenario('temporal_scenario_id')
    t_id = next_available_subscenario_id(temporal_base)
    temporal = create_temporal_subscenario(temporal_base, steps, granularity, t_id)
    project_availability_base = base.get_subscenario('project_availability_scenario_id')
    pa_id = next_available_subscenario_id(project_availability_base)
    project_availability = create_availability_subscenario(csv_location,
                                                           availability,
                                                           endogenous,
                                                           description="endo",
                                                           id_=pa_id)
    popchars_base = base.get_subscenario('project_operational_chars_scenario_id')
    popchars_id = next_available_subscenario_id(popchars_base)
    desc = get_opchars_file_desc(popchars_base, steps)
    popchars = create_project_operational_chars_subscenario(popchars_base,
                                                            popchars_id,
                                                            steps,
                                                            desc
                                                            )

    scenarios = pd.read_csv(base.get_scenarios_csv())
    scenarios.set_index('optional_feature_or_subscenarios', inplace=True)
    c = scenarios[base_scenario].copy()
    c['temporal_scenario_id'] = temporal.id_
    c['project_operational_chars_scenario_id'] = popchars.id_
    c['project_availability_scenario_id'] = project_availability.id_
    scenarios[output_scenario] = c
    scenarios.reset_index(inplace=True)
    scenarios.to_csv(base.get_scenarios_csv(), index=False)
    
    
    

@click.command()
@click.option("-b", "--base_scenario", default="rpo30", help="Base scenario from which other scenario will be generated.")
@click.option("-o", "--output_scenario", default="rpo30_monthly", help="Name of scenario to generate")
@click.option("-s", "--steps", default="month", help="steps as one of year, month, day")
@click.option("-g", "--granularity", default="daily", help="granularity option as one of 15min, hourly, daily, monthly, yearly")
@click.option("-a", "--availability", default=None, help="path to availability data")
@click.option("-e", "--endo", default=None, help="Path to file which contains endogenous availability data in single file")
@click.option("-c", "--csv_location", default="csvs_toy", help="Path to folder where csvs are")
@click.option("--dev/--no-dev", default=False, help="Run only for one project")
def main(base_scenario,
         output_scenario,
         steps,
         granularity,
         availability,
         endo,
         csv_location,
         dev):
    create_new_scenario(base_scenario,
                        output_scenario,
                        csv_location,
                        steps,
                        granularity,
                        availability,
                        endo
                        )
    
if __name__ == "__main__":
    main()
