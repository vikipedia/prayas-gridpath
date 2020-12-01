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
    rpo30 = Scenario("/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh", "rpo30")
    
    assert rpo30.scenario_name == "rpo30"
    assert rpo30.csv_location == "/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh"
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
        with open(master) as f:
            csvf = csv.DictReader(f)
            rows = [row for row in csvf if row['subscenario']==self.name]
            sub_types = [row['subscenario_type'] for row in rows]
            filenames = [r['filename'] for r in rows if r['filename']]
            if "dir_subsc_only" in sub_types:
                self.sub_type = "dir_subsc_only"
                subfolders = [f for f in os.listdir(self.get_folder()) if p.match(f)]
                path = os.path.join(self.get_folder(), subfolders[0])
                files = [os.path.join(path, f) for f in filenames]
            if "simple" in sub_types:
                self.sub_type = "simple"
                path = self.get_folder()
                files = [os.path.join(path, f) for f in os.listdir(self.get_folder()) if p.match(f)]
            self.path = path
            self.files = files

    def get_files(self):
        return self.files

class Temporal_Scenario_Id(Subscenario):
    """Documentation for Temporal_Scenario_Id

    """
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
    
    def __init__(self, id_, csv_location):
        super(Temporal_Scenario_Id, self).__init__("temporal_scenario_id", id_, csv_location)
        
    def get_timepoints(self):
        s = self.structure
        s = s[s.spinup_or_lookahead==0]
        return s['timepoint']

    def get_balancing_type_horizon(self):
        hparams = self.horizon_params
        return hparams['balancing_type_horizon'].unique()[0]

    def get_period(self):
        return self.period_params['period'].iloc[0]


    def create_new_subscenario(self,
                               balancing_type_horizon,
                               granularity,
                               id_):
        
        def checkexisting(folder):
            
            return os.path.exists(folder) and [f for f in os.listdir(folder) if f.startswith(str(id_))]

        def writefile(**data):
            for name, value in data.items():
                value.to_csv(os.path.join(tscid_folder,name+".csv"),index=False)


        structure,horizon_params,horizon_timepoints, period_params = create_temporal_subscenario_data(self, balancing_type_horizon, granularity, id_)
        
        folder = self.get_folder()
        steps = len(structure['subproblem_id'].unique())
        granularity = len(structure[structure.spinup_or_lookahead==0])
        d = Temporal_Scenario_Id.GRAN
        gran = d[granularity]
        subfolder = f"{id_}_{steps}steps_{gran}_timepoints"
        tscid_folder = os.path.join(folder, subfolder)
        if checkexisting(tscid_folder):
            raise Exception(f"Folder for temporal_scenario_id = {id_} exists")
        os.makedirs(tscid_folder)
        writefile(structure=structure)
        writefile(horizon_params=horizon_params)
        writefile(horizon_timepoints=horizon_timepoints)
        writefile(period_params=period_params)
        return Temporal_Scenario_Id(id_, self.csv_location)


def test_temporal_scenario_id_class():
    tmp5 = Temporal_Scenario_Id(5, "/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh")
    print(tmp5.structure.shape)
    assert len(tmp5.get_timepoints())==365*96

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
    period = base.get_period()
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


def create_structure(base:Temporal_Scenario_Id,
                     balancing_type_horizon:str,
                     granularity:str):

    ns = [key for key, value in base.GRAN.items() if value == granularity]
    if not ns:
        raise ValueError("Invalid granularity specified. valid granularity values are {}".format(base.GRAN.values()))
    
    size = ns[0]
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
    horizon = pd.Series(data=create_horizon(base.get_period(), len(index)),
                        index = index,
                        name = "horizon_" + balancing_type_horizon)
    s_ = firstcols.join(sumcols).join(subproblem_id).join(horizon)
    
    s_  = create_timepoint_col(s_, horizon, 'timepoint')
    s_['linked_timepoint'].iloc[:] = 0
    colnames = ['subproblem_id','stage_id','timepoint','period','number_of_hours_in_timepoint','timepoint_weight','previous_stage_timepoint_map','spinup_or_lookahead','linked_timepoint','month','hour_of_day','timestamp',horizon.name]
    return s_[colnames]


def create_timepoint_col(s_, horizon, tindex):
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

        
def create_temporal_subscenario_data(base:Temporal_Scenario_Id,
                                     balancing_type_horizon:str,
                                     granularity:str,
                                     id_:int):
    base_balancing_type = base.get_balancing_type_horizon()
    bt = Temporal_Scenario_Id.BALANCING_TYPE
    if bt[balancing_type_horizon] > bt[base_balancing_type]:
        raise ValueError(f"New Scenario Can not have balancing_type_horizon={balancing_type_horizon}")
    
    structure = create_structure(base, balancing_type_horizon, granularity)
    period_params = create_period_params(base)
    horizon_params = create_horizon_params(base, balancing_type_horizon, structure)
    horizon_timepoints = create_horizon_timepoints(structure, balancing_type_horizon )

    return structure,horizon_params,horizon_timepoints,period_params
    

CSV_LOCATION = "/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh"
            
def test_subscenario_class():
    rpo30 = Scenario(CSV_LOCATION, "rpo30")
    assert rpo30.get_subscenario('temporal_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/period_params.csv', '/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/horizon_params.csv', '/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/structure.csv', '/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/temporal/5_365steps_2030_15min_timepoints/horizon_timepoints.csv']
    
    assert rpo30.get_subscenario('load_zone_scenario_id').get_files()==['/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/system/load_zones/1_load_zone_msedcl_voll20.csv']
    assert rpo30.get_subscenario('load_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/system/load/1_load_msedcl_5pc_all.csv', '/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/system/load/1_load_msedcl_5pc_all.txt']
    assert rpo30.get_subscenario('project_portfolio_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/project/portfolios/1_portfolio_msedcl_2030_rpo30.csv']
    assert rpo30.get_subscenario('project_operational_chars_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/project/opchar/3_opchar_msedcl_rpo30_daily.csv']
    assert rpo30.get_subscenario('project_availability_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/project/availability/3_availability_rpo30.csv']
    assert rpo30.get_subscenario('project_load_zone_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/project/load_zones/1_project_load_zones_msedcl.csv']
    assert rpo30.get_subscenario('project_specified_capacity_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/project/specified/capacity/1_specified_msedcl_2030_rpo30.csv']
    assert rpo30.get_subscenario('project_specified_fixed_cost_scenario_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/project/specified/fixed_cost/1_fc_msedcl_2030_rpo30.csv']
    assert rpo30.get_subscenario('solver_options_id').get_files() == ['/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/solver/1_cplex_mipgap_0.5.txt', '/home/vikrant/programming/work/publicgit/gridpath-0.8.1/gridpath/db/csvs_mh/solver/1_cplex_mipgap_0.5.csv']
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
    
def create_new_scenario(base_scenario,
                        output_scenario,
                        csv_location):
    """
    create a new scenario using a base scenario.
    In the new scenario temporal definations are 
    different. accordingly all time dependent files 
    should change.
    """
    subscenarios = get_subscenarios(base_scenario, csv_location)
    paths = get_subscenario_paths(csv_location)
    for subscenario, sid in subscenarios.items():
        if subscenario =="":
            pass




@click.command()
@click.option("-b", "--base_scenario", default="rpo30", help="Base scenario from which other scenario will be generated.")
@click.option("-o", "--output_scenario", default="rpo30_monthly", help="Name of scenario to generate")
@click.option("-g", "--granularity", default="csvs_toy", help="Path to folder where csvs are")
@click.option("-c", "--csv_location", default="csvs_toy", help="Path to folder where csvs are")
@click.option("--dev/--no-dev", default=False, help="Run only for one project")
def main(base_scenario,
         output_scenario,
         csv_location,
         dev):
    create_new_scenario(base_scenario,
                        output_scenario,
                        csv_location)
    
if __name__ == "__main__":
    main()
