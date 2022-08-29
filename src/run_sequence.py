import click
import common
import scenario_generation
import availability
import hydro_opchars

@click.command()
@click.option("-b", "--base_scenario", default="S1_pass3_24hr", help="Base scenario from which other scenario will be generated.")
@click.option("-A", "--availability_pass1", default="../../../db/toy1/project/availability/1_availability_S1_pass1.csv", help="path to availability data for pass1")
@click.option("-a", "--availability_pass2", default='../../../db/toy1/project/availability/2_availability_S1_pass2.csv', help="path to availability data for pass2")
@click.option("-e", "--endo", default='../../../db/toy1/project/availability/endogenous/endo.csv', help="Path to file which contains endogenous availability data for pass1 in single file")
@click.option("-H", "--hydro_dir1", default='../../../db/toy1/project/opchar/hydro_opchar', help="Path to directory which contains data for hydro_operational_chars_scenario_id for pass1")
@click.option("-h", "--hydro_dir2", default='../../../db/toy1/project/opchar/hydro_opchar', help="Path to directory which contains data for hydro_operational_chars_scenario_id for pass2")
@click.option("-f", "--forced_outage", default=None, help="Path of excel file that has forced outage")
@click.option("-c", "--csv_location", default="../../../db/toy1", help="Path to folder where csvs are")
@click.option("-d", "--database", default="../../../db/toy1.db", help="Path to database")
@click.option("-g", "--gridpath_rep", default="../../..", help="Path of gridpath source repository")
@click.option("-u", "--update/--no-update", default=True, help="Update new data in csv files even if it exists.")
@click.option("-m", "--map_file", default=None, help="Base scenario from which other scenario will be generated.")
#@click.option("-m", "--map_file", default='../../../db/timepoint_map_2025_old_format.xlsx', help="Base scenario from which other scenario will be generated.")

def main(base_scenario,
         availability_pass1,    
         availability_pass2,
         endo,
         hydro_dir1,
         hydro_dir2,
         forced_outage,
         csv_location,
         database,
         gridpath_rep,
         map_file,
         update):
    pass1 = base_scenario + "_auto_pass1"
    scenario_generation.create_new_scenario(base_scenario,
                                            pass1,
                                            csv_location,
                                            "year",
                                            "daily",
                                            availability_pass1,
                                            endo,
                                            hydro_dir1,
                                            database,
                                            gridpath_rep,
                                            map_file,
                                            update
                                            )
    
    
    common.run_scenario(pass1, csv_location, database)
    pass2 = base_scenario + "_auto_pass2"
    scenario_generation.create_new_scenario(base_scenario,
                                            pass2,
                                            csv_location,
                                            "month",
                                            "daily",
                                            availability_pass2,
                                            None,
                                            hydro_dir2,
                                            database,
                                            gridpath_rep,
                                            map_file,
                                            update
                                            )
    
    availability.endogenous_to_exogenous(pass1,
                                                 pass2,
                                                 base_scenario,
                                                 forced_outage,
                                                 csv_location,
                                                 database,
                                                 map_file,
                                                 gridpath_rep,
                                                 skip_scenario2=None,
                                                 project=None,
                                                 name=base_scenario+"auto",
                                                 update_database=True)
    hydro_opchars.hydro_op_chars(pass1,
                                 pass2,
                                 csv_location,
                                 database,
                                 gridpath_rep,
                                 None,
                                 True,
                                 description=base_scenario+"auto")

    
    common.run_scenario(pass2, csv_location, database)
    
    hydro_opchars.hydro_op_chars(pass2,
                                 base_scenario,
                                 csv_location,
                                 database,
                                 gridpath_rep,
                                 None,
                                 True,
                                 description=base_scenario+"auto")

    common.run_scenario(base_scenario, csv_location, database)


               
if __name__ == "__main__":
    main()
