import click
import common
import scenario_genaration
import endogenous_exogenous
import hydro_opchars

@click.command()
@click.option("-b", "--base_scenario", default="rpo30", help="Base scenario from which other scenario will be generated.")
@click.option("-A", "--availability_pass1", default=None, help="path to availability data for pass1")
@click.option("-a", "--availability_pass2", default=None, help="path to availability data for pass2")
@click.option("-e", "--endo", default=None, help="Path to file which contains endogenous availability data for pass1 in single file")
@click.option("-H", "--hydro_dir1", default=None, help="Path to directory which contains data for hydro_operational_chars_scenario_id for pass1")
@click.option("-h", "--hydro_dir2", default=None, help="Path to directory which contains data for hydro_operational_chars_scenario_id for pass2")
@click.option("-c", "--csv_location", default="csvs_mh", help="Path to folder where csvs are")
@click.option("-d", "--database", default="../mh.db", help="Path to database")
@click.option("-g", "--gridpath_rep", default="../", help="Path of gridpath source repository")
@click.option("-u", "--update/--no-update", default=False, help="Update new data in csv files even if it exists.")
def main(base_scenario,
         output_scenario,
         availability_pass1,
         availability_pass2,
         endo,
         hydro_dir1,
         hydro_dir2,
         csv_location,
         database,
         gridpath_rep,
         update):
    pass1 = base_scenario + "_auto_pass1"
    scenario_genaration.create_new_scenario(base_scenario,
                                            pass1,
                                            csv_location,
                                            "year",
                                            "daily",
                                            availability_pass1,
                                            endo,
                                            hydro_dir1,
                                            database,
                                            gridpath_rep,
                                            update)
    
    
    common.run_scenario(pass1, csv_location, db_path)
    pass2 = base_scenario + "auto_pass2"
    scenario_genaration.create_new_scenario(base_scenario,
                                            pass2,
                                            csv_location,
                                            "month",
                                            "daily",
                                            availability_pass2,
                                            None,
                                            hydro_dir2,
                                            database,
                                            gridpath_rep,
                                            update)

    endogenous_exogenous.endogenous_to_exogenous(pass1,
                                                 pass2,
                                                 base_scenario,
                                                 None,
                                                 csv_location,
                                                 database,
                                                 gridpath_rep,
                                                 skip_scenario2=None,
                                                 project=None,
                                                 description=base_scenario+"auto",
                                                 update=True)

    hydro_opchars.hydro_op_chars(pass1,
                                 pass2,
                                 csv_location,
                                 database,
                                 gridpath_rep,
                                 None,
                                 True,
                                 description=base_scenario+"auto")

    
    common.run_scenario(pass2, csv_location, db_path)
    
    hydro_opchars.hydro_op_chars(pass2,
                                 base_scenario,
                                 csv_location,
                                 database,
                                 gridpath_rep,
                                 None,
                                 True,
                                 description=base_scenario+"auto")

    common.run_scenario(base_scenario, csv_location, db_path)


               
if __name__ == "__main__":
    main()
