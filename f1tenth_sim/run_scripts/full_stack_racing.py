from f1tenth_sim.simulator import F1TenthSim_TrueLocation 
from f1tenth_sim.classic_racing.GlobalPurePursuit import GlobalPurePursuit
from f1tenth_sim.classic_racing.particle_filter import ParticleFilter
import numpy as np
from f1tenth_sim.data_tools.specific_plotting.plot_pf_errors import plot_pf_errors
from f1tenth_sim.run_scripts.testing_functions import *



def test_pf_perception():
    test_id = "mu50"
    map_name = "aut"
    planner = GlobalPurePursuit(test_id, False, planner_name="pf_testing")
    # planner = PurePursuit(test_id, True)
    test_full_stack_all_maps(planner, test_id)
    # test_full_stack_single_map(planner, map_name, test_id)
    # test_planning_all_maps(planner, test_id)

    plot_pf_errors("pf_testing", test_id)

def test_full_stack_pure_pursuit():
    test_id = "mu50"
    map_name = "aut"
    planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPurePursuit_FullStack")
    # planner = PurePursuit(test_id, True)
    test_full_stack_all_maps(planner, test_id)
    # test_full_stack_single_map(planner, map_name, test_id)
    # test_planning_all_maps(planner, test_id)

    # calculate_tracking_accuracy(planner.name)
    # plot_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)







if __name__ == "__main__":
    # test_pf_perception()
    test_full_stack_pure_pursuit()



