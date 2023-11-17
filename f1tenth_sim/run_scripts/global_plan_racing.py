from f1tenth_sim.classic_racing.GlobalPurePursuit import GlobalPurePursuit
from f1tenth_sim.classic_racing.GlobalMPCC import GlobalMPCC

from f1tenth_sim.data_tools.calculations.calculate_tracking_accuracy import calculate_tracking_accuracy
from f1tenth_sim.data_tools.general_plotting.plot_raceline_tracking import plot_raceline_tracking
from f1tenth_sim.data_tools.general_plotting.plot_trajectory_analysis import plot_trajectory_analysis

from f1tenth_sim.run_scripts.testing_functions import *


def test_pure_pursuit():
    # test_id = "mu75"
    test_id = "mu50"
    # test_id = "mu70"
    map_name = "aut"
    planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP")
    # planner = PurePursuit(test_id, True)
    # test_planning_single_map(planner, map_name, test_id)
    test_planning_all_maps(planner, test_id)

    # calculate_tracking_accuracy(planner.name)
    plot_trajectory_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)


def test_mpcc():
    test_id = "mu70"
    map_name = "aut"
    planner = GlobalMPCC(test_id, True, planner_name="GlobalPlanMPCC")
    test_planning_single_map(planner, map_name, test_id)
    # test_planning_all_maps(planner, test_id)





if __name__ == "__main__":

    test_pure_pursuit()
    # test_mpcc()




