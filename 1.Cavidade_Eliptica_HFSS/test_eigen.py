import os
from ansys.aedt.core import Hfss

try:
    hfss = Hfss(project="Cavidade_Eliptica_Eigen", design="Cavidade", solution_type="Eigenmode", non_graphical=False, new_desktop=False)
    
    print("Project opened.")
    
    # Try different ways to get eigenmodes
    try:
        print("Trying to get eigenmodes from setup...")
        setup = hfss.get_setup("Setup1")
        print("Setup found:", setup.name)
        # Check if there is a get_solutions method or similar
        print(dir(setup))
    except Exception as e:
        print("Error getting setup:", e)

    try:
        print("\nTrying hfss.post...")
        # Just creating the Post object might be what crashed, let's see
        print(hfss.post)
    except Exception as e:
        print("Error with hfss.post:", e)

    hfss.release_desktop(close_projects=True, close_desktop=False)
except Exception as e:
    print("Fatal error:", e)

