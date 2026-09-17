import os
from ansys.aedt.core import Hfss

try:
    hfss = Hfss(project="Test_Material", design="Test", solution_type="Eigenmode", non_graphical=False, new_desktop=False)
    
    mat = hfss.materials.add_material("copper_thin")
    mat.conductivity = 580000000
    try:
        mat.solve_inside = True
        print("Set solve_inside = True successfully.")
    except Exception as e:
        print("Failed to set solve_inside:", e)
    
    print(dir(mat))
    
except Exception as e:
    print("Fatal error:", e)

