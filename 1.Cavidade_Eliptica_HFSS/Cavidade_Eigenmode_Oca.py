import os
import numpy as np

try:
    from ansys.aedt.core import Hfss
    from ansys.aedt.core.generic.constants import Plane
except ImportError as e:
    print(f"Erro ao importar ansys.aedt.core: {e}")
    exit(1)

class CavidadeElipticaEigenmodeOca:
    def __init__(self, non_graphical=False, num_modes=6):
        print("Iniciando o HFSS via PyAEDT (Eigenmode - Cavidade Oca)...")
        self.num_modes = num_modes
        self.hfss = Hfss(
            non_graphical=non_graphical,
            new_desktop=False,
            project="Cavidade_Eliptica_Eigen_Oca",
            design="Cavidade",
            solution_type="Eigenmode",
            remove_lock=True
        )
        self.setup_variables()
        self.create_geometry()
        self.setup_analysis()

    def setup_variables(self):
        """Define as variáveis do projeto conforme as dimensões da cavidade."""
        print("Configurando variáveis do projeto...")
        # Variáveis Globais do Projeto
        self.hfss["$a"] = "36.37mm"
        self.hfss["$b"] = "24.26mm"
        self.hfss["$L"] = "-28.89mm"
        self.hfss["$cobre_pec"] = "0.001mm"
        
        self.hfss["$a2"] = "$a + $cobre_pec"
        self.hfss["$b2"] = "$b + $cobre_pec"
        # O vetor L2 precisa varrer do topo (+$cobre_pec) até o fundo (-$cobre_pec abaixo de L)
        self.hfss["$L2"] = "$L - 2 * $cobre_pec"
        
        # Calculando $c no Python (mantido apenas por referência, conectores foram removidos)
        a_val = 36.37
        b_val = 24.26
        c_val = np.sqrt(a_val**2 - b_val**2)
        self.hfss["$c"] = f"{c_val}mm"

    def create_geometry(self):
        """Constrói a cavidade de vidro e a carcaça de metal. Os conectores foram removidos."""
        print("Construindo geometria 3D...")
        self.hfss.modeler.model_units = "mm"

        # 1. Cavidade de Vidro (Interna)
        glass_sheet = self.hfss.modeler.create_ellipse(
            orientation=Plane.XY,
            origin=[0, 0, 0],
            major_radius="$a",
            ratio="$b/$a",
            is_covered=True,
            name="Glass_Cavity"
        )
        self.hfss.modeler.sweep_along_vector(glass_sheet, [0, 0, "$L"])
        glass_cavity = self.hfss.modeler.get_object_from_name("Glass_Cavity")
        glass_cavity.material_name = "glass"
        glass_cavity.color = (128, 0, 128)  # Roxo
        glass_cavity.transparency = 0.5

        # 2. Cavidade de Metal (Externa - Casca)
        metal_sheet = self.hfss.modeler.create_ellipse(
            orientation=Plane.XY,
            origin=[0, 0, "$cobre_pec"],
            major_radius="$a2",
            ratio="$b2/$a2",
            is_covered=True,
            name="Metal_Cavity"
        )
        self.hfss.modeler.sweep_along_vector(metal_sheet, [0, 0, "$L2"])
        metal_cavity = self.hfss.modeler.get_object_from_name("Metal_Cavity")
        metal_cavity.material_name = "copper"
        metal_cavity.transparency = 0.8
        
        # Subtrai o vidro do metal para criar a casca (hollow)
        self.hfss.modeler.subtract(tool_list=[glass_cavity.name], blank_list=[metal_cavity.name], keep_originals=True)

        # 3. Portas Coaxiais Removidas
        # Esta é uma simulação puramente oca, revelando apenas os modos teóricos da cavidade.

        # ---- CRIAÇÃO DAS FOLHAS PARA PLOTAGEM DE CAMPOS ----
        a_val = 36.37
        b_val = 24.26
        L_val = -28.89
        
        plot_sheet_xy = self.hfss.modeler.create_ellipse(
            orientation=Plane.XY,
            origin=[0, 0, L_val/2],
            major_radius=a_val,
            ratio=b_val/a_val,
            is_covered=True,
            name="Plot_Sheet_XY"
        )
        plot_sheet_xy.model = False
        
        plot_sheet_xz = self.hfss.modeler.create_polyline(
            points=[
                [-a_val, 0, 0],
                [a_val, 0, 0],
                [a_val, 0, L_val],
                [-a_val, 0, L_val],
                [-a_val, 0, 0]
            ],
            cover_surface=True,
            name="Plot_Sheet_XZ"
        )
        plot_sheet_xz.model = False

        plot_sheet_yz = self.hfss.modeler.create_polyline(
            points=[
                [0, -b_val, 0],
                [0, b_val, 0],
                [0, b_val, L_val],
                [0, -b_val, L_val],
                [0, -b_val, 0]
            ],
            cover_surface=True,
            name="Plot_Sheet_YZ"
        )
        plot_sheet_yz.model = False

        print("Geometria e planos de corte finalizados com sucesso.")

    def setup_analysis(self):
        """Configura o setup de simulação Eigenmode."""
        print("Configurando análise Eigenmode...")
        if "Setup1" in self.hfss.setup_names:
            setup = self.hfss.get_setup("Setup1")
        else:
            setup = self.hfss.create_setup("Setup1")
            # Configurações para Eigenmode
            setup.props["MinimumFrequency"] = "1.5GHz"
            setup.props["NumModes"] = self.num_modes
            setup.props["MaximumPasses"] = 15
            setup.props["MinimumConvergedPasses"] = 2
            setup.props["PercentRefinement"] = 30

    def print_eigenmodes(self):
        print("\n--- Resultados Eigenmode ---")
        print("Para visualizar as frequências de ressonância e gerar os gráficos de campo:")
        print("1. Na interface do HFSS, vá até 'Results' -> 'Solution Data'.")
        print("2. Você verá as frequências calculadas para cada modo.")
        print("3. Para visualizar os campos, selecione 'Plot_Sheet_XY', 'Plot_Sheet_XZ' ou 'Plot_Sheet_YZ', vá em 'Field Overlays' e plote 'Mag_E' ou 'Mag_H'.")

    def plot_fields(self):
        pass

    def close(self):
        pass

if __name__ == "__main__":
    num_modes = 6
    cavidade = CavidadeElipticaEigenmodeOca(non_graphical=False, num_modes=num_modes)
    print("Projeto gerado no HFSS! Iniciando a simulação...")
    
    cavidade.hfss.analyze_setup("Setup1")
    cavidade.print_eigenmodes()
    
    print("Simulação Eigenmode da Cavidade Oca concluída!")

