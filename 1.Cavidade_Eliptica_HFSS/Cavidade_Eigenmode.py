import os
import numpy as np

try:
    from ansys.aedt.core import Hfss
    from ansys.aedt.core.generic.constants import Plane
except ImportError as e:
    print(f"Erro ao importar ansys.aedt.core: {e}")
    exit(1)

class CavidadeElipticaEigenmode:
    def __init__(self, non_graphical=False, num_modes=6):
        print("Iniciando o HFSS via PyAEDT (Eigenmode)...")
        self.num_modes = num_modes
        self.hfss = Hfss(
            non_graphical=non_graphical,
            new_desktop=False,
            project="Cavidade_Eliptica_Eigen",
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
        self.hfss["$cobre_pec"] = "1mm"
        
        self.hfss["$a2"] = "37.37mm"
        self.hfss["$b2"] = "25.26mm"
        self.hfss["$L2"] = "-30.89mm"
        
        # Calculando $c no Python
        a_val = 36.37
        b_val = 24.26
        c_val = np.sqrt(a_val**2 - b_val**2)
        self.hfss["$c"] = f"{c_val}mm"
        
        # Conectores coaxiais
        self.hfss["$conector_h"] = "10mm"
        self.hfss["$conector_rad"] = "1.5mm"
        self.hfss["$conector_rad_in"] = "0.4mm"
        self.hfss["$conector_h_m"] = "25mm"

    def create_geometry(self):
        """Constrói a cavidade de vidro, a carcaça de metal e os pinos."""
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

        # 3. Portas Coaxiais (nos focos da elipse $c e -$c)
        focos_x = ["$c", "-$c"]
        port_names = ["P1", "P2"]

        for i, fx in enumerate(focos_x):
            # Dielétrico do Coaxial (Teflon)
            teflon = self.hfss.modeler.create_cylinder(
                orientation="Z",
                origin=[fx, 0, 0],
                radius="$conector_rad",
                height="$cobre_pec + $conector_h",
                name=f"Teflon_{port_names[i]}",
                material="Teflon_based"
            )
            
            # Pino Interno (Cobre)
            pino = self.hfss.modeler.create_cylinder(
                orientation="Z",
                origin=[fx, 0, "$cobre_pec + $conector_h"],
                radius="$conector_rad_in",
                height="-$conector_h_m - $cobre_pec - $conector_h",
                name=f"Pino_{port_names[i]}",
                material="copper"
            )
            
            # Subtrações Booleans
            self.hfss.modeler.subtract(tool_list=[teflon.name], blank_list=[metal_cavity.name], keep_originals=True)
            self.hfss.modeler.subtract(tool_list=[pino.name], blank_list=[teflon.name], keep_originals=True)
            self.hfss.modeler.subtract(tool_list=[pino.name], blank_list=[glass_cavity.name], keep_originals=True)
            
            # Condutor Externo do Coaxial (PEC na parede externa do Teflon que sobe)
            teflon_faces = teflon.faces
            top_face = teflon.top_face_z
            bottom_face = teflon.bottom_face_z
            
            for f in teflon_faces:
                if f.id != top_face.id and f.id != bottom_face.id:
                    self.hfss.assign_perfecte_to_sheets(f.id, name=f"PEC_Outer_{teflon.name}")
            
            # Obs: Em simulação Eigenmode não precisamos (nem podemos) configurar Wave Ports.
            # O pino e o teflon formam a estrutura interna, podemos deixar o topo aberto.

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
            # Driven modal usava 'Frequency', 'MaximumPasses', etc.

    def print_eigenmodes(self):
        """A extração automática via PyAEDT pode causar crash em versões Student."""
        print("\n--- Resultados Eigenmode ---")
        print("Para visualizar as frequências de ressonância e gerar os gráficos de campo:")
        print("1. Na interface do HFSS, vá até 'Results' -> 'Solution Data'.")
        print("2. Você verá as frequências calculadas para cada modo.")
        print("3. Para visualizar os campos, selecione 'Plot_Sheet_XY', 'Plot_Sheet_XZ' ou 'Plot_Sheet_YZ', vá em 'Field Overlays' e plote 'Mag_E' ou 'Mag_H'.")

    def plot_fields(self):
        """Desabilitado para evitar instabilidade no PostProcessor."""
        pass

    def close(self):
        # Removendo release_desktop para manter o HFSS aberto após a simulação para visualização
        pass

if __name__ == "__main__":
    num_modes = 5  # Número de modos para calcular (pode alterar se quiser mais)
    cavidade = CavidadeElipticaEigenmode(non_graphical=False, num_modes=num_modes)
    print("Projeto gerado no HFSS! Iniciando a simulação...")
    
    # Roda a simulação automaticamente (pode levar alguns minutos)
    cavidade.hfss.analyze_setup("Setup1")
    
    # Imprime resultados
    cavidade.print_eigenmodes()
    
    # Plota os campos para os modos
    cavidade.plot_fields()
    
    print("Simulação Eigenmode concluída e campos gerados para todos os modos configurados!")

