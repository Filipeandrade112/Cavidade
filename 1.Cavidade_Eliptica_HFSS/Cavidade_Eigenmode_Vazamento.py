import os
import numpy as np

try:
    from ansys.aedt.core import Hfss
    from ansys.aedt.core.generic.constants import Plane
except ImportError as e:
    print(f"Erro ao importar ansys.aedt.core: {e}")
    exit(1)

class CavidadeElipticaVazamento:
    def __init__(self, non_graphical=False, num_modes=2):
        print("Iniciando o HFSS via PyAEDT (Estudo de Vazamento - Espessura Ultra Fina)...")
        self.num_modes = num_modes
        self.hfss = Hfss(
            non_graphical=non_graphical,
            new_desktop=False,
            project="Cavidade_Eliptica_Vazamento_v3",
            design="Cavidade",
            solution_type="Eigenmode",
            remove_lock=True
        )
        self.setup_variables()
        self.create_geometry()
        self.setup_analysis()

    def setup_variables(self):
        print("Configurando variáveis do projeto...")
        self.hfss["$a"] = "36.37mm"
        self.hfss["$b"] = "24.26mm"
        self.hfss["$L"] = "-28.89mm"
        
        # Espessura padrão de placa de circuito impresso (1 oz copper = 35 micrômetros)
        self.hfss["$cobre_pec"] = "0.035mm"
        
        self.hfss["$a2"] = "$a + $cobre_pec"
        self.hfss["$b2"] = "$b + $cobre_pec"
        self.hfss["$L2"] = "$L - 2 * $cobre_pec"

    def create_geometry(self):
        print("Construindo geometria 3D e Caixa de Radiação...")
        self.hfss.modeler.model_units = "mm"

        # 1. Cavidade de Vidro (Interna)
        glass_sheet = self.hfss.modeler.create_ellipse(
            orientation=Plane.XY, origin=[0, 0, 0],
            major_radius="$a", ratio="$b/$a",
            is_covered=True, name="Glass_Cavity"
        )
        self.hfss.modeler.sweep_along_vector(glass_sheet, [0, 0, "$L"])
        glass_cavity = self.hfss.modeler.get_object_from_name("Glass_Cavity")
        glass_cavity.material_name = "glass"
        glass_cavity.transparency = 0.5

        # 2. Casca de Metal usando a técnica eficiente de "Finite Conductivity"
        # Em vez de desenhar a malha do metal microscópica, aplicamos um "Coating".
        print("Aplicando condição Finite Conductivity nas paredes do vidro...")
        try:
            # Cria a condição de "Finite Conductivity" com espessura usando o PyAEDT
            self.hfss.assign_finite_conductivity(
                glass_cavity.name, 
                material="copper", 
                thickness="$cobre_pec", 
                name="Casca_Fina_Cobre"
            )
        except Exception as e:
            print("Falha ao aplicar Coating via script. Aplique manualmente nas faces do vidro.")

        # 3. Caixa de Radiação (Vacuum Box) para permitir que a energia fuja
        box_padding = 15 # Margem de 15mm para a radiação
        rad_box = self.hfss.modeler.create_box(
            origin=["-$a - 15mm", "-$a - 15mm", "15mm"],
            sizes=["2*$a + 30mm", "2*$a + 30mm", "$L - 30mm"],
            name="Radiation_Box",
            matname="vacuum"
        )
        rad_box.transparency = 0.95
        
        # Aplica a condição de Impedância (377 Ohms) nas faces da caixa,
        # que funciona como um "falso PML" / espaço livre e é aceita no Eigenmode.
        try:
            for face in rad_box.faces:
                self.hfss.assign_impedance_to_sheet(face.id, resistance=377, reactance=0, name=f"Imp_Space_{face.id}")
        except Exception as e:
            print("Aviso: Falha ao aplicar Impedância, aplique manualmente na caixa.")

        print("Geometria finalizada.")

    def setup_analysis(self):
        print("Configurando análise...")
        if "Setup1" in self.hfss.setup_names:
            setup = self.hfss.get_setup("Setup1")
        else:
            setup = self.hfss.create_setup("Setup1")
            setup.props["MinimumFrequency"] = "1.5GHz"
            setup.props["NumModes"] = self.num_modes
            setup.props["MaximumPasses"] = 10  # Menos passes para ir rápido
            setup.props["MinimumConvergedPasses"] = 2
            setup.props["PercentRefinement"] = 30

    def close(self):
        pass

if __name__ == "__main__":
    num_modes = 5 # Vamos olhar só os 2 primeiros modos para ser mais rápido
    cavidade = CavidadeElipticaVazamento(non_graphical=False, num_modes=num_modes)
    print("Projeto gerado no HFSS! Iniciando a simulação...")
    
    cavidade.hfss.analyze_setup("Setup1")
    
    print("Simulação Eigenmode de Vazamento concluída! Vá no HFSS e verifique o Solution Data.")

