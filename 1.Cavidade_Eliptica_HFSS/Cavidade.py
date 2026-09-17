import os
import numpy as np

try:
    from ansys.aedt.core import Hfss
    from ansys.aedt.core.generic.constants import Plane
except ImportError as e:
    print(f"Erro ao importar ansys.aedt.core: {e}")
    exit(1)

class CavidadeEliptica:
    def __init__(self, non_graphical=False):
        print("Iniciando o HFSS via PyAEDT...")
        self.hfss = Hfss(
            non_graphical=non_graphical,
            new_desktop=False,
            project="Cavidade_Eliptica_Proj",
            design="Cavidade",
            solution_type="DrivenModal",
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
        
        self.hfss["$a2"] = "$a + $cobre_pec"
        self.hfss["$b2"] = "$b + $cobre_pec"
        # O vetor L2 precisa varrer do topo (+$cobre_pec) até o fundo (-$cobre_pec abaixo de L)
        self.hfss["$L2"] = "$L - 2 * $cobre_pec"
        
        # Calculando $c no Python para evitar bug do Ansys avaliar a raiz em metros e quebrar a simulação
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
        """Constrói a cavidade de vidro, a carcaça de metal e os conectores."""
        print("Construindo geometria 3D...")
        self.hfss.modeler.model_units = "mm"

        # 1. Cavidade de Vidro (Interna)
        # Cria a elipse base e extruda para formar o cilindro elíptico
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
        # Começa no topo com a espessura do cobre ($cobre_pec = 1mm) e vai até o fundo
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
            # A altura total perfura o metal e sobe pelo conector
            teflon = self.hfss.modeler.create_cylinder(
                orientation="Z",
                origin=[fx, 0, 0],
                radius="$conector_rad",
                height="$cobre_pec + $conector_h",
                name=f"Teflon_{port_names[i]}",
                material="Teflon_based"
            )
            
            # Pino Interno (Cobre)
            # Desce de dentro do teflon até dentro da cavidade de vidro
            pino = self.hfss.modeler.create_cylinder(
                orientation="Z",
                origin=[fx, 0, "$cobre_pec + $conector_h"],
                radius="$conector_rad_in",
                height="-$conector_h_m - $cobre_pec - $conector_h",
                name=f"Pino_{port_names[i]}",
                material="copper"
            )
            
            # Subtrações Booleans
            # Fura o metal externo para a passagem do Teflon
            self.hfss.modeler.subtract(tool_list=[teflon.name], blank_list=[metal_cavity.name], keep_originals=True)
            # Fura o Teflon para colocar o pino
            self.hfss.modeler.subtract(tool_list=[pino.name], blank_list=[teflon.name], keep_originals=True)
            # Fura o Vidro da cavidade para colocar o pino
            self.hfss.modeler.subtract(tool_list=[pino.name], blank_list=[glass_cavity.name], keep_originals=True)
            
            # Condutor Externo do Coaxial (PEC na parede externa do Teflon que sobe)
            teflon_faces = teflon.faces
            # Identificar a face do topo para a Wave Port
            top_face = teflon.top_face_z
            bottom_face = teflon.bottom_face_z
            
            # Aplica a condição de contorno PEC no cilindro ao redor do Teflon
            for f in teflon_faces:
                if f.id != top_face.id and f.id != bottom_face.id:
                    self.hfss.assign_perfecte_to_sheets(f.id, name=f"PEC_Outer_{teflon.name}")
            
            # Wave Port no topo do Teflon
            self.hfss.wave_port(
                assignment=top_face.id,
                name=port_names[i]
            )

        # ---- CRIAÇÃO DAS FOLHAS PARA PLOTAGEM DE CAMPOS (Antes da Simulação) ----
        # Vamos usar os valores em float no Python para ser 100% à prova de falhas do Ansys
        a_val = 36.37
        b_val = 24.26
        L_val = -28.89
        
        # Folha no meio da cavidade (z = L/2) para visualização XY
        plot_sheet_xy = self.hfss.modeler.create_ellipse(
            orientation=Plane.XY,
            origin=[0, 0, L_val/2],
            major_radius=a_val,
            ratio=b_val/a_val,
            is_covered=True,
            name="Plot_Sheet_XY"
        )
        plot_sheet_xy.model = False  # Desativa para não afetar a física (apenas para gráficos)
        
        # Folha de corte transversal XZ (passando pelos focos)
        # Usando polyline ponto a ponto para evitar que o Ansys jogue o retângulo pra fora
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

        print("Geometria finalizada com sucesso.")

    def setup_analysis(self):
        """Configura o setup de simulação modal e o sweep de frequência."""
        print("Configurando análise...")
        if "Setup1" in self.hfss.setup_names:
            setup = self.hfss.get_setup("Setup1")
        else:
            setup = self.hfss.create_setup("Setup1")
            setup.props["Frequency"] = "2.87GHz"
            setup.props["MaximumPasses"] = 15
            setup.props["MinimumConvergedPasses"] = 2
            setup.props["PercentRefinement"] = 30
        
        if "Sweep1" not in setup.get_sweep_names():
            self.hfss.create_linear_count_sweep(
                setup=setup.name,
                name="Sweep1", 
                unit="GHz", 
                start_frequency=1.5, 
                stop_frequency=3.5, 
                num_of_freq_points=1001,
                sweep_type="Interpolating"
            )

    def plot_fields(self):
        """Cria os planos de corte e plota os campos E e H a 2.87 GHz."""
        print("Criando os gráficos de campos a 2.87 GHz...")
        
        # Cria os gráficos de Campo Elétrico (E) e Magnético (H)
        for quantity in ["Mag_E", "Mag_H"]:
            plot_xy = self.hfss.post.create_fieldplot_surface(
                assignment="Plot_Sheet_XY",
                quantity=quantity,
                setup="Setup1 : LastAdaptive",
                intrinsics={"Freq": "2.87GHz", "Phase": "0deg"},
                plot_name=f"{quantity}_XY"
            )
            plot_xz = self.hfss.post.create_fieldplot_surface(
                assignment="Plot_Sheet_XZ",
                quantity=quantity,
                setup="Setup1 : LastAdaptive",
                intrinsics={"Freq": "2.87GHz", "Phase": "0deg"},
                plot_name=f"{quantity}_XZ"
            )
            # Tenta exportar a imagem do campo, se a versão do PyAEDT suportar diretamente
            try:
                plot_xy.export_image(os.path.join(os.getcwd(), f"{quantity}_XY.jpg"))
                plot_xz.export_image(os.path.join(os.getcwd(), f"{quantity}_XZ.jpg"))
            except Exception:
                pass
                
        print("Gráficos de campo gerados na aba 'Field Overlays' do HFSS!")

        print("Gerando gráficos de Parâmetros S (S11 e S21)...")
        report = self.hfss.post.create_report(
            expressions=["dB(S(P1,P1))", "dB(S(P2,P1))"],
            setup_sweep_name="Setup1 : Sweep1",
            domain="Sweep",
            plot_name="Parametros_S"
        )
        try:
            report.export_to_image(os.path.join(os.getcwd(), "Parametros_S.jpg"))
            print("Gráfico Parametros_S.jpg salvo na pasta!")
        except Exception as e:
            print(f"Erro ao salvar imagem do gráfico S: {e}")

    def close(self):
        self.hfss.release_desktop(close_projects=True, close_desktop=False)

if __name__ == "__main__":
    cavidade = CavidadeEliptica(non_graphical=False)
    print("Projeto gerado no HFSS! Iniciando a simulação...")
    
    # Roda a simulação automaticamente (pode levar alguns minutos)
    cavidade.hfss.analyze_setup("Setup1")
    
    # Plota os campos degenerados em 2.87 GHz
    cavidade.plot_fields()
    
    print("Simulação concluída e campos gerados!")