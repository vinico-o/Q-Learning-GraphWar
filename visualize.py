import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class GraphWarVizualizer:
    
    # Construtor
    def __init__(self, env):
        # env é a instancia do GraphWarEnv
        self.env = env

        # Configura o tamanho da figura
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # configuracoa de cores
        self.cores = {
            'fundo': '#11111b',
            'grade': '#313244',
            'texto': '#cdd6f4',
            'atirador': '#89b4fa',
            'alvo': '#f38ba8',
            'obstaculo': '#6c7086',
            'tiros': ['#f9e2af', '#cba6f7', '#a6e3a1']
        }
        

    # Método para preparar o cenário
    def preparar_cenario(self):
        self.ax.clear()

        # limites
        self.ax.set_xlim(0, self.env.LARGURA)
        self.ax.set_ylim(0, self.env.ALTURA)

        # trava a escala do grafico
        self.ax.set_aspect("equal", adjustable="box")

        # configuiracao da grade
        self.ax.grid(True, which='both', color=self.cores['grade'], linestyle='-', linewidth=0.7)
        
        # textos
        self.ax.set_xlabel('X', color=self.cores['texto'], fontsize=11)
        self.ax.set_ylabel('Y', color=self.cores['texto'], fontsize=11)
        self.ax.set_title('Simulação Graphwar', color=self.cores['texto'], fontsize=13, fontweight='bold')

        self.ax.tick_params(colors=self.cores["texto"])

    # Método para desenhar o atirador
    def desenhar_atirador(self):
        x = self.env.pos_atirador[0]
        y = self.env.pos_atirador[1]

        self.ax.scatter(x, y, color=self.cores["atirador"], s=120, zorder=5)
        self.ax.text(x, y + 3, f"Atirador ({x},{y})", color=self.cores["atirador"], fontsize=8, ha="center")
    
    # Método para desenhar o alvo
    def desenhar_alvo(self):
        x = self.env.pos_alvo[0]
        y = self.env.pos_alvo[1] 

        # cria um circulo de fato (hitbox), nao um ponto expandido
        alvo = plt.patches.Circle((x, y), radius=self.env.raio_alvo, facecolor=self.cores["alvo"], zorder=5)
        self.ax.add_patch(alvo)

        self.ax.text(x, y + 3, f"Alvo ({x},{y})", color=self.cores["alvo"], fontsize=8, ha="center")
          

    # Método para desenhar o obstáculo
    def desenhar_obstaculo(self):
        x = self.env.pos_obstaculo[0]
        y = self.env.pos_obstaculo[1] 

        # cria um circulo de fato (hitbox), nao um ponto expandido
        obstaculo = plt.patches.Circle((x, y), radius=self.env.raio_obstaculo, facecolor=self.cores["obstaculo"], zorder=5, hatch="///")
        self.ax.add_patch(obstaculo)


    # Método para desenhar a trajetória
    def desenhar_trajetoria(self, pontos_curva, indice_tiro, status):
        if len(pontos_curva) >= 2:
            # separar em array de xs e ys
            xs, ys = [], []
            for i in pontos_curva:
                xs.append(i[0])
                ys.append(i[1])

            cor = self.cores["tiros"][indice_tiro % len(self.cores["tiros"])]

            #plot da linha
            self.ax.plot(xs, ys, color=cor, linewidth=2, zorder=10, linestyle='-')

            #verificando status do tiro
            if status == "acertou":
                self.ax.scatter(xs[-1], ys[-1], color="green", s=100, zorder=11)
            elif status == "errou":
                self.ax.scatter(xs[-1], ys[-1], color="red", s=100, zorder=11)
            elif status == "colidiu_obstaculo":
                self.ax.scatter(xs[-1], ys[-1], color="orange", s=100, zorder=11)
            


    # Método para desenhar a HUD
    def desenhar_hud(self, estado, recompensa, info):
        # Extrai os dados do estado e info com valores padrão de segurança
        tiros_restantes = estado.get('tiros_restantes', 0) if isinstance(estado, dict) else getattr(estado, 'tiros_restantes', 0)
        ultimo_erro = estado.get('ultimo_erro', 0.0) if isinstance(estado, dict) else getattr(estado, 'ultimo_erro', 0.0)
        d_min = info.get('distancia_minima', 0.0) if isinstance(info, dict) else getattr(info, 'distancia_minima', 0.0)

        # Determina a mensagem de status do último disparo
        if isinstance(info, dict):
            if info.get('acertou'):
                status_txt = "ACERTOU!"
            elif info.get('colidiu_obstaculo'):
                status_txt = "COLIDIU NO OBSTÁCULO"
            else:
                status_txt = "ERROU"
        else:
            status_txt = "EM ANDAMENTO"

        # Monta a string formatada em várias linhas
        texto_hud = (
            f"TELEMETRIA DA RODADA\n"
            f"• Status: {status_txt}\n"
            f"• Tiros restantes: {tiros_restantes}\n"
            f"• Recompensa: {recompensa:+.2f}\n"
            f"• Distância mínima: {d_min:.2f}\n"
            f"• Erro vertical: {ultimo_erro:+.2f}"
        )

        # Estilo do hud
        estilo_caixa = dict(
            facecolor='#181825',
            edgecolor=self.cores['grade'],
            alpha=0.9
        )

        # Plota o texto ancorado no topo esquerdo
        self.ax.text(
            0.03, 0.96,
            texto_hud,
            transform=self.ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            color=self.cores['texto'],
            bbox=estilo_caixa,
            zorder=20,
            fontfamily='monospace'
        )

    # Método para renderizar o episódio
    def renderizar_episodio(self, historico_tiros, estado_final):
        self.preparar_cenario()

        self.desenhar_atirador()
        self.desenhar_alvo()
        self.desenhar_obstaculo()

        # desenha todas as trajetorias
        # Ideia do passo a passo:
        for indice, tiro in enumerate(historico_tiros):
            pontos = tiro.get('curva', [])
            status = tiro.get('status', 'errou')
            self.desenhar_trajetoria(pontos, indice, status)

            # Atualiza a HUD para este tiro específico
            self.desenhar_hud(estado_final, tiro.get('recompensa', 0), tiro)

            # Dá tempo para o usuário enxergar o tiro na tela antes do próximo:
            plt.draw()
            plt.pause(1.0) # Espera 1 segundo
        
        self.ax.legend(loc="upper right", labels=["tiro 1", "tiro 2", "tiro 3"], title="Trajetórias", fontsize=8)
        plt.tight_layout()
        plt.show()

    # Método para animar o tiro
    def animar_tiro(self, pontos_curva, indice_tiro=0):
        if not pontos_curva or len(pontos_curva) < 2:
            return

        cor = self.cores["tiros"][indice_tiro % len(self.cores["tiros"])]

        # Cria uma linha vazia no plano
        linha, = self.ax.plot([], [], color=cor, linewidth=2.5, zorder=10)

        xs_anim = []
        ys_anim = []

        #Percorre os pontos aos poucos
        passo = 2
        for i in range(0, len(pontos_curva), passo):
            sub_pontos = pontos_curva[i:i + passo]
            for p in sub_pontos:
                xs_anim.append(p[0])
                ys_anim.append(p[1])

            # Atualiza os dados da linha gradativamente
            linha.set_data(xs_anim, ys_anim)
            plt.draw()
            plt.pause(0.008) 

        # Garante que o ponto final exato seja incluído
        if xs_anim[-1] != pontos_curva[-1][0]:
            xs_anim.append(pontos_curva[-1][0])
            ys_anim.append(pontos_curva[-1][1])
            linha.set_data(xs_anim, ys_anim)

            plt.draw()