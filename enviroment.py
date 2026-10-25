from visualize import GraphWarVizualizer
import numpy as np
import random

class GraphwarEnv:
    def __init__(self):
        self.LARGURA = 100.0
        self.ALTURA = 100.0
        self.diagonal = np.sqrt(self.LARGURA ** 2 + self.ALTURA ** 2)
        
        self.pos_atirador = (10.0, 50.0)
        self.pos_obstaculo = (40.0, 45.0)
        self.raio_obstaculo = 5.0
        self.raio_alvo = 5.0
        self.max_tiros = 3
        
        # Catalogo de acoes
        # A IA vai escolher uma combinacao de coeficientes a e b
        self.acoes = []
        a_coeficientes = [-0.02, -0.014, -0.007, 0.0, 0.007, 0.014, 0.02]
        b_coeficientes = [-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0]

        for a in a_coeficientes:
            for b in b_coeficientes:
                self.acoes.append((a,b))

        # chama o reset para sortear as posicoes do obstaculo e do alvo
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        # sorteio do alvo
        x_alvo = random.uniform(55.0, 90.0)
        y_alvo = random.uniform(15.0, 85.0)
        self.pos_alvo = (x_alvo, y_alvo)

        # calculo do meio
        x_meio = (self.pos_atirador[0] + x_alvo) / 2.0
        y_meio = (self.pos_atirador[1] + y_alvo) / 2.0

        # sorteio do obstaculo
        x_obs = x_meio + random.uniform(-15, 15)
        y_obs = y_meio + random.uniform(-6, 6)
        self.pos_obstaculo = (x_obs, y_obs)

        self.tiros_restantes = self.max_tiros
        self.ultimo_erro = 0.0

        return self.obter_estado()

    def obter_estado(self):
        estado = {
            "pos_atirador": self.pos_atirador,
            "pos_obstaculo": self.pos_obstaculo,
            "pos_alvo": self.pos_alvo,
            "tiros_restantes": self.tiros_restantes,
            "ultimo_erro": self.ultimo_erro
        }

        return estado

    def gerar_trajetoria(self, a, b):
        x0, y0 = self.pos_atirador

        pontos = []
        # Começa no x0 (10) e vai até a LARGURA (100) inclusive
        for x in range(int(x0), int(self.LARGURA) + 1): 
            # Equação da parábola ancorada no atirador:
            y_calculado = a * (x - x0) ** 2 + b * (x - x0) + y0
            pontos.append((x, y_calculado))

        return pontos

    def step(self, acao_idx):
        acao_escolhida = self.acoes[acao_idx]
        a, b = acao_escolhida

        curva_percorrida = []
        pontos = self.gerar_trajetoria(a,b)

        acertou = False
        colidiu_obstaculo = False
        distancia_minima = float('inf')
        
        # cálculo das metricas
        for (x, y) in pontos:
            # verifica se saiu do mapa (teto ou chão) e trava o ponto exatamente na borda
            if y < 0:
                curva_percorrida.append((x, 0.0))
                break
            elif y > self.ALTURA:
                curva_percorrida.append((x, self.ALTURA))
                break

            curva_percorrida.append((x, y))

            # calcula a distancia euclidiana para o obstaculo
            distancia_obstaculo = np.sqrt((x - self.pos_obstaculo[0]) ** 2 + (y - self.pos_obstaculo[1]) ** 2)
            
            if distancia_obstaculo <= self.raio_obstaculo:
                colidiu_obstaculo = True
                break
            
            # calcula a distancia euclidiana para o alvo
            distancia_alvo = np.sqrt((x - self.pos_alvo[0]) ** 2 + (y - self.pos_alvo[1]) ** 2)

            # atualiza a distancia minima
            if distancia_alvo < distancia_minima:
                distancia_minima = distancia_alvo
            
            # verifica se houve colisao com o alvo
            if distancia_alvo <= self.raio_alvo:
                acertou = True
                break

        # reduzindo o numero de tiros 
        self.tiros_restantes -= 1

        # calculo do erro vertical no X do alvo a partir da origem do disparo
        x0, y0 = self.pos_atirador
        dx_alvo = self.pos_alvo[0] - x0
        y_previsto = a * (dx_alvo ** 2) + b * dx_alvo + y0
        erro = y_previsto - self.pos_alvo[1]

        self.ultimo_erro = erro

        # calcula a recompensa
        r = -1.0 # penalidade padrao por disparo
        if acertou == True:
            r += 20.0
        elif colidiu_obstaculo == True:
            r += -5.0
        else:
            r += -2.0 * (distancia_minima / self.diagonal)

        finalizado = False
        if acertou or self.tiros_restantes <= 0:
            finalizado = True

        info = {
            "curva": curva_percorrida,
            "status": 'acertou' if acertou else ('colidiu_obstaculo' if colidiu_obstaculo else 'errou'),
            "acertou": acertou,
            "colidiu_obstaculo": colidiu_obstaculo,
            "distancia_minima": distancia_minima,
            "recompensa": r,
            "acao_ab": (a,b)
        }
            
        return (self.obter_estado(), r, finalizado, info)

#teste grafico
if __name__ == '__main__':
    env = GraphwarEnv()
    viz = GraphWarVizualizer(env)

    estado = env.reset()
    historico_tiros = []
    finalizado = False

    print("Iniciando rodada de teste com tela...")

    acoes_exemplo = [
        random.randint(0, 25),
        random.randint(0, 25),
        random.randint(0, 25)
    ]

    print(acoes_exemplo)

    while not finalizado:
        acao_teste = acoes_exemplo[len(historico_tiros) % len(acoes_exemplo)]
        estado, recompensa, finalizado, info = env.step(acao_teste)
        
        # Guarda o tiro no histórico para a tela poder desenhar
        historico_tiros.append(info)
        print(f"Tiro realizado: Status = {info['status']}, Recompensa = {recompensa:.2f}")

    # 3. Manda o visualizador desenhar a rodada completa!
    viz.renderizar_episodio(historico_tiros, estado)
