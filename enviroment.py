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
        a_coeficientes = [-0.02, -0.01, 0.0, 0.01, 0.02]
        b_coeficientes = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for a in a_coeficientes:
            for b in b_coeficientes:
                self.acoes.append((a,b))

        # chama o reset para sortear as posicoes do obstaculo e do alvo
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        # sorteio do obstaculo
        x_obs = random.uniform(30.0, 48.0)
        y_obs = random.uniform(20.0, 70.0)
        self.pos_obstaculo = (x_obs, y_obs)

        # sorteio do alvo
        x_alvo = random.uniform(55.0, 90.0)
        y_alvo = random.uniform(15.0, 85.0)
        self.pos_alvo = (x_alvo, y_alvo)

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
            curva_percorrida.append((x,y))

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

            # verifica se saiu do mapa  
            if y < 0 or y > self.ALTURA:
                break

            # reduzindo o numero de tiros 
            self.tiros_restantes -= 1

            # calculo do erro vertical
            dx = self.pos_alvo[0] - x
            y_previsto = y + a * (dx) ** 2 + b * (dx)
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


            
