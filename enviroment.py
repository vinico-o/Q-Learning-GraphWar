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
        a = [-0.02, -0.01, 0.0, 0.01, 0.02]
        b = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for a in a:
            for b in b:
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

