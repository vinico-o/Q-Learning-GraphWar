import numpy as np 
import random

class AgenteQLearning:
    #Definindo o construtor do agente
    def __init__(self,env,alpha=0.1,gamma=0.9,epsilon=0.1):
        self.NUM_FEATURES = 10
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  
        self.pesos = np.zeros(self.NUM_FEATURES)
    
    #Recebe o estado atual do jogo e o índice de uma das 25 ações.
    #Apenas calcula as características daquela combinação de estado e ação
    
    def extract_features(self,estado, acao_idx):
        #Característica 0 - Viés
        f0 = 1.0
        
        #Característica 1 - Distância horizontal normalizada entre o alvo e atirador - Intervalo [-1,1]
        xatirador, yatirador = estado["pos_atirador"]
        xalvo, yalvo = estado["pos_alvo"]
        xobs,yobs = estado["pos_obstaculo"]
        f1 = (xalvo - xatirador)/self.env.LARGURA #Val igauis para 25 ações
        f1 = np.interp(f1,[0.45,0.8],[-1.,1.])
        
        #Característica 2 - Distância vertical normalizada entre o alvo e atirador - Intervalo [-1,1]
        f2 = (yalvo - yatirador)/self.env.ALTURA # val iguais para 25 ações
        f2 = np.interp(f2,[-0.35,0.35],[-1.,1.])
        
        #Característica 3 - Erro vertical com sinal em relação ao alvo - Intervalo [-1,1]
        a,b = self.env.acoes[acao_idx]
        ycalc = yatirador + a*(xalvo-xatirador)**2 + b*(xalvo-xatirador)
        f3 = (ycalc - yalvo)/self.env.ALTURA
        f3 = np.interp(f3,[-2.43,2.43],[-1,1])
        
        #Característica 4 - Erro vertical absoluto em relação ao alvo - Intervalo: 0-1
        f4 = abs(ycalc-yalvo)/self.env.ALTURA
        f4 = np.interp(f4,[0,2.43],[0.,1.])
        
        #Característica 5 - Erro vertical com sinal em relação ao obstaculo - Intervalo [-1,1]
        ycalc_obs = yatirador + a*(xobs-xatirador)**2 + b*(xobs-xatirador)
        f5 = (ycalc_obs- yobs)/self.env.ALTURA
        f5 = np.clip(f5 / 0.9688, -1.0, 1.0)
        
        #Característica 6 - Erro vertical absoluto em relação ao obstaculo - Intervalo: 0-1
        f6 = abs(ycalc_obs - yobs)/self.env.ALTURA
        f6 = np.interp(f6,[0.,0.9688],[0.,1.])
        
        #Característica 7 - Tiros restantes
        f7 = estado["tiros_restantes"]/self.env.max_tiros
        
        #caracteristica 8 e 9 - Erro do tiro anterior
        # essa feature combinada com os coeficientes é importante 
        # para ajustar se o tiro vai mais para cima ou para baixo
        # Tentando resolver um problema, onde a IA manda o tiro sempre para o mesmo lugar quando erra
        
        ultimo_erro_normalizado = estado["ultimo_erro"] / self.env.ALTURA
        a_normalizado = np.interp(a,[-0.02,0.02],[-1,1])
        b_normalizado = np.interp(b,[-1,1],[-1,1])
        f8 = ultimo_erro_normalizado * a_normalizado
        f9 = ultimo_erro_normalizado * b_normalizado

        array_features = np.array([f0,f1,f2,f3,f4,f5,f6,f7,f8,f9])
        
        return array_features
    
    
    def q_value(self,estado,acao_idx):
        features = self.extract_features(estado,acao_idx)
        
        Q = np.dot(self.pesos,features)
        
        return Q
    
    def select_action(self,estado):
        num_rnd = np.random.rand()
        if num_rnd<self.epsilon:
            acao_rnd = np.random.randint(0,len(self.env.acoes))
            return acao_rnd
        else:
            q_max = self.q_value(estado,0)
            melhores_acoes = [0]
            for i in range(1,len(self.env.acoes)):
                q = self.q_value(estado,i)
                if q> q_max:
                    q_max = q
                    melhores_acoes = [i]
                elif q == q_max:
                    melhores_acoes.append(i)
            return np.random.choice(melhores_acoes) #Sorteia uma das melhores ações.
        
    def update(self,estado,acao_idx,r,proximo_estado,done):
        if done:
            q_futuro = 0
        else:
            q_futuro = self.q_value(proximo_estado,0)
            for i in range(1,len(self.env.acoes)):
                q = self.q_value(proximo_estado,i)
                if q> q_futuro:
                    q_futuro = q
        q_atual = self.q_value(estado,acao_idx)
        
        erro_temporal = r + self.gamma*q_futuro - q_atual
        
        features = self.extract_features(estado,acao_idx)
        for i in range (self.NUM_FEATURES):
            self.pesos[i]  = self.pesos[i] + self.alpha*erro_temporal*features[i]
        tds_verdade = np.all(np.isfinite(self.pesos))
        if not tds_verdade:
            raise ValueError("Foram encontrados pesos inválidos (NaN ou infinito).")
            
    def salvar_pesos(self,caminho):
        np.save(caminho,self.pesos)
        
    def carregar_pesos(self,caminho):
        pesos = np.load(caminho)
        tds_verdade = np.all(np.isfinite(pesos))
        if len(pesos) == self.NUM_FEATURES and tds_verdade:
            self.pesos = pesos
        else:
            raise ValueError("Numero de pesos importados diferente do número de features ou algum peso é 'inf'")

#Teste foi feito pela IA ai eu depois eu entendi oque tava acontecendo
if __name__ == "__main__":
    from enviroment import GraphwarEnv

    # Criar ambiente e agente
    env = GraphwarEnv()
    agente = AgenteQLearning(env)

    # Obter um estado inicial
    estado = env.reset(seed=42)
    acao = 7

    # Verificar características e Q
    print("Features:", agente.extract_features(estado, acao))
    print("Q inicial:", agente.q_value(estado, acao))
    print("Pesos antes:", agente.pesos.copy())

    # Simular uma atualização com recompensa positiva
    agente.update(estado, acao, 10, estado, True)

    print("Pesos depois:", agente.pesos)
    print("Q depois:", agente.q_value(estado, acao))