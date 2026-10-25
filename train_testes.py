import numpy as np
from enviroment import GraphwarEnv
from agent import AgenteQLearning

env = GraphwarEnv()
estado = env.reset(seed=42)
agente = AgenteQLearning(env)


def treino():
    
    alphas = [0.01, 0.1, 0.2]
    gammas = [0.8, 0.90, 0.99]
    n_episodes_lista = [500, 1000]
    epslons = [0.1, 0.5, 1.0]

    env = GraphwarEnv()
    for alpha in alphas:
        for gamma in gammas:
            for n_episodes in n_episodes_lista:
                for epsilon in epslons:
                    print(f"Testando -> Alpha: {alpha} | Gamma: {gamma} | Episódios: {n_episodes} | Epsilon: {epsilon}")
        
                    agent = AgenteQLearning(env,alpha=alpha,gamma=gamma,epsilon=epsilon)

                    
                    max_de_passos_p_ep = 100


                    # metrica para ver a precisão geral (melhor q visualizar o resultado somente)
                    sucessos_no_bloco = 0
                    intervalo_log = max(1, n_episodes // 10)

                    for episodio in range(n_episodes):
                        state = env.reset()
                        total_recompensa = 0
                        
                        for step in range(max_de_passos_p_ep):
                            
                            acao = agent.select_action(state)
                            
                            proximo_estado, r, done, info = env.step(acao)
                            
                            agent.update(state, acao, r, proximo_estado, done)
                            
                            state = proximo_estado
                            total_recompensa += r
                            
                            if done:
                                break
                        
                        if total_recompensa >= 19:
                            sucessos_no_bloco += 1

                        # Mostra os resultados obtidos de acordo com os parâmetros no episódio a cada 10% do ep
                        if (episodio + 1) % intervalo_log== 0:
                            
                            precisao = (sucessos_no_bloco / intervalo_log) * 100
                            eps_val = getattr(agent, 'epsilon', 0.1)
                            
                            print(f"Episódio {episodio + 1}/{n_episodes} | Recompensa Total: {total_recompensa:.2f} | Epsilon: {eps_val:.3f} | Precisão: {precisao:.1f}%")
                            sucessos_no_bloco = 0
                        

    print("Treinamento concluído")


treino()