import os
import numpy as np
from enviroment import GraphwarEnv
from agent import AgenteQLearning

env = GraphwarEnv()
estado = env.reset(seed=42)
agente = AgenteQLearning(env)


def treino():
    
    alphas = [0.1, 0.2]
    gammas = [0.9, 0.99]
    n_episodes_lista = [5000, 10000]

    env = GraphwarEnv()
    for alpha in alphas:
        for gamma in gammas:
            for n_episodes in n_episodes_lista:
                
                print(f"Testando -> Alpha: {alpha} | Gamma: {gamma} | Episódios: {n_episodes}")
        
                agent = AgenteQLearning(env,alpha=alpha,gamma=gamma,epsilon=1.0)

                
                max_de_passos_p_ep = env.max_tiros


                # metrica para ver a precisão geral (melhor q visualizar o resultado somente)
                sucessos_no_bloco = 0
                precisao_geral = 0
                intervalo_log = max(1, n_episodes // 10)

                for episodio in range(n_episodes):
                    state = env.reset()
                    total_recompensa = 0
                    
                    acertou_no_episodio = False

                    for step in range(max_de_passos_p_ep):
                            
                        acao = agent.select_action(state)
                            
                        proximo_estado, r, done, info = env.step(acao)
                        if info.get("acertou", False):
                            acertou_no_episodio = True

                        agent.update(state, acao, r, proximo_estado, done)
                            
                        state = proximo_estado
                        total_recompensa += r
                            
                        if done:
                            break

                    # decair o epslon -> reduz a aleatoridade a cada ep até o 0.01    
                    agent.epsilon = max(0.01, agent.epsilon * 0.995)

                    if acertou_no_episodio:
                        sucessos_no_bloco += 1

                    # Mostra os resultados obtidos de acordo com os parâmetros no episódio a cada 10% do ep
                    if (episodio + 1) % intervalo_log== 0:
                        precisao = (sucessos_no_bloco / intervalo_log) * 100
                        precisao_geral += precisao
                        eps_val = getattr(agent, 'epsilon', 0.1)
                        
                        print(f"Episódio {episodio + 1}/{n_episodes} | "
                              f"Taxa de acerto (episódios): {precisao:.1f}% | "
                              f"Recompensa Última: {total_recompensa:.2f} | "
                              f"Epsilon: {eps_val:.3f}")
                        sucessos_no_bloco = 0
                
                precisao_geral = precisao_geral/10
                print(f"Taxa de acerto geral: {precisao_geral:.1f}%")

                #salvar os pesos de cada combinação de parâmetros
                pasta_destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pesos_testes")
                os.makedirs(pasta_destino, exist_ok=True)
                nome_arquivo = f"pesos_alpha_{alpha}_gamma_{gamma}_ep_{n_episodes}.npy"
                caminho_salvar = os.path.join(pasta_destino, nome_arquivo)
                agent.salvar_pesos(caminho_salvar)

    print("Treinamento concluído")


treino()