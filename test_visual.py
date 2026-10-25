import os
from enviroment import GraphwarEnv
from agent import AgenteQLearning
from visualize import GraphWarVizualizer

QUANTIDADE_JOGADAS = 5


if __name__ == "__main__":
    env = GraphwarEnv()
    agente = AgenteQLearning(env)

    # busca os pesos no arquivo de pesos
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    arquivo_pesos = os.path.join(pasta_script, "pesos_testes", "pesos_alpha_0.1_gamma_0.99_ep_10000.npy")

    carregou = False
    if os.path.exists(arquivo_pesos):
        try:
            agente.carregar_pesos(arquivo_pesos)
            carregou = True
        except ValueError:
            print("Pesos salvos incompatíveis com as features atuais.")

    if not carregou:
        print("Não foi possível carregar os pesos.")
        print("Realize o Treino primeiro.")
        exit()

    # nao permite tiros aleatorios, pois queremos focar somente no teste
    agente.epsilon = 0.0

    # mostra visualmente as jogadas
    for partida in range(1, QUANTIDADE_JOGADAS + 1):
        print(f"\nIniciando Partida {partida}/{QUANTIDADE_JOGADAS}...")
        
        estado = env.reset()
        historico_tiros = []
        finalizado = False

        while not finalizado:
            acao = agente.select_action(estado)
            estado, r, finalizado, info = env.step(acao)
            historico_tiros.append(info)

        # mostra as jogadas
        viz = GraphWarVizualizer(env)
        viz.renderizar_episodio(historico_tiros, estado)
