import numpy as np
from enviroment import GraphwarEnv
from agent import AgenteQLearning

env = GraphwarEnv()
estado = env.reset(seed=42)
agente = AgenteQLearning(env)

# TESTE 1 - Exploração
print("\nTESTE 1 - EXPLORAÇÃO")

agente.epsilon = 1.0

acoes = []
for i in range(20):
    acao = agente.select_action(estado)
    acoes.append(int(acao))

print("Ações sorteadas:", acoes)
print("Todas são válidas:", all(0 <= a < len(env.acoes) for a in acoes))


# TESTE 2 - Explotação
print("\nTESTE 2 - EXPLOTAÇÃO")

agente.epsilon = 0.0
agente.pesos[3] = -1.0

acao_escolhida = agente.select_action(estado)

valores_q = [
    agente.q_value(estado, i)
    for i in range(len(env.acoes))
]

print("Ação escolhida:", acao_escolhida)
print("Q da ação escolhida:", agente.q_value(estado, acao_escolhida))
print("Maior Q encontrado:", max(valores_q))

assert np.isclose(
    agente.q_value(estado, acao_escolhida),
    max(valores_q)
)

print("Teste de explotação passou!")


# TESTE 3 - Atualização terminal
print("\nTESTE 3 - UPDATE TERMINAL")

agente.pesos = np.zeros(agente.NUM_FEATURES)

acao = 7
features = agente.extract_features(estado, acao).copy()
pesos_antes = agente.pesos.copy()

agente.update(estado, acao, 10, estado, True)

pesos_esperados = pesos_antes + agente.alpha * 10 * features

print("Pesos obtidos:", agente.pesos)
print("Pesos esperados:", pesos_esperados)

assert np.allclose(agente.pesos, pesos_esperados)
print("Teste terminal passou!")


# TESTE 4 - Atualização não terminal
print("\nTESTE 4 - UPDATE NÃO TERMINAL")

agente.pesos = np.array([
    0.2, -0.1, 0.3, -0.4,
    0.5, -0.2, 0.1, 0.05
])

pesos_antes = agente.pesos.copy()

proximo_estado = estado.copy()
proximo_estado["tiros_restantes"] = 2

acao = 7
recompensa = -1.0

q_atual = agente.q_value(estado, acao)

q_futuro = max(
    agente.q_value(proximo_estado, i)
    for i in range(len(env.acoes))
)

delta = recompensa + agente.gamma * q_futuro - q_atual

features = agente.extract_features(estado, acao)

pesos_esperados = pesos_antes + agente.alpha * delta * features

agente.update(estado, acao, recompensa, proximo_estado, False)

print("Erro temporal:", delta)
print("Pesos obtidos:", agente.pesos)
print("Pesos esperados:", pesos_esperados)

assert np.allclose(agente.pesos, pesos_esperados)
print("Teste não terminal passou!")



print("\nTESTE 5 - SALVAR E CARREGAR PESOS")

# Criar um agente com pesos conhecidos
agente_original = AgenteQLearning(env)

agente_original.pesos = np.array([
    0.5, -0.3, 1.2, 0.8,
    -0.6, 0.4, 0.9, -0.2
])

print("Pesos originais:", agente_original.pesos)

# Salvar os pesos
agente_original.salvar_pesos("pesos_teste.npy")

# Criar outro agente com pesos zerados
agente_novo = AgenteQLearning(env)

print("Pesos antes de carregar:", agente_novo.pesos)

# Carregar os pesos salvos
agente_novo.carregar_pesos("pesos_teste.npy")

print("Pesos depois de carregar:", agente_novo.pesos)

# Comparar os dois vetores
assert np.array_equal(agente_original.pesos, agente_novo.pesos)

print("Teste de salvar e carregar pesos passou!")