# Q-Learning Aproximado

* **Armazenamento:** Armazena conhecimento por meio de um vetor de pesos.
* **Avaliação:** Avalia suas ações por meio do cálculo baseado em uma função.
* **Generalização:** Melhor para generalização a partir de características.

## Q-Learning com Aproximação Linear

* **Abordagem:** Usa uma função para estimação do valor; estados parecidos podem compartilhar características e ter valores semelhantes. Aqui, o agente aprende os parâmetros da função e não uma tabela gigantesca (como no Q-Learning tabular).
* **Objetivo:** Aprender um vetor de coeficientes que pode ser usado para obter uma aproximação linear da Q-função.
* **Pesos:** Cada peso representa a importância de uma característica, sendo positivo (tende a aumentar o valor estimado) ou negativo (tende a reduzir o valor estimado).
* **Características:** Função que transforma um par (estado, ação) em um vetor numérico:
  $$f(s, a) = (f_1(s, a), f_2(s, a), \dots, f_n(s, a))$$
  Ela descreve aspectos do par $(s, a)$, e as características podem ser diversas.

* **Memória:** O agente nunca memoriza o ambiente inteiro.
* **Propagação:** As características são calculadas a partir do estado e da ação, e os pesos são ajustados com base na experiência do agente. Estados diferentes podem gerar vetores de características parecidos.
* **Qualidade:** Uma ação pode parecer positiva, mas o valor final com base na função depende da qualidade dos pesos aprendidos.

---

## Componentes Principais

* **Entrada:** Vetor de características $f(s, a)$.
* **Saída Prevista:** $Q(s, a)$.
* **Alvo Temporal:** 
  $$y = r + \gamma \max_{a'} Q(s', a')$$
  *(Ao observar uma transição $(s, a, r, s')$, o Q-Learning constrói um novo valor de referência para $Q(s, a)$).*
* **Predição Atual:** $Q(s, a) = w \cdot f(s, a)$.
* **Erro Temporal ($\delta$):** $\delta = y - Q(s, a)$ (Alvo - Predição).
* **Escopo da Atualização:** A atualização de pesos não corrige apenas um estado específico; ela altera a avaliação de todos os estados-ações que compartilham características semelhantes.

---

## Vantagens e Desvantagens

* **Vantagens:**
  * Usa menos memória que a tabela.
  * Generaliza para estados não visitados.
  * Permite aprendizado em espaços grandes.
* **Desvantagens:**
  * Depende de boas características.
  * Pode confundir estados diferentes com características parecidas.
  * Fora do caso tabular, as garantias teóricas são mais frágeis.

---

## Processo de Aprendizado

O aprendizado consiste em ajustar os pesos $w$ para reproduzir estimativas cada vez melhores seguindo o fluxo: **Experiência $\rightarrow$ Alvo $\rightarrow$ Erro $\rightarrow$ Atualização dos pesos**.

Para uma mesma experiência, temos dois valores:
* **Predição atual:** $Q(s, a) = w \cdot f(s, a)$ (o agente estima antes de observar o resultado).
* **Novo alvo:** $y = r + \gamma \max_{a'} Q(s', a')$ (o que a nova experiência sugere para esse valor).

O aprendizado deve aproximar a **predição** do **alvo**.

### Erros da Aproximação ($\delta$)
$$\delta = y - Q(s, a) = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$

* Se $\delta > 0$ $\rightarrow$ Valor subestimado.
* Se $\delta < 0$ $\rightarrow$ Valor superestimado.
* Se $\delta \approx 0$ $\rightarrow$ Predição e alvo próximos.

---

## Minimização do Erro e Gradiente

Uma forma de reduzir a diferença entre alvo e predição é baseada em mínimos quadrados:
* Função de perda: $l(w) = \frac{1}{2} \delta^2$
* Erros maiores recebem maior penalização.

A derivada parcial mede a influência do peso $w_i$ no erro:
$$
\frac{\partial l}{\partial w_i} = -\delta f_i(s, a)
$$
*(Aqui, a característica determina o quanto o peso $w_i$ participa da correção do erro).*

Para reduzir o erro, movemos os pesos na direção oposta ao gradiente:
$$
w_i \leftarrow w_i + \alpha \delta f_i(s, a)
$$

* $\delta > 0$ $\rightarrow$ O resultado foi melhor que o previsto; os pesos das características ativas tendem a aumentar.
* $\delta < 0$ $\rightarrow$ O resultado foi pior que o previsto; os pesos das características ativas tendem a diminuir.
* Se $f_i(s, a) = 0$ $\rightarrow$ Aquela característica não participa da atualização.

---

## Anotações em Relação ao código

Precisamos saber para o agente:

* Distância horizontal relativa	-> Informar onde está o alvo;
* Diferença vertical relativa	-> Informar se o alvo está acima ou abaixo;
* Erro vertical da curva	-> Avaliar quanto a ação se aproxima da altura do alvo;
* Proximidade do obstáculo	-> Avaliar o risco da trajetória;
* Tiros restantes	-> Informar quantas tentativas ainda existem.

Precisamos também normalizar o erro que vamos considerar:

* f erro(s,a) = (y(xalvo) - yalvo)/100
* Erro com sinal: \f_1(s,a)=\frac{y(x_{alvo})-y_{alvo}}{100}\ -> Permite distinguir se estamos passando por cima ou por abaixo do alvo.
* Erro absoluto: \[f_2(s,a)=\frac{|y(x_{alvo})-y_{alvo}|}{100}\] -> O tamanho do desvio, sem considerar a direção.
