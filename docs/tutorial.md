# Tutorial Completo: Perceptron

## 📚 Índice
1. [Introdução](#introdução)
2. [Fundamentos Teóricos](#fundamentos-teóricos)
3. [Implementação Passo a Passo](#implementação-passo-a-passo)
4. [Datasets e Experimentos](#datasets-e-experimentos)
5. [Análise de Resultados](#análise-de-resultados)
6. [Limitações e Extensões](#limitações-e-extensões)

## Introdução

O Perceptron é um dos algoritmos mais fundamentais do aprendizado de máquina, proposto por Frank Rosenblatt em 1957. É um classificador binário linear que forma a base para o entendimento de redes neurais mais complexas.

### Conceitos Fundamentais:
- **Classificador Linear**: Separa classes usando um hiperplano
- **Supervisionado**: Aprende a partir de exemplos rotulados
- **Online**: Pode aprender incrementalmente
- **Convergência Garantida**: Para dados linearmente separáveis

## Fundamentos Teóricos

### Modelo Matemático

O Perceptron calcula uma soma ponderada das entradas e aplica uma função de ativação:

```
y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

Onde:
- `x₁, x₂, ..., xₙ` são as features de entrada
- `w₁, w₂, ..., wₙ` são os pesos
- `b` é o bias (termo de deslocamento)
- `f` é a função de ativação (step function)

### Função de Ativação

A função step (degrau) é definida como:

```
f(z) = {
    1, se z ≥ 0
    0, se z < 0
}
```

### Regra de Aprendizado

O Perceptron atualiza seus pesos usando a regra delta:

```
w_novo = w_antigo + η × (y_real - y_pred) × x
b_novo = b_antigo + η × (y_real - y_pred)
```

Onde:
- `η` é a taxa de aprendizado (learning rate)
- `y_real` é a classe verdadeira
- `y_pred` é a classe predita

## Implementação Passo a Passo

### Passo 1: Inicialização

```javascript
class Perceptron {
    constructor(learningRate = 0.01, nEpochs = 100) {
        this.learningRate = learningRate;
        this.nEpochs = nEpochs;
        this.weights = null;
        this.bias = null;
    }
}
```

### Passo 2: Função de Ativação

```javascript
activation(x) {
    return x >= 0 ? 1 : 0;
}
```

### Passo 3: Treinamento

```javascript
fit(X, y) {
    // Inicializar pesos com zeros
    this.weights = new Array(X[0].length).fill(0);
    this.bias = 0;
    
    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
        let errors = 0;
        
        for (let i = 0; i < X.length; i++) {
            // Calcular saída
            let linearOutput = this.bias;
            for (let j = 0; j < X[i].length; j++) {
                linearOutput += X[i][j] * this.weights[j];
            }
            
            // Aplicar função de ativação
            const yPred = this.activation(linearOutput);
            
            // Atualizar pesos se houver erro
            const error = y[i] - yPred;
            if (error !== 0) {
                const update = this.learningRate * error;
                this.bias += update;
                for (let j = 0; j < X[i].length; j++) {
                    this.weights[j] += update * X[i][j];
                }
                errors++;
            }
        }
        
        // Parar se convergiu
        if (errors === 0) break;
    }
}
```

### Passo 4: Predição

```javascript
predict(X) {
    const predictions = [];
    for (let i = 0; i < X.length; i++) {
        let sum = this.bias;
        for (let j = 0; j < X[i].length; j++) {
            sum += X[i][j] * this.weights[j];
        }
        predictions.push(this.activation(sum));
    }
    return predictions;
}
```

## Datasets e Experimentos

### Dataset 1: Blobs Sintéticos

**Características:**
- Linearmente separável
- Dois clusters gaussianos
- Ideal para demonstração

**Experimento:**
1. Gere o dataset com 200 amostras
2. Use learning rate = 0.01
3. Observe convergência em ~10-20 épocas
4. Acurácia esperada: ~100%

### Dataset 2: Iris (Setosa vs Versicolor)

**Características:**
- Dataset real de flores
- Duas classes linearmente separáveis
- 2 features selecionadas para visualização

**Experimento:**
1. Use apenas comprimento de sépala e pétala
2. Normalize os dados
3. Compare com diferentes learning rates
4. Acurácia esperada: 95-100%

### Dataset 3: Moons

**Características:**
- NÃO linearmente separável
- Formato de duas luas entrelaçadas
- Demonstra limitações do Perceptron

**Experimento:**
1. Varie o nível de ruído
2. Observe que nunca converge
3. Acurácia máxima: ~60%
4. A linha reta não consegue separar as luas

### Dataset 4: Classificação com Ruído

**Características:**
- Controle sobre separação e ruído
- Útil para testar robustez

**Experimento:**
1. Comece com separação = 3.0 (fácil)
2. Reduza gradualmente até 0.5
3. Observe degradação da performance
4. Identifique o ponto de falha

## Análise de Resultados

### Métricas de Avaliação

1. **Acurácia**: Taxa de acertos geral
   ```
   Acurácia = (VP + VN) / (VP + VN + FP + FN)
   ```

2. **Precisão**: Taxa de verdadeiros positivos entre os preditos positivos
   ```
   Precisão = VP / (VP + FP)
   ```

3. **Recall**: Taxa de verdadeiros positivos entre os reais positivos
   ```
   Recall = VP / (VP + FN)
   ```

4. **F1-Score**: Média harmônica de precisão e recall
   ```
   F1 = 2 × (Precisão × Recall) / (Precisão + Recall)
   ```

### Interpretação da Fronteira de Decisão

Para 2D, a equação da fronteira é:
```
w₁x₁ + w₂x₂ + b = 0
```

Resolvendo para x₂:
```
x₂ = -(w₁/w₂)x₁ - (b/w₂)
```

Isso representa uma linha reta com:
- Inclinação: `-w₁/w₂`
- Intercepto: `-b/w₂`

## Limitações e Extensões

### Limitações do Perceptron

1. **Apenas Linearmente Separável**: Não consegue resolver XOR
2. **Classificação Binária**: Limitado a duas classes
3. **Sem Probabilidades**: Saída é discreta (0 ou 1)
4. **Sensível a Outliers**: Pode ser afetado por pontos extremos

### Extensões e Melhorias

1. **Multi-Layer Perceptron (MLP)**
   - Adiciona camadas ocultas
   - Resolve problemas não-lineares

2. **Adaline**
   - Usa função de ativação linear durante treinamento
   - Minimiza erro quadrático

3. **Perceptron com Kernel**
   - Transforma features para espaço dimensional maior
   - Permite fronteiras não-lineares

4. **Regularização**
   - Adiciona penalidade aos pesos grandes
   - Melhora generalização

### Código de Exemplo: XOR Problem

```javascript
// XOR não é linearmente separável
const X_xor = [
    [0, 0],  // Classe 0
    [1, 1],  // Classe 0
    [0, 1],  // Classe 1
    [1, 0]   // Classe 1
];
const y_xor = [0, 0, 1, 1];

// O Perceptron simples falhará neste problema
// Necessita de MLP ou transformação de features
```

## Exercícios Práticos

### Exercício 1: Variação de Learning Rate
1. Use o dataset Blobs
2. Teste learning rates: 0.001, 0.01, 0.1, 1.0
3. Registre número de épocas até convergência
4. Qual é o trade-off?

### Exercício 2: Análise de Convergência
1. Implemente early stopping
2. Registre erro por época
3. Identifique padrões de convergência
4. Compare diferentes datasets

### Exercício 3: Feature Engineering
1. Pegue o dataset Moons
2. Adicione features polinomiais (x₁², x₂², x₁×x₂)
3. Retreine o Perceptron
4. Melhora a acurácia?

## Conclusão

O Perceptron é fundamental para entender redes neurais modernas. Embora limitado a problemas linearmente separáveis, seus princípios formam a base para arquiteturas mais complexas. A compreensão profunda do Perceptron facilita o aprendizado de deep learning e outras técnicas avançadas.

## Referências

1. Rosenblatt, F. (1958). "The perceptron: A probabilistic model"
2. Minsky, M. & Papert, S. (1969). "Perceptrons"
3. Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"
4. Haykin, S. (2008). "Neural Networks and Learning Machines"