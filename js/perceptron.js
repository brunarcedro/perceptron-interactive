/**
 * perceptron.js
 * Implementação do algoritmo Perceptron para classificação binária
 */

class Perceptron {
    /**
     * Construtor do Perceptron
     * @param {number} learningRate - Taxa de aprendizado (0.0 a 1.0)
     * @param {number} nEpochs - Número máximo de épocas de treinamento
     */
    constructor(learningRate = 0.01, nEpochs = 100) {
        this.learningRate = learningRate;
        this.nEpochs = nEpochs;
        this.weights = null;
        this.bias = null;
        this.errorsHistory = [];
        this.trainTime = 0;
    }

    /**
     * Função de ativação step (degrau)
     * @param {number} x - Entrada líquida
     * @returns {number} 1 se x >= 0, 0 caso contrário
     */
    activation(x) {
        return x >= 0 ? 1 : 0;
    }

    /**
     * Calcula a entrada líquida (weighted sum + bias)
     * @param {Array} X - Matriz de features
     * @returns {Array} Array com as entradas líquidas
     */
    netInput(X) {
        const result = [];
        for (let i = 0; i < X.length; i++) {
            let sum = this.bias;
            for (let j = 0; j < X[i].length; j++) {
                sum += X[i][j] * this.weights[j];
            }
            result.push(sum);
        }
        return result;
    }

    /**
     * Faz predições para novos dados
     * @param {Array} X - Matriz de features para predição
     * @returns {Array} Classes preditas (0 ou 1)
     */
    predict(X) {
        const netInputs = this.netInput(X);
        return netInputs.map(x => this.activation(x));
    }

    /**
     * Treina o perceptron
     * @param {Array} X - Matriz de features de treino
     * @param {Array} y - Array de labels de treino
     * @returns {Perceptron} Instância treinada
     */
    fit(X, y) {
        const startTime = performance.now();
        const nSamples = X.length;
        const nFeatures = X[0].length;
        
        // Inicializar pesos e bias com zeros
        this.weights = new Array(nFeatures).fill(0);
        this.bias = 0;
        this.errorsHistory = [];
        
        // Loop de treinamento
        for (let epoch = 0; epoch < this.nEpochs; epoch++) {
            let errors = 0;
            
            // Para cada exemplo de treinamento
            for (let idx = 0; idx < nSamples; idx++) {
                // Calcular saída líquida
                let linearOutput = this.bias;
                for (let j = 0; j < nFeatures; j++) {
                    linearOutput += X[idx][j] * this.weights[j];
                }
                
                // Aplicar função de ativação
                const yPredicted = this.activation(linearOutput);
                
                // Calcular erro
                const error = y[idx] - yPredicted;
                
                // Atualizar pesos e bias se houver erro (Regra Delta)
                if (error !== 0) {
                    const update = this.learningRate * error;
                    this.bias += update;
                    for (let j = 0; j < nFeatures; j++) {
                        this.weights[j] += update * X[idx][j];
                    }
                    errors++;
                }
            }
            
            // Armazenar número de erros
            this.errorsHistory.push(errors);
            
            // Parada antecipada se convergiu
            if (errors === 0) {
                console.log(`✓ Convergiu na época ${epoch + 1}`);
                break;
            }
        }
        
        this.trainTime = performance.now() - startTime;
        return this;
    }

    /**
     * Retorna informações sobre o modelo treinado
     * @returns {Object} Informações do modelo
     */
    getModelInfo() {
        return {
            weights: this.weights,
            bias: this.bias,
            learningRate: this.learningRate,
            epochs: this.errorsHistory.length,
            converged: this.errorsHistory[this.errorsHistory.length - 1] === 0,
            convergenceEpoch: this.errorsHistory.indexOf(0) + 1,
            trainTime: this.trainTime
        };
    }

    /**
     * Calcula a equação da fronteira de decisão
     * @returns {Object} Coeficientes da equação
     */
    getDecisionBoundary() {
        if (!this.weights || this.weights[1] === 0) {
            return null;
        }
        
        // Para 2D: w1*x1 + w2*x2 + bias = 0
        // Resolvendo para x2: x2 = -(w1/w2)*x1 - (bias/w2)
        const slope = -this.weights[0] / this.weights[1];
        const intercept = -this.bias / this.weights[1];
        
        return {
            slope: slope,
            intercept: intercept,
            equation: `x₂ = ${slope.toFixed(3)} × x₁ + ${intercept.toFixed(3)}`
        };
    }
}

// Exportar para uso em outros módulos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Perceptron;
}