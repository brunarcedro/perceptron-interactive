// Variáveis globais
let currentDataset = null;
let trainData = null;
let testData = null;
let perceptron = null;

// Função principal para gerar dataset
function generateDataset() {
    const datasetType = document.getElementById('dataset-select').value;
    let data;
    
    switch (datasetType) {
        case 'blobs':
            data = DatasetGenerator.generateBlobs();
            break;
        case 'iris':
            data = DatasetGenerator.generateIris();
            break;
        case 'moons':
            const noise = parseFloat(document.getElementById('noise-level').value);
            data = DatasetGenerator.generateMoons(200, noise);
            break;
        case 'cancer':
            data = DatasetGenerator.generateCancer();
            break;
        case 'noisy':
            const separation = parseFloat(document.getElementById('separation').value);
            data = DatasetGenerator.generateNoisyClassification(200, separation);
            break;
        case 'custom':
            data = DatasetGenerator.generateCustom();
            break;
    }
    
    currentDataset = data;
    updateDatasetInfo(data.info);
    Visualization.plotDataset(data.X, data.y);
}

// Função para treinar o Perceptron
function trainPerceptron() {
    if (!currentDataset) {
        alert('Por favor, gere um dataset primeiro!');
        return;
    }
    
    const testSize = parseFloat(document.getElementById('test-size').value) / 100;
    const learningRate = parseFloat(document.getElementById('learning-rate').value);
    const epochs = parseInt(document.getElementById('epochs').value);
    
    // Dividir dados
    const split = trainTestSplit(currentDataset.X, currentDataset.y, testSize);
    
    // Normalizar dados
    const normTrain = normalize(split.XTrain);
    const XTestNorm = normalizeWithParams(split.XTest, normTrain.mean, normTrain.std);
    
    trainData = {
        X: normTrain.data,
        y: split.yTrain,
        XOriginal: split.XTrain
    };
    
    testData = {
        X: XTestNorm,
        y: split.yTest,
        XOriginal: split.XTest
    };
    
    // Treinar perceptron
    perceptron = new Perceptron(learningRate, epochs);
    perceptron.fit(trainData.X, trainData.y);
    
    // Calcular métricas
    const yPredTrain = perceptron.predict(trainData.X);
    const yPredTest = perceptron.predict(testData.X);
    
    const trainAcc = accuracy(trainData.y, yPredTrain);
    const testAcc = accuracy(testData.y, yPredTest);
    
    // Atualizar UI
    updateResults(trainAcc, testAcc);
    Visualization.plotDecisionBoundary(trainData, perceptron);
    Visualization.plotConvergence(perceptron.errorsHistory);
    
    const metrics = Visualization.plotConfusionMatrix(testData.y, yPredTest);
    updateMetricsDisplay(metrics);
    
    Visualization.plotWeights(perceptron.weights, perceptron.bias);
    updateWeightsInterpretation();
}

function updateResults(trainAcc, testAcc) {
    document.getElementById('results').style.display = 'block';
    document.getElementById('train-acc').textContent = (trainAcc * 100).toFixed(1) + '%';
    document.getElementById('test-acc').textContent = (testAcc * 100).toFixed(1) + '%';
    
    const modelInfo = perceptron.getModelInfo();
    document.getElementById('convergence').textContent = 
        modelInfo.converged ? `Época ${modelInfo.convergenceEpoch}` : 'Não convergiu';
    document.getElementById('training-time').textContent = modelInfo.trainTime.toFixed(1);
    
    // Mostrar equação da fronteira
    const boundary = perceptron.getDecisionBoundary();
    if (boundary) {
        document.getElementById('equation').style.display = 'block';
        document.getElementById('equation').innerHTML = 
            `<strong>Equação da Fronteira:</strong> ${boundary.equation}`;
    }
}

function updateDatasetInfo(info) {
    const infoBox = document.getElementById('dataset-info');
    infoBox.innerHTML = `
        <h4>${info.name}</h4>
        <p>${info.description}</p>
        <p><strong>Linearmente Separável:</strong> ${info.linearSeparable ? '✅ Sim' : '❌ Não'}</p>
    `;
}

function updateMetricsDisplay(metrics) {
    const metricsHtml = `
        <div class="metrics-grid" style="margin-top: 20px;">
            <div class="metric-card">
                <div class="metric-label">Precisão</div>
                <div class="metric-value">${(metrics.precision * 100).toFixed(1)}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">${(metrics.recall * 100).toFixed(1)}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">${metrics.f1.toFixed(3)}</div>
            </div>
        </div>
    `;
    
    const existingMetrics = document.querySelector('#confusion-matrix + .metrics-grid');
    if (existingMetrics) {
        existingMetrics.remove();
    }
    document.getElementById('confusion-matrix').insertAdjacentHTML('afterend', metricsHtml);
}

function updateWeightsInterpretation() {
    const interpretation = `
        <div class="info-box" style="margin-top: 20px;">
            <h4>Interpretação dos Pesos</h4>
            <p><strong>w₁ = ${perceptron.weights[0].toFixed(4)}:</strong> Peso para a Feature 1</p>
            <p><strong>w₂ = ${perceptron.weights[1].toFixed(4)}:</strong> Peso para a Feature 2</p>
            <p><strong>bias = ${perceptron.bias.toFixed(4)}:</strong> Termo de deslocamento</p>
            <p>A decisão é tomada calculando: w₁×x₁ + w₂×x₂ + bias</p>
            <p>Se o resultado ≥ 0, classifica como Classe 1; caso contrário, Classe 0.</p>
        </div>
    `;
    
    const existingInterpretation = document.querySelector('#weights-visualization + .info-box');
    if (existingInterpretation) {
        existingInterpretation.remove();
    }
    document.getElementById('weights-visualization').insertAdjacentHTML('afterend', interpretation);
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Dataset selector
    document.getElementById('dataset-select').addEventListener('change', function() {
        const dataset = this.value;
        document.getElementById('noise-control').style.display = 
            dataset === 'moons' ? 'block' : 'none';
        document.getElementById('separation-control').style.display = 
            dataset === 'noisy' ? 'block' : 'none';
    });

    // Range inputs
    document.getElementById('learning-rate').addEventListener('input', function() {
        document.getElementById('lr-value').textContent = this.value;
    });

    document.getElementById('test-size').addEventListener('input', function() {
        document.getElementById('ts-value').textContent = this.value + '%';
    });

    document.getElementById('noise-level').addEventListener('input', function() {
        document.getElementById('noise-value').textContent = this.value;
    });

    document.getElementById('separation').addEventListener('input', function() {
        document.getElementById('sep-value').textContent = this.value;
    });

    // Inicializar com um dataset
    generateDataset();
});