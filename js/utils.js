function normalize(data) {
    const mean = [];
    const std = [];
    const normalized = [];
    
    // Calcular média e desvio padrão
    for (let j = 0; j < data[0].length; j++) {
        const column = data.map(row => row[j]);
        const m = column.reduce((a, b) => a + b) / column.length;
        const s = Math.sqrt(column.reduce((a, b) => a + Math.pow(b - m, 2), 0) / column.length);
        mean.push(m);
        std.push(s || 1); // Evitar divisão por zero
    }
    
    // Normalizar
    for (let i = 0; i < data.length; i++) {
        const row = [];
        for (let j = 0; j < data[i].length; j++) {
            row.push((data[i][j] - mean[j]) / std[j]);
        }
        normalized.push(row);
    }
    
    return { data: normalized, mean, std };
}

function normalizeWithParams(data, mean, std) {
    const normalized = [];
    for (let i = 0; i < data.length; i++) {
        const row = [];
        for (let j = 0; j < data[i].length; j++) {
            row.push((data[i][j] - mean[j]) / std[j]);
        }
        normalized.push(row);
    }
    return normalized;
}

function trainTestSplit(X, y, testSize) {
    const n = X.length;
    const indices = Array.from({length: n}, (_, i) => i);
    
    // Shuffle
    for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    
    const testCount = Math.floor(n * testSize);
    const trainIndices = indices.slice(testCount);
    const testIndices = indices.slice(0, testCount);
    
    const XTrain = trainIndices.map(i => X[i]);
    const yTrain = trainIndices.map(i => y[i]);
    const XTest = testIndices.map(i => X[i]);
    const yTest = testIndices.map(i => y[i]);
    
    return { XTrain, yTrain, XTest, yTest };
}

function accuracy(yTrue, yPred) {
    let correct = 0;
    for (let i = 0; i < yTrue.length; i++) {
        if (yTrue[i] === yPred[i]) correct++;
    }
    return correct / yTrue.length;
}

function showTab(tabName) {
    // Esconder todas as tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remover classe active de todos os botões
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Mostrar tab selecionada
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Adicionar classe active ao botão clicado
    event.target.classList.add('active');
}