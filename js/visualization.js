/**
 * visualization.js
 * Funções de visualização usando Plotly.js
 */

const Visualization = {
    /**
     * Plota o dataset original
     */
    plotDataset(X, y, elementId = 'decision-plot') {
        const trace0 = {
            x: X.filter((_, i) => y[i] === 0).map(point => point[0]),
            y: X.filter((_, i) => y[i] === 0).map(point => point[1]),
            mode: 'markers',
            name: 'Classe 0',
            marker: {
                color: 'red',
                size: 8,
                line: {
                    color: 'darkred',
                    width: 1
                }
            }
        };
        
        const trace1 = {
            x: X.filter((_, i) => y[i] === 1).map(point => point[0]),
            y: X.filter((_, i) => y[i] === 1).map(point => point[1]),
            mode: 'markers',
            name: 'Classe 1',
            marker: {
                color: 'blue',
                size: 8,
                line: {
                    color: 'darkblue',
                    width: 1
                }
            }
        };
        
        const layout = {
            title: 'Dataset Original',
            xaxis: { title: 'Feature 1' },
            yaxis: { title: 'Feature 2' },
            hovermode: 'closest',
            showlegend: true,
            height: 400
        };
        
        Plotly.newPlot(elementId, [trace0, trace1], layout);
    },

    /**
     * Plota as regiões de decisão e a fronteira
     */
    plotDecisionBoundary(trainData, perceptron, elementId = 'decision-plot') {
        const resolution = 0.02;
        
        // Encontrar limites
        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;
        
        for (const point of trainData.X) {
            xMin = Math.min(xMin, point[0]);
            xMax = Math.max(xMax, point[0]);
            yMin = Math.min(yMin, point[1]);
            yMax = Math.max(yMax, point[1]);
        }
        
        // Adicionar margem
        const margin = 0.5;
        xMin -= margin;
        xMax += margin;
        yMin -= margin;
        yMax += margin;
        
        // Criar grid
        const xRange = [];
        const yRange = [];
        for (let x = xMin; x <= xMax; x += resolution) xRange.push(x);
        for (let y = yMin; y <= yMax; y += resolution) yRange.push(y);
        
        // Classificar cada ponto do grid
        const z = [];
        for (let i = 0; i < yRange.length; i++) {
            const row = [];
            for (let j = 0; j < xRange.length; j++) {
                const point = [[xRange[j], yRange[i]]];
                const pred = perceptron.predict(point);
                row.push(pred[0]);
            }
            z.push(row);
        }
        
        // Criar traces
        const contour = {
            x: xRange,
            y: yRange,
            z: z,
            type: 'contour',
            showscale: false,
            colorscale: [
                [0, 'rgba(255, 0, 0, 0.2)'],
                [1, 'rgba(0, 0, 255, 0.2)']
            ],
            contours: {
                coloring: 'fill',
                showlines: false
            }
        };
        
        // Pontos de treino
        const trace0 = {
            x: trainData.X.filter((_, i) => trainData.y[i] === 0).map(p => p[0]),
            y: trainData.X.filter((_, i) => trainData.y[i] === 0).map(p => p[1]),
            mode: 'markers',
            name: 'Classe 0 (Treino)',
            marker: {
                color: 'red',
                size: 8,
                line: { color: 'darkred', width: 1 }
            }
        };
        
        const trace1 = {
            x: trainData.X.filter((_, i) => trainData.y[i] === 1).map(p => p[0]),
            y: trainData.X.filter((_, i) => trainData.y[i] === 1).map(p => p[1]),
            mode: 'markers',
            name: 'Classe 1 (Treino)',
            marker: {
                color: 'blue',
                size: 8,
                line: { color: 'darkblue', width: 1 }
            }
        };
        
        // Linha de decisão
        const lineX = [xMin, xMax];
        const lineY = lineX.map(x => 
            -(perceptron.weights[0] * x + perceptron.bias) / perceptron.weights[1]
        );
        
        const decisionLine = {
            x: lineX,
            y: lineY,
            mode: 'lines',
            name: 'Fronteira de Decisão',
            line: {
                color: 'black',
                width: 2,
                dash: 'dash'
            }
        };
        
        const layout = {
            title: 'Regiões de Decisão',
            xaxis: { title: 'Feature 1 (Normalizada)' },
            yaxis: { title: 'Feature 2 (Normalizada)' },
            hovermode: 'closest',
            showlegend: true,
            height: 400
        };
        
        Plotly.newPlot(elementId, [contour, trace0, trace1, decisionLine], layout);
    },

    /**
     * Plota a curva de convergência
     */
    plotConvergence(errorsHistory, elementId = 'convergence-plot') {
        const trace = {
            x: Array.from({length: errorsHistory.length}, (_, i) => i + 1),
            y: errorsHistory,
            mode: 'lines+markers',
            name: 'Erros',
            line: {
                color: 'purple',
                width: 2
            },
            marker: {
                size: 5
            }
        };
        
        const layout = {
            title: 'Convergência do Treinamento',
            xaxis: { title: 'Época' },
            yaxis: { title: 'Número de Erros' },
            hovermode: 'closest',
            showlegend: false,
            height: 400
        };
        
        Plotly.newPlot(elementId, [trace], layout);
    },

    /**
     * Plota a matriz de confusão
     */
    plotConfusionMatrix(yTrue, yPred, elementId = 'confusion-matrix') {
        // Calcular matriz de confusão
        let tp = 0, tn = 0, fp = 0, fn = 0;
        
        for (let i = 0; i < yTrue.length; i++) {
            if (yTrue[i] === 1 && yPred[i] === 1) tp++;
            else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
            else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
            else if (yTrue[i] === 1 && yPred[i] === 0) fn++;
        }
        
        const confusionData = {
            z: [[tn, fp], [fn, tp]],
            x: ['Pred: 0', 'Pred: 1'],
            y: ['Real: 0', 'Real: 1'],
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true,
            text: [[`TN: ${tn}`, `FP: ${fp}`], [`FN: ${fn}`, `TP: ${tp}`]],
            texttemplate: '%{text}',
            textfont: { size: 20, color: 'white' }
        };
        
        const layout = {
            title: 'Matriz de Confusão (Teste)',
            xaxis: { title: 'Predito', side: 'top' },
            yaxis: { title: 'Real', autorange: 'reversed' },
            width: 400,
            height: 400
        };
        
        Plotly.newPlot(elementId, [confusionData], layout);
        
        // Retornar métricas
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        
        return { tp, tn, fp, fn, precision, recall, f1 };
    },

    /**
     * Plota os pesos e bias
     */
    plotWeights(weights, bias, elementId = 'weights-visualization') {
        const data = [{
            type: 'bar',
            x: ['w₁', 'w₂', 'bias'],
            y: [weights[0], weights[1], bias],
            marker: {
                color: ['#667eea', '#764ba2', '#f76b8a']
            }
        }];
        
        const layout = {
            title: 'Pesos e Bias Aprendidos',
            yaxis: { title: 'Valor' },
            showlegend: false,
            height: 300
        };
        
        Plotly.newPlot(elementId, data, layout);
    },

    /**
     * Cria gráfico de dispersão 3D (para datasets com 3 features)
     */
    plot3DDataset(X, y, elementId) {
        const trace0 = {
            x: X.filter((_, i) => y[i] === 0).map(p => p[0]),
            y: X.filter((_, i) => y[i] === 0).map(p => p[1]),
            z: X.filter((_, i) => y[i] === 0).map(p => p[2]),
            mode: 'markers',
            marker: {
                color: 'red',
                size: 5,
            },
            type: 'scatter3d',
            name: 'Classe 0'
        };

        const trace1 = {
            x: X.filter((_, i) => y[i] === 1).map(p => p[0]),
            y: X.filter((_, i) => y[i] === 1).map(p => p[1]),
            z: X.filter((_, i) => y[i] === 1).map(p => p[2]),
            mode: 'markers',
            marker: {
                color: 'blue',
                size: 5,
            },
            type: 'scatter3d',
            name: 'Classe 1'
        };

        const layout = {
            title: 'Dataset 3D',
            scene: {
                xaxis: {title: 'Feature 1'},
                yaxis: {title: 'Feature 2'},
                zaxis: {title: 'Feature 3'},
            },
            height: 500
        };

        Plotly.newPlot(elementId, [trace0, trace1], layout);
    }
};

// Exportar para uso em outros módulos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Visualization;
}