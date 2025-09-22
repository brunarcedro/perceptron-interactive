/**
 * datasets.js
 * Geradores de datasets sintéticos e clássicos para teste do Perceptron
 */

const DatasetGenerator = {
    /**
     * Gera dataset de blobs sintéticos (linearmente separável)
     * @param {number} samples - Número de amostras
     * @param {number} clusterStd - Desvio padrão dos clusters
     * @returns {Object} {X: features, y: labels}
     */
    generateBlobs(samples = 200, clusterStd = 1.5) {
        const X = [];
        const y = [];
        const halfSamples = Math.floor(samples / 2);
        
        // Classe 0: centro em (-3, -3)
        for (let i = 0; i < halfSamples; i++) {
            X.push([
                -3 + this.randn() * clusterStd,
                -3 + this.randn() * clusterStd
            ]);
            y.push(0);
        }
        
        // Classe 1: centro em (3, 3)
        for (let i = 0; i < halfSamples; i++) {
            X.push([
                3 + this.randn() * clusterStd,
                3 + this.randn() * clusterStd
            ]);
            y.push(1);
        }
        
        return { X, y, info: {
            name: 'Blobs Sintéticos',
            description: 'Dataset linearmente separável com dois clusters gaussianos.',
            linearSeparable: true
        }};
    },

    /**
     * Gera dataset Iris simplificado (Setosa vs Versicolor)
     * @param {number} samples - Número de amostras
     * @returns {Object} {X: features, y: labels}
     */
    generateIris(samples = 100) {
        const X = [];
        const y = [];
        const halfSamples = Math.floor(samples / 2);
        
        // Setosa (classe 0)
        for (let i = 0; i < halfSamples; i++) {
            X.push([
                4.5 + Math.random() * 1.5,  // Comprimento da sépala
                1.0 + Math.random() * 0.8   // Comprimento da pétala
            ]);
            y.push(0);
        }
        
        // Versicolor (classe 1)
        for (let i = 0; i < halfSamples; i++) {
            X.push([
                5.5 + Math.random() * 2,    // Comprimento da sépala
                3.5 + Math.random() * 1.5   // Comprimento da pétala
            ]);
            y.push(1);
        }
        
        return { X, y, info: {
            name: 'Iris (Setosa vs Versicolor)',
            description: 'Classificação de duas espécies de flores Iris. Estas classes são linearmente separáveis.',
            linearSeparable: true
        }};
    },

    /**
     * Gera dataset Moons (não linearmente separável)
     * @param {number} samples - Número de amostras
     * @param {number} noise - Nível de ruído
     * @returns {Object} {X: features, y: labels}
     */
    generateMoons(samples = 200, noise = 0.15) {
        const X = [];
        const y = [];
        const halfSamples = Math.floor(samples / 2);
        
        for (let i = 0; i < samples; i++) {
            const angle = Math.PI * i / halfSamples;
            if (i < halfSamples) {
                // Primeira lua
                X.push([
                    Math.cos(angle) + (Math.random() - 0.5) * noise,
                    Math.sin(angle) + (Math.random() - 0.5) * noise
                ]);
                y.push(0);
            } else {
                // Segunda lua
                X.push([
                    1 - Math.cos(angle) + (Math.random() - 0.5) * noise,
                    0.5 - Math.sin(angle) + (Math.random() - 0.5) * noise
                ]);
                y.push(1);
            }
        }
        
        return { X, y, info: {
            name: 'Moons Dataset',
            description: 'Dataset com formato de duas luas entrelaçadas. NÃO é linearmente separável - demonstra as limitações do Perceptron.',
            linearSeparable: false
        }};
    },

    /**
     * Gera dataset de câncer de mama simplificado
     * @param {number} samples - Número de amostras
     * @returns {Object} {X: features, y: labels}
     */
    generateCancer(samples = 200) {
        const X = [];
        const y = [];
        const halfSamples = Math.floor(samples / 2);
        
        // Benigno (classe 0)
        for (let i = 0; i < halfSamples; i++) {
            X.push([
                10 + Math.random() * 5,     // Raio médio
                0.05 + Math.random() * 0.15 // Textura média
            ]);
            y.push(0);
        }
        
        // Maligno (classe 1)
        for (let i = 0; i < halfSamples; i++) {
            X.push([
                18 + Math.random() * 7,     // Raio médio
                0.15 + Math.random() * 0.2  // Textura média
            ]);
            y.push(1);
        }
        
        return { X, y, info: {
            name: 'Breast Cancer Wisconsin',
            description: 'Classificação de tumores em benignos ou malignos. Problema médico real simplificado para visualização.',
            linearSeparable: true
        }};
    },

    /**
     * Gera dataset com ruído controlável
     * @param {number} samples - Número de amostras
     * @param {number} separation - Separação entre classes
     * @param {number} flipProb - Probabilidade de inverter labels
     * @returns {Object} {X: features, y: labels}
     */
    generateNoisyClassification(samples = 200, separation = 1.5, flipProb = 0.05) {
        const X = [];
        const y = [];
        const halfSamples = Math.floor(samples / 2);
        
        // Classe 0
        for (let i = 0; i < halfSamples; i++) {
            X.push([
                Math.random() * 4 - 2 - separation,
                Math.random() * 4 - 2
            ]);
            y.push(0);
        }
        
        // Classe 1
        for (let i = 0; i < halfSamples; i++) {
            X.push([
                Math.random() * 4 - 2 + separation,
                Math.random() * 4 - 2
            ]);
            y.push(1);
        }
        
        // Adicionar ruído aos rótulos
        for (let i = 0; i < y.length; i++) {
            if (Math.random() < flipProb) {
                y[i] = 1 - y[i];
            }
        }
        
        return { X, y, info: {
            name: 'Dataset com Ruído',
            description: 'Dataset com sobreposição controlada entre classes. Útil para testar robustez a ruído.',
            linearSeparable: separation > 1
        }};
    },

    /**
     * Gera dataset personalizado
     * @param {number} samples - Número de amostras
     * @param {Array} center0 - Centro da classe 0
     * @param {Array} center1 - Centro da classe 1
     * @param {number} spread - Espalhamento dos pontos
     * @returns {Object} {X: features, y: labels}
     */
    generateCustom(samples = 100, center0 = [-2, -2], center1 = [2, 2], spread = 1) {
        const X = [];
        const y = [];
        const halfSamples = Math.floor(samples / 2);
        
        // Classe 0
        for (let i = 0; i < halfSamples; i++) {
            X.push([
                center0[0] + (Math.random() - 0.5) * 2 * spread,
                center0[1] + (Math.random() - 0.5) * 2 * spread
            ]);
            y.push(0);
        }
        
        // Classe 1
        for (let i = 0; i < halfSamples; i++) {
            X.push([
                center1[0] + (Math.random() - 0.5) * 2 * spread,
                center1[1] + (Math.random() - 0.5) * 2 * spread
            ]);
            y.push(1);
        }
        
        return { X, y, info: {
            name: 'Dataset Personalizado',
            description: 'Dois grupos bem separados. Permite explorar a geometria da solução do Perceptron.',
            linearSeparable: true
        }};
    },

    /**
     * Gera número aleatório com distribuição normal (Box-Muller)
     * @returns {number} Número aleatório com média 0 e desvio 1
     */
    randn() {
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
};

// Exportar para uso em outros módulos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DatasetGenerator;
}