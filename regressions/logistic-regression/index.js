require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const LogisticRegression = require('./logistic-regression')
const plot = require('node-remote-plot')

const {features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
        dataColumns: [
            'horsepower',
            'displacement',
            'weight'
        ],
        labelColumns: [
            'passedemissions'
        ],
        shuffle: true,
        splitTest: 50,
        converters: {
            passedemissions: (value) => value == 'TRUE' ? 1 : 0
        }
    }
)

const logisticRegression = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    maxIterations: 100,
    batchSize: 50,
})
logisticRegression.train()
// logisticRegression.predict([
//     [130, 307, 1.75],
//     [88, 97, 1.065]
// ]).print()
console.log(logisticRegression.test(testFeatures, testLabels))

plot({
    x: logisticRegression.costHistory.reverse()
})
