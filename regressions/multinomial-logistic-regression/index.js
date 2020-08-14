require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const LogisticRegression = require('./logistic-regression')
const _ = require('lodash')
const plot = require('node-remote-plot')

const {features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
        dataColumns: [
            'horsepower',
            'displacement',
            'weight'
        ],
        labelColumns: [
            'mpg'
        ],
        shuffle: true,
        splitTest: 50,
        converters: {
            mpg: (value) => {
                const mpg = parseFloat(value)
                if (mpg < 15) {
                    return [1, 0, 0]
                } else if (mpg < 30) {
                    return [0, 1, 0]
                } else {
                    return [0, 0, 1]
                }
            }
        }
    }
)

const logisticRegression = new LogisticRegression(features, _.flatMap(labels), {
    learningRate: 0.5,
    maxIterations: 100,
    batchSize: 10,
})
logisticRegression.weights.print()

logisticRegression.train()
logisticRegression.predict([
    [150, 200, 2.22223],
]).print()

console.log(logisticRegression.test(testFeatures, _.flatMap(testLabels)))
// console.log(logisticRegression.test(testFeatures, testLabels))
//
// plot({
//     x: logisticRegression.costHistory.reverse()
// })
