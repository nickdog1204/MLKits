require("@tensorflow/tfjs-node")
const tf = require("@tensorflow/tfjs")
const loadCSV = require("../load-csv")
const LinearRegression = require('./linear-regression')
const plot = require('node-remote-plot')

let {features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
})

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    maxIterations: 3,
    batchSize: 10
    // batchSize: 1 => sgd
})
regression.train()
// console.log("Updated M:", regression.weights.dataSync()[1], "Updated b", regression.weights.dataSync()[0])
const r2 = regression.test(testFeatures, testLabels)

// console.log("MSE History", regression.mseHistory)
plot({
    x: regression.mseHistory.reverse(),
    xLabel: "Iteration #",
    yLabel: "Mean Squared Error"
})
// plot({
//     x: regression.bHistory,
//     y: regression.mseHistory.reverse(),
//     xLabel: "Value of b",
//     yLabel: "Mean Squared Error"
// })
console.log("R2:", r2)
regression.predict([[120, 2, 380]]).print()

