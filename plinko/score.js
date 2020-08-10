const outputs = []

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
    // Ran every time a ball drops into a bucket
    outputs.push([dropPosition, bounciness, size, bucketLabel])
}

function runAnalysis() {
    // Write code here to analyze stuff
    const testSetSize = 100
    const [testSet, trainingSet] = splitDataset(minMax(outputs, 3), testSetSize)
    _.range(1, 20).forEach(k => {
        const accuracy = _.chain(testSet)
            .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === testPoint[3])
            .size()
            .divide(testSetSize)
            .value()
        console.log('For k of', k, ', accuracy is:', accuracy)
    })

    // let numberCorrect = 0
    // for (let i = 0; i < testSet.length; i++) {
    //     const bucket = knn(trainingSet, testSet[i][0])
    //     if (bucket == testSet[i][3]) {
    //         numberCorrect++
    //     }
    // }
    // console.log('Accuracy: ', numberCorrect / testSetSize)

}

function splitDataset(data, testSetSize) {
    const shuffled = _.shuffle(data)
    const testSet = _.slice(shuffled, 0, testSetSize)
    const trainingSet = _.slice(shuffled, testSetSize)
    return [testSet, trainingSet]
}

function knn(trainingSet, point, k) {
    // point has 3 values!!!!
    return _.chain(trainingSet)
        .map(row => {
            return [
                distance(_.initial(row), point),
                _.last(row)
            ]
        })
        .sortBy(row => row[0])
        .slice(0, k)
        .countBy(row => row[1])
        .toPairs()
        .sortBy(pair => pair[1])
        .last()
        .first()
        .parseInt()
        .value()
}

function distance(pointA, pointB) {
    return _.chain(pointA)
        .zip(pointB)
        .map(([a, b]) => (a - b) ** 2)
        .sum()
        .value() ** 0.5
    // return Math.abs(pointA - pointB)
}

function minMax(data, featureCount) {
    const clonedData = _.cloneDeep(data)
    for (let col = 0; col < featureCount; col++) {
        const featureColumnArray = clonedData.map(row => row[col])
        const min = _.min(featureColumnArray)
        const max = _.max(featureColumnArray)
        const fullRange = max - min

        for (let row = 0; row < clonedData.length; row++) {
            clonedData[row][col] = (clonedData[row][col] - min) / fullRange
        }
    }
    return clonedData
}

