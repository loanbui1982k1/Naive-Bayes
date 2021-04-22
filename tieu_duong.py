import csv
import random
import math
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load data tu CSV file
def load_data(filename):
    lines = csv.reader(open(filename), delimiter=',')
    dataset = list()
    for row in lines:
        temp = list()
        for i in range(len(row)):
            temp.append((float) (row[i]))
        dataset.append(temp)
    return dataset

# Phan chia tap du lieu theo class
def separate_data(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

# Phan chia tap du lieu thanh training va testing
def split_data(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def probLabel(trainSet, classValue):
    count = 0
    for x in trainSet:
        if x[-1] == classValue:
            count += 1
    if count > 0:
        return count / len(trainSet)
    else:
        return 1 / (len(trainSet) + 1)            

# tinh toan gia tri trung binh cua moi thuoc tinh
def mean(numbers):
    
    return sum(numbers) / float(len(numbers))

# Tinh toan do lech chuan cho tung thuoc tinh
def standard_deviation(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)

    return math.sqrt(variance)

# list (Gia tri trung binh , do lech chuan)
def summarize(dataset):
    attributes = list(zip(*dataset))
    summaries = []
    for i in range(len(attributes)):   
        summaries.append((mean(attributes[i]), standard_deviation(attributes[i])))
    del summaries[-1]
    return summaries


def summarize_by_class(dataset):
    separated = separate_data(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    #print(summaries)
    return summaries

# Tinh toan xac suat theo phan phoi Gaussian cua bien lien tuc
def calculate_prob(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))

    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# Tinh xac suat cho moi thuoc tinh phan chia theo class
def calculate_class_prob(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = prob_label[classValue]
        #probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculate_prob(x, mean, stdev)
            #probabilities[classValue] += math.log(calculate_prob(x, mean, stdev))
      
    return probabilities

# Du doan vector thuoc phan lop nao
def predict(summaries, inputVector):
    probabilities = calculate_class_prob(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

# Du doan tap du lieu testing thuoc vao phan lop nao
def get_predictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)

    return predictions

# Tinh toan do chinh xac cua phan lop
def get_accuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def get_data_label(dataset):
    data = []
    label = []
    for x in dataset:
        data.append(x[:len(x) - 1])
        label.append(x[-1])

    return data, label

def main():
    filename = 'tieu_duong.csv'
    splitRatio = 0.8
    dataset = load_data(filename= 'tieu_duong.csv')
    trainingSet, testSet = split_data(dataset, splitRatio)

    filein = open('input.txt')
    testInput = filein.read()
    inputVector = testInput.split(',')
    for i in range(len(inputVector)):
        inputVector[i] = float (inputVector[i])
    fileout = open('output.txt', 'w')
    fileout.write('Data size {} \nTraining Size={} \nTest Size={}\n'.format(len(dataset), len(trainingSet), len(testSet)))
   
    global prob_label
    prob_label = {}
    for i in separate_data(dataset).keys():
        prob_label[i] = probLabel(trainingSet, i)
    
    
    # prepare model
    summaries = summarize_by_class(trainingSet)
    get_data_label(trainingSet)
    predictInput = predict(summaries, inputVector)
    print(predictInput)
    #fileout.write('Chuẩn đoán: {} \n'.format(predictInput))
    if predictInput == 0:
        fileout.write('Chuan doan: Khong mac benh\n')
    else: 
        fileout.write('Chuan doan: Mac benh\n')
    print(calculate_class_prob(summaries, inputVector))

    
    # test model
    predictions = get_predictions(summaries, testSet)
    accuracy = get_accuracy(testSet, predictions)
    fileout.write('Accuracy of my implement: {}%\n'.format(accuracy))

    # Compare with sklearn
    dataTrain, labelTrain = get_data_label(trainingSet)
    dataTest, labelTest = get_data_label(testSet)

    #from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(dataTrain, labelTrain)
    score = clf.score(dataTest, labelTest)

    fileout.write('Accuracy of sklearn: {}%'.format(score*100))
    fileout.close()
    filein.close()

if __name__ == "__main__":
    main()