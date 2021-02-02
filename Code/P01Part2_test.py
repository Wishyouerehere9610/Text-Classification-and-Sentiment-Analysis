import os
import matplotlib.pyplot as plt
from sentimentAnalysis import sentimentAnalysis
posFolder = '../Data/SA/pos';
negFolder = '../Data/SA/neg';

totalNeg = 0
totalPos = 0
correctNeg = 0;
correctPos = 0

posfiles = []
negfiles = []
posscores = []
negscores = []

for root, dirs, files in os.walk(negFolder):
    for f in files:
        negfiles.append(f)
        score = sentimentAnalysis(negFolder+"/"+f)
        negscores.append(score)
        print('Groundtruth: Negtive, sentiment score: %8.2f' % score)
        if(score < 0):
            correctNeg += 1
        totalNeg += 1



for root, dirs, files in os.walk(posFolder):
    for f in files:
        posfiles.append(f)
        score = sentimentAnalysis(posFolder+"/"+f)
        posscores.append(score)
        print('Groundtruth: Positive, sentiment score: %8.2f' % score)
        if (score > 0):
            correctPos += 1
        totalPos += 1

# plot pos and neg graphs
# x axis for file name, y for score
plt.figure(figsize=(20,10))
plt.title("Positive Scores")
plt.xlabel("File names")
plt.ylabel("Score")
plt.bar(posfiles,posscores)
plt.show()

plt.figure(figsize=(20,10))
plt.title("Negative Scores")
plt.xlabel("File names")
plt.ylabel("Score")
plt.bar(negfiles,negscores)
plt.show()

print("Total Positive Number: "+str(totalPos))
print("Accuracy in positive set: %.2f" % (correctPos/totalPos))
print("Total Negtive Number: "+str(totalNeg))
print("Accuracy in Negtive set: %.2f" % (correctNeg/totalNeg))
