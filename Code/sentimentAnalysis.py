import string

# Preprocess the review
# Iterate the review and calculate the score
def sentimentAnalysis(filename):
    # Read wordWithStrength.txt and store them in hashtable
    dict = {}
    for  line in open('../Data/SA/wordWithStrength.txt','r',encoding='utf-8'):
        rs = line.replace('\n', '')
        items = rs.split("\t")
        dict[items[0]] = float(items[1])
    # print(dict)

    # Read a file named "filename"
    ## Merge the lines
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    sentence = ""
    for line in open(filename, 'r',encoding='utf-8'):
        rs = line.replace('\n', '')
        sentence += " " + rs

    ## Remove punctuations
    result = ""
    for char in sentence:
        if char not in punctuations:
            result = result + char
    # print(result)
    ## Separate the sentence into words
    words = result.split(" ");
    # print(words)

    ## Calculate the score
    score = 0
    for word in words:
        if(word!=''):
            if(word in dict.keys()):
                score += dict[word]
                print(word+" "+ str(dict[word]))



    # feel free to format output as you see fit
    if(score > 0):
        if(score > 0.7):
            print('File %s \n Sentiment Score: %8.2f Highly Positive Sentiment\n' % (filename, score))
        else:
            print('File %s \n Sentiment Score: %8.2f Positive Sentiment\n' % (filename, score))
    if (score < 0):
        if(score < -0.7):
            print('File %s \n Sentiment Score: %8.2f Highly Negative Sentiment\n' % (filename, score))
        else:
            print('File %s \n Sentiment Score: %8.2f Negative Sentiment\n' % (filename, score))
    if (score == 0):
        print('File %s \n Sentiment Score: %8.2f Neutral Sentiment\n' % (filename, score))

    return score

# sentimentAnalysis("../Data/SA/neg/10.txt")