from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

#initilizing of embedding & recognizer
embeddingFile = "C:/Users/nanim/Downloads/attendence Recognition-20240714T145610Z-001/attendence Recognition/code/output/embeddings.pickle"
#New & Empty at initial
recognizerFile = "C:/Users/nanim/Downloads/attendence Recognition-20240714T145610Z-001/attendence Recognition/code/output/recognizer.pickle"
labelEncFile = "C:/Users/nanim/Downloads/attendence Recognition-20240714T145610Z-001/attendence Recognition/code/output/le.pickle"

print("Loading face embeddings...")
data = pickle.loads(open(embeddingFile, "rb").read())

print("Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])


print("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open(recognizerFile, "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(labelEncFile, "wb")
f.write(pickle.dumps(labelEnc))
f.close()
