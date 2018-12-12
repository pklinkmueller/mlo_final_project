import datasets.data as data

def data_sms_spam():
    text, labels = data.load_sms_spam()
    print(text.shape)
    print(labels.shape)