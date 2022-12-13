import numpy as np
from data2feature import AAC, PseAAC, Am_PseAAC

def generate_features(path, dataset):

    with open(path,"r") as f:
        lines = f.readlines()
        y_label = []
        seq = []
        for i, line in enumerate(lines):
            if i % 2 ==0:    
                if line.__contains__("neg"):
                    y_label.append(0)
                else:
                    y_label.append(1)
            else:
                line = line[:-1]
                seq.append(line)
        y_label = np.array(y_label)    # (160,) 
        if dataset == "NT15":
            AAC_array = np.array(AAC(seq)) # (160,20)
            PseAAC_array = np.array(PseAAC(seq, AAC_array, lamda = 2, weight = 0.1))
            Am_PseAAC_array = np.array(Am_PseAAC(seq, AAC_array, lamda = 3, weight = 0.2))
        else:
            AAC_array = np.array(AAC(seq)) # (160,20)
            PseAAC_array = np.array(PseAAC(seq, AAC_array, lamda = 1, weight = 0.9))
            Am_PseAAC_array = np.array(Am_PseAAC(seq, AAC_array, lamda = 1, weight = 0.9))

    return y_label, AAC_array, PseAAC_array, Am_PseAAC_array

if __name__ == "__main__":

    path = "./dataset/NT15dataset_train.fasta"
    NT15_train_y, NT15_train_AAC, NT15_train_PAAC, NT15_train_APAAC = generate_features(path, "NT15")
    print(NT15_train_y.shape)
    print(NT15_train_AAC)
    print(NT15_train_PAAC)
    print(NT15_train_APAAC)
