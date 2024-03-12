from tqdm import tqdm
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

K = 5
tot_fqdn = 56  ## item - 1
feature_all = []
label_all = [0] * tot_fqdn
fea_path = './feature.csv'
ans_path = './result.csv'

cnt = 0
df = pd.read_csv(fea_path)
for index,row in tqdm(df.iterrows()):
    label_all[cnt] = row[0]
    feature_all.append(row[1:])
    cnt += 1

train_feature = [] 
train_label = []
test_feature = [] 
test_label = []    

train_num = int(len(label_all) * 0.7)
test_num = len(label_all) - train_num
for i in range(tot_fqdn):
    if i < train_num:
        train_feature.append(feature_all[i])
        train_label.append(label_all[i]) 
    else:
        test_feature.append(feature_all[i])
        test_label.append(label_all[i])

# 创建分类器
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(train_feature,train_label)

correct = 0.0
with open(ans_path, "w") as outputfile:
    for i in tqdm(range(test_num)):
        ans = clf.predict([test_feature[i]])[0]
        if test_label[i] == ans:
            correct = correct + 1
        outputfile.write(str(i) + ": " + str(int(test_label[i])) + " -> " + str(int(ans)) + "\n")
print(correct / test_num * 100)