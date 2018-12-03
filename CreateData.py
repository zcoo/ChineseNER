

with open('mydata/train.txt', encoding='utf-8') as f:
    lines = f.readlines()
word_list = []
format_word_list = []
for line in lines:
    words = line.split(' ')
    word_list += words
# 首先将词和标注分开
for word in word_list:
    realword, wordtype = word.split('/')
    for oneword in realword:
        format_word_list.append([oneword, wordtype.strip('\n')])

real_format_word_list = []
ORGANIZATION = ['B-ORGANIZATION', 'I-ORGANIZATION', 'O-ORGANIZATION']
TIME = ['B-TIME', 'I-TIME', 'O-TIME']
PERSON = ['B-PERSON', 'I-PERSON', 'O-PERSON']
LOCATION = ['B-LOCATION', 'I-LOCATION', 'O-LOCATION']
B = [ORGANIZATION[0], TIME[0], PERSON[0], LOCATION[0]]
newB = ['B-ORG', 'B-TIM', 'B-PER', 'B-LOC']
newI = ['I-ORG', 'I-TIM', 'I-PER', 'I-LOC']
for word in format_word_list:
    if word[1] == 'O':
        real_format_word_list.append(word[0] + '	O')
        continue
    index = -1
    for i in range(0, 4):
        if word[1] == B[i]:
            index = i
            break

    if index != -1:
        # 是B开头，如果是某个实体的第一个字，上一个字必须不是该实体的B开头
        if real_format_word_list[-1].split('	')[1] != newB[index]:
            real_format_word_list.append(word[0]+'	'+newB[index])
        else:
            real_format_word_list.append(word[0] + '	' + newI[index])
    else:
        real_format_word_list.append(word[0] + '	I' + word[1][1:5])
with open("mydata/train",'wb') as f:
    for tmp in real_format_word_list:
        tmpstr=tmp+'\n'
        zzz=tmpstr.encode("utf-8")
        f.write(zzz)
        if tmp[0]=='。':
            f.write('\n'.encode("utf-8"))