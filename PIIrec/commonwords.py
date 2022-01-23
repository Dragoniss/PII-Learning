from os import read
from itertools import accumulate


def read_file(filename):
    # data=pd.read_csv(filename,header=None)
    passwords=[]
    with open(filename, "r") as f:
        for line in f.readlines():
            passwords.append(line)
    return passwords
 
def all_sub_string(a_string):
    a_string = a_string.strip('\n')
    if len(a_string) <= 3:
        return [a_string]
    else:
        return [a_string] + all_sub_string(a_string[1:])
 
 

if __name__=="__main__":
    pw = read_file("F:\\PII-Learning\\data\\clean_password.csv")
    #print(pw[:10])
    dic = dict(zip(all_sub_string(pw[0]),[1]*len(all_sub_string(pw[0]))))
    for i in range(1,len(pw)):
        temp_list = all_sub_string(pw[i])
        #print(temp_list)
        for item in temp_list:
            if item in dic:
                dic[item] += 1
            else:
                dic[item] = 1
        #print(dic)
    dic = dict(sorted(dic.items(), key = lambda kv:(len(kv[0]),kv[1]),reverse=True))
    with open("F://comm.txt", "w") as f:
        [f.write('{0},{1}\n'.format(key,value)) for key,value in dic.items()]