from config import base_dir,dataset_dict

def read_file(filename):
    PIIs=[]
    passwords=[]
    with open(filename,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if 'csv' in filename:
                temp = line.split(',')
            temp = line.split('\t')

            email=temp[0]
            name=temp[2][:-1]
            username=temp[4]
            password=temp[1]
            if ',./' in password:
                print(password)
            phone=temp[5]
            birth=temp[6][:-1]
            PII=[username,email,name,birth,phone]
            PIIs.append(PII)
            passwords.append(password)

    return PIIs,passwords

def write_file(filename,PIIs,passwords):
    with open(filename, "w") as f:
        for i in range(len(PIIs)):
            string=""
            for item in PIIs[i]:
                string+=item+" "
            string+=passwords[i]
            f.write(string)
def getPasswordRange(passwords):
    for i in range(len(passwords)):
        if len(passwords[i])>max:
            max=len(passwords[i])
        if len(passwords[i])<min:
            min=len(passwords[i])
    return min,max
if __name__=="__main__":
    PIIs,passwords=read_file(dataset_dict['CSDN'])
    min=6
    max=16
    cleaned_PII=[]
    cleaned_password=[]
    for i in range(len(passwords)):
        if PIIs[i][0]=='':
            continue
        PIIs[i][0]=PIIs[i][0].replace(' ','')

        if len(passwords[i])>=min and len(passwords[i])<=max:
            if len(PIIs[i][3])==8 and len(PIIs[i][4])==11:
                passwords[i]=passwords[i].replace(' ','')
                cleaned_password.append(passwords[i]+'\n')
                cleaned_PII.append(PIIs[i])


    print(max,min)
    print(len(passwords),len(cleaned_password))
    write_file(base_dir+"\clean\CSDN_cleaned.txt",cleaned_PII,cleaned_password)
