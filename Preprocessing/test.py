

base_dir='F:\PII Research\Dataset\PII_Dataset\\'
dataset_dict={'12306':base_dir+'12306_PI_addusr_gid.csv',
              'ClixSense':base_dir+'ClixSense_username_email_password_firstname_lastname_birthday_gender_SSN_Zip_ascii（去掉口令中非ASCII; 2222045）.csv',
              'CSDN':base_dir+'csdn_PI_addusr_gid.csv',
              'dodo':base_dir+'dodonew_PI_addusr_gid.csv',
              'tianya':base_dir+'tianya_cleaned_username_email_password.txt',
              'yahoo_all':base_dir+'Yahoo_all.txt',
              'yahoo_cleaned':base_dir+'yahoo_cleaned_email_password.txt'
              }

def read_file(filename):
    PIIs=[]
    passwords=[]
    with open(filename,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if 'csv' in filename:
                temp = line.split(',')
            temp = line.split('\t')
            PIIs.append(temp[:])
            # passwords.append(temp[-1])
    return PIIs

for dataset in dataset_dict:
    PIIs=read_file(dataset_dict[dataset])
    print('--------',dataset,'-------------')
    for i in range(10):
        print(PIIs[i])
    print(len(PIIs))