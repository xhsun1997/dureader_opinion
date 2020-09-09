import json
import os
import collections
import jieba
import tqdm

def analysis_answer_feature(data_file,answer_type,thesold=2000):
    with open(data_file) as f:
        lines=f.readlines()
    feature_words=[]
    for line in lines:
        one_example=json.loads(line)
        if one_example["yesno_answer"]==answer_type:
            answer=one_example["answer"]
            feature_words.extend(list(jieba.cut(answer)))
    sorted_feature_words=sorted(collections.Counter(feature_words).items(),key=lambda x:x[1],reverse=True)
    result_feature_words=[word for word,word_freq in sorted_feature_words if word_freq>thesold]
    result_feature_words=[word for word in result_feature_words if word not in ["。","，","、","！"," "]]
    return result_feature_words
# yes_feature_words=analysis_answer_feature("train.json","Yes")
# no_feature_words=analysis_answer_feature("train.json","No")


def change_json_to_tsv(json_files,tsv_file):
    f_write=open(tsv_file,"w")
    f_write.write("label"+"\t"+"text_a"+"\t"+"text_b"+"\n")
    for json_file in json_files:
        f_read=open(json_file)
        lines=f_read.readlines()
        f_read.close()
        for line in lines:
            one_example=json.loads(line)
            text_a=one_example["question"]
            text_b=one_example["answer"]
            label=one_example["yesno_answer"]
            if label=="Yes":
                label=0
            elif label=="No":
                label=1
            else:
                assert label=="Depends"
                label=2
            f_write.write(str(label)+"\t"+text_a+"\t"+text_b+"\n")
    f_write.close()

#change_json_to_tsv(["more_train.json","more_dev.json"],"ver_large.tsv")

def cleaned_title(title):
    if title.find('-')!=-1:
        return title.strip().split('-')[0]
    elif title.find('_')!=-1:
        return title.strip().split("_")[0]
    return title

def is_worth_to_add(each_sentence,answer):
    if abs(len(each_sentence)-len(answer))<=3 and each_sentence.find(answer)!=-1:
        return False
    if each_sentence.find(answer)!=-1:
        return True
    each_sentence_list=list(jieba.cut(each_sentence))
    if len(each_sentence_list)<=5:
        return False
    answer_list=list(jieba.cut(answer))
    common_word=[word for word in each_sentence if word in answer_list]
    if len(common_word)>=int(len(each_sentence_list)/2):
        return True
    return False
    

def get_more_data(orig_data_file,output_data_file):
    with open(orig_data_file) as f:
        lines=f.readlines()
    f_write=open(output_data_file,"w",encoding="utf-8")
    new_id_start=100000
    for i in tqdm.tqdm(range(len(lines))):
        line=lines[i]
        one_example=json.loads(line)
        #one_example.keys()=={"question","answer","yesno_answer","documents"}
        documents=one_example["documents"]
        answer=one_example["answer"]
        question=one_example["question"]
        yesno_answer=one_example["yesno_answer"]
        qid=one_example["id"]
        one_new_example={"id":qid,"answer":answer,"question":question,"yesno_answer":yesno_answer}
        f_write.write(json.dumps(one_new_example,ensure_ascii=False)+"\n")

        for each_document in documents:
            #each_document.keys()=={"title","paragraphs"}
            title=each_document["title"]
            title=cleaned_title(title)
            if len(title)<=5:
                continue
            one_new_example["id"]="null"
            one_new_example["question"]=title
            f_write.write(json.dumps(one_new_example,ensure_ascii=False)+"\n")
            paragraphs=each_document["paragraphs"]
            for each_sentence in paragraphs:
                if is_worth_to_add(each_sentence,answer):
                    one_new_example["id"]="null"
                    one_new_example["answer"]=each_sentence
                    one_new_example["question"]=question
                    f_write.write(json.dumps(one_new_example,ensure_ascii=False)+"\n")
    f_write.close()


                
            
import json
with open("palm_practice/predictions.json") as f:
    lines=f.readlines()
predict_data=[]
for line in lines:
    line=json.loads(line)
    predict_data.append(line)
print(len(predict_data))

with open("test.json") as f:
    lines=f.readlines()
assert len(lines)==len(predict_data)==9253
f_w=open("predict.json","w")
for i,line in enumerate(lines):
    line=json.loads(line)
    yesno_answer=predict_data[i]["label"]
    if yesno_answer==0:
        yesno_answer="Yes"
    elif yesno_answer==1:
        yesno_answer="No"
    else:
        assert yesno_answer==2
        yesno_answer="Depends"
    one_example={"id":line["id"],"yesno_answer":yesno_answer}
    f_w.write(json.dumps(one_example,ensure_ascii=False)+"\n")






import json
import os
import numpy as np
def get_test_(all_json_files,test_json_file):
    all_answer_id=[]
    all_answer_value=[]
    path_root=all_json_files
    f_write=open(test_json_file,"w")
    test_json_data=[]
    all_json_files=os.listdir(all_json_files)
    with open("/home/xhsun/Desktop/test.json") as f:
        lines=f.readlines()
        for line in lines:
            test_json_data.append(json.loads(line))
    print("test json length ",len(test_json_data))
    print(all_json_files)
    num_files=len(all_json_files)
    for json_file in all_json_files:
        answer_id=[]
        answer_value=[]
        json_file=open(os.path.join(path_root,json_file))
        for line in json_file.readlines():
            line=json.loads(line)
            answer_id.append(line["id"])
            if line["yesno_answer"]=="Yes":
                answer_value.append(0)
            elif line["yesno_answer"]=="No":
                answer_value.append(1)
            else:
                assert line["yesno_answer"]=="Depends"
                answer_value.append(2)
        all_answer_value.append(answer_value)
        all_answer_id.append(answer_id)
        json_file.close()
        
    all_answer_value_arr=np.array(all_answer_value)
    all_answer_id=all_answer_id[0]
    print(all_answer_value_arr.shape,len(all_answer_id))#(8,9253)
    all_answer_value_arr=np.transpose(all_answer_value_arr,axes=[1,0])#(9253,8)
    print(all_answer_value_arr.shape)
    num_=0
    for i,each_vector in enumerate(all_answer_value_arr):
        id_=test_json_data[i]["id"]
        answer=test_json_data[i]["answer"]
        question=test_json_data[i]["question"]
        if sum(each_vector)==0:
            yesno_answer="Yes"
            content={"id":id_,"answer":answer,"question":question,"yesno_answer":yesno_answer}
            f_write.write(json.dumps(content,ensure_ascii=False)+"\n")
            num_+=1
        if sum(each_vector)==num_files:
            yesno_answer="No"
            content={"id":id_,"answer":answer,"question":question,"yesno_answer":yesno_answer}
            num_+=1
            f_write.write(json.dumps(content,ensure_ascii=False)+"\n")
        if sum(each_vector)==num_files*2:
            yesno_answer="Depends"
            content={"id":id_,"answer":answer,"question":question,"yesno_answer":yesno_answer}
            f_write.write(json.dumps(content,ensure_ascii=False)+"\n")
            num_+1
    print(num_)