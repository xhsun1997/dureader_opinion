import json
import os
import collections
import jieba

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


def change_json_to_tsv(json_file,tsv_file):
    f_read=open(json_file)
    f_write=open(tsv_file,"w")
    f_write.write("label"+"\t"+"text_a"+"text_b"+"\n")
    lines=f_read.readlines()
    f_read.close()
    for line in lines:
        one_example=json.loads(line)
        text_a=one_example["question"]
        text_b=one_example["answer"]
        label=one_example["yesno_answer"]
        f_write.write(label+"\t"+text_a+"\t"+text_b+"\n")
    f_write.close()

def get_more_data(orig_data_file,output_data_file):
    with open(orig_data_file) as f:
        lines=f.readlines()
    f_write=open(output_data_file,"w",encoding="utf-8")
    for line in lines:
        one_example=json.loads(line)
        #one_example.keys()=={"question","answer","yesno_answer","documents"}
        documents=one_example["documents"]
        answer=one_example["answer"]
        question=one_example["question"]
        yesno_answer=one_example["yesno_answer"]
        qid=one_example["id"]
        if yesno_answer=="Depends":
            one_new_example={"id":qid,"answer":answer,"question":question,"yesno_answer":yesno_answer}
            f_write.write(json.dumps(one_new_example)+"\n")
            continue
        for each_document in documents:
            #each_docuemnt.keys()=={"title","paragraphs"}
            paragraphs=each_document["paragraphs"]
            for each_sentence in paragraphs:
                
            
        