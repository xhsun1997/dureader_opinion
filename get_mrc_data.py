import json

def cleaned_title(title):
    if title.find('-')!=-1:
        return title.strip().split('-')[0]
    elif title.find('_')!=-1:
        return title.strip().split("_")[0]
    return title

def get_mrc_data_(datasets,mrc_data_file):
    f_write=open(mrc_data_file,"w")
    nums=0
    for dataset in datasets:
        f_read=open(dataset)
        for line in f_read.readlines():
            one_example=json.loads(line)#{answer,documents,yesno_answer,question,"id"}
            question=one_example["question"].strip()
            answer_text=one_example["answer"].strip()
            id_=one_example["id"]
            for document in one_example["documents"]:
                #keys()=={"paragraphs","title"}
                paragraphs=document["paragraphs"]
                title=cleaned_title(document["title"])
                for sentence in paragraphs:
                    answer_start=sentence.find(answer_text)
                    if answer_start==-1:
                        continue
                    answer={"text":answer_text,"answer_start":answer_start,"yesno_answer":one_example["yesno_answer"]}
                    new_example={"question":question,"answer":answer,"id":id_,"context":sentence}
                    f_write.write(json.dumps(new_example,ensure_ascii=False)+"\n")
                    nums+=1
    print(nums)