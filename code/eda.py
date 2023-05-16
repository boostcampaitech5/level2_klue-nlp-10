import random
import pandas as pd
import pickle
import re
import sys

def update_idx(dataset):
    update_sub_idx, update_obj_idx= [], []
    for sen,sub_idx,obj_idx in zip(dataset['sentence'],dataset['subject_idx'],dataset['object_idx']):
     
        # TEM(Typed Entity Marker) 후 subject_idx 재설정
        sub_start_idx = sen.find('@')
        sub_end_idx = sub_start_idx+(sub_idx[1]-sub_idx[0]+1)+6
        new_sub_i = [sub_start_idx, sub_end_idx]
        update_sub_idx.append(new_sub_i)
            
        # TEM(Typed Entity Marker) 후 object_idx 재설정    
        obj_start_idx = sen.find('#')
        obj_end_idx = obj_start_idx+(obj_idx[1]-obj_idx[0]+1)+6
        new_obj_i = [obj_start_idx, obj_end_idx]
        update_obj_idx.append(new_obj_i)
    
    out_sentence = pd.DataFrame({'id': dataset['id'], 'sentence' : dataset['sentence'], 'subject_entity': dataset['subject_entity'], 
                                       'object_entity': dataset['object_entity'],'subject_type': dataset['subject_type'],
                                       'object_type': dataset['object_type'], 'label': dataset['label'],
                                       'subject_idx': update_sub_idx, 'object_idx': update_obj_idx})
    return out_sentence

def random_delete(dataset, p):
    new_sentence= []
    for sen,sub_idx,obj_idx in zip(dataset['sentence'],dataset['subject_idx'],dataset['object_idx']):

        # 0~1 사이의 random한 수를 뽑아 확률 p보다 작을 시 delete
        if random.random() <= p:
        
            sub_start_idx = sen.find('@')
            sub_len = sub_idx[1]-sub_idx[0]+1
            tmp_sub = sen[sub_start_idx:sub_start_idx+sub_len]
            
            obj_start_idx = sen.find('#')
            obj_len = obj_idx[1]-obj_idx[0]+1
            tmp_obj = sen[obj_start_idx:obj_start_idx+obj_len]
            
            # TEM한 sub_entity를 @로 바꿔준다
            sen=sen.replace(tmp_sub,'@')

            # TEM한 obj_entity를 @로 바꿔준다
            sen=sen.replace(tmp_obj,'#')
            is_delete = False

            '''
            위의 두 번의 치환 후 다음과 같은 결과가 나온다.
            Before
            〈Something〉는 #^PER^조지 해리슨#이 쓰고 @*ORG*비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다.

            After
            〈Something〉는 #이 쓰고 @가 1969년 앨범 《Abbey Road》에 담은 노래다.
            '''

            words = sen.split()
            while is_delete == False:
                '''
                After된 문장의 길이가 9이다.
                0~8 사이의 임의의 정수를 뽑아 @, #이 아닌
                '''
                delete_idx = random.randint(0,len(words)-1)

                # 선택한 단어에 @ or #이 들어있지 않다면 삭제해준다.
                if '@' not in words[delete_idx] and '#' not in words[delete_idx]:
                    is_delete=True
                    del words[delete_idx]
                    sen=" ".join(words)

                    # @/#을 다시 원래 entity로 바꿔준다.
                    sen=sen.replace('@',tmp_sub)
                    sen=sen.replace('#',tmp_obj)
                    new_sentence.append(sen)

        else:
            new_sentence.append(sen)

    out_sentence = pd.DataFrame({'id': dataset['id'], 'sentence' : new_sentence, 'subject_entity': dataset['subject_entity'], 
                                       'object_entity': dataset['object_entity'],'subject_type': dataset['subject_type'],
                                       'object_type': dataset['object_type'], 'label': dataset['label'],
                                       'subject_idx': dataset['subject_idx'], 'object_idx': dataset['object_idx']})
    return out_sentence

def random_swap(dataset, p):
    new_sentence= []
    for sen,sub_idx,obj_idx in zip(dataset['sentence'],dataset['subject_idx'],dataset['object_idx']):
        #print(sen)
        # p보다 낮은 값이 나오면 random swap 큰 값이 나오면 그대로
        sub_start_idx = sen.find('@')
        sub_len = sub_idx[1]-sub_idx[0]+1
        tmp_sub = sen[sub_start_idx:sub_start_idx+sub_len]
            
        obj_start_idx = sen.find('#')
        obj_len = obj_idx[1]-obj_idx[0]+1
        tmp_obj = sen[obj_start_idx:obj_start_idx+obj_len]

        if random.random() <= p:
            sen = sen.replace(tmp_sub,"@")
            sen = sen.replace(tmp_obj,"#")
            
            words = sen.split()
            random_idx_1 = random.randint(0, len(words) - 1)
            random_idx_2 = random_idx_1

            while random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, len(words) - 1)

            words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
            sen =" ".join(words)
            sen = sen.replace('@',tmp_sub)
            sen = sen.replace('#',tmp_obj)
            new_sentence.append(sen)
        else:
            new_sentence.append(sen)

    out_sentence = pd.DataFrame({'id': dataset['id'], 'sentence' : new_sentence, 'subject_entity': dataset['subject_entity'], 
                                       'object_entity': dataset['object_entity'],'subject_type': dataset['subject_type'],
                                       'object_type': dataset['object_type'], 'label': dataset['label'],
                                       'subject_idx': dataset['subject_idx'], 'object_idx': dataset['object_idx']})
    return out_sentence