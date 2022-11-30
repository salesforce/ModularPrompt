import torch
import os
import numpy as np
import random
import csv
import pickle5 as pickle
from collections import Counter, defaultdict
import heapq
import logging

logger = logging.getLogger('root')

def getonebatchresult_classification(sen,target,preds):
    preds = [pred.replace('&', 'and').lower() for pred in preds]
    target = [t.replace('&', 'and').lower() for t in target]
    return list(map(lambda x: x, target)), list(map(lambda x: x, preds))

def getonebatchresult_ner(sen,target,preds):
    ''' convert text format prediction to BIO format
    '''
    # all labels of fewNERD
    typedic = {'art music', 'art written art', 'art film', 'art other', 'art broadcast', 'person soldier', 'person athlete', 'person politician', 'person director', 'person scholar', 'person actor', 'person artist or author', 'person other', 'location GPE', 'location island', 'location bodies of water', 'location park', 'location other', 'location way', 'location mountain', 'building hotel', 'building airport', 'building hospital', 'building other', 'building library', 'building sports', 'building restaurant', 'building theater', 'other astronomy', 'other educational degree', 'other language', 'other disease', 'other livingthing', 'other award', 'other chemical', 'other law', 'other biology', 'other currency', 'other god', 'other medical', 'organization political party', 'organization media', 'organization government', 'organization religion', 'organization sports team', 'organization company', 'organization show', 'organization other', 'organization sports league', 'organization education', 'event other', 'event war', 'event protest', 'event disaster', 'event sports', 'product other', 'product weapon', 'product ship', 'product train', 'product software', 'product food', 'product game', 'product airplane', 'product car'}
    
    sennum = len(sen)
    restar = []
    respred = []
    for i in range(sennum):
        thissen, thistar, thispred = sen[i], target[i], preds[i]

        thissenlow = thissen.lower()

        sensplit = thissen.split(' ')
        sensplitlow = thissenlow.split(' ')

        tarres = ['O' for j in range(len(sensplit))]
        predres = ['O' for j in range(len(sensplit))]

        if thistar == 'end' and thispred == 'end':
            restar.append(tarres)
            respred.append(predres)
            continue

        if len(thistar) > 0 and thistar[-1] == ';':
            thistar = thistar[:-1]

        tarsplit1 = thistar.split(';')

        if thistar != 'end':
            for j in range(len(tarsplit1)):
                tarsplit2 = tarsplit1[j].split('!')
                if len(tarsplit2) != 2:
                    continue
                entity = tarsplit2[0].strip(' ')
                entitylow = entity.lower()
                type = tarsplit2[1].strip(' ')
                if type not in typedic:
                    continue
                #if thissen.find(entity) == -1:
                if thissenlow.find(entitylow) == -1:
                    continue
                trueindex = -100
                #entitysplit = entity.split(' ')
                entitysplit = entitylow.split(' ')
                for k in range(len(sensplit)):
                    #if sensplit[k] == entitysplit[0]:
                    if sensplitlow[k] == entitysplit[0] or entitysplit[0] in sensplitlow[k]:
                        iftrue = True
                        for l in range(1, len(entitysplit)):
                            if sensplitlow[k + l] != entitysplit[l] and (entitysplit[0] not in sensplitlow[k]):
                                iftrue = False
                                break
                        if iftrue:
                            trueindex = k
                            break
                if trueindex == -100:
                    continue
                for k in range(trueindex, trueindex + len(entitysplit)):
                    if k == trueindex:
                        tarres[k] = 'B-' + type #typedic[type]
                    else:
                        tarres[k] = 'I-' + type #typedic[type]

        if len(thispred) > 0 and thispred[-1] == ';':
            thispred = thispred[:-1]

        tarsplit3 = thispred.split(';')

        if thispred != "end":
            for j in range(len(tarsplit3)):
                tarsplit4 = tarsplit3[j].split('!')
                if len(tarsplit4) != 2:
                    continue
                entity = tarsplit4[0].strip(' ')
                entitylow = entity.lower()
                type = tarsplit4[1].strip(' ')
                if type not in typedic:
                    continue
                #if thissen.find(entity) == -1:
                if thissenlow.find(entitylow) == -1:
                    continue
                trueindex = -100
                #entitysplit = entity.split(' ')
                entitysplit = entitylow.split(' ')
                for k in range(len(sensplit)):
                    #if sensplit[k] == entitysplit[0]:
                    if sensplitlow[k] == entitysplit[0] or entitysplit[0] in sensplitlow[k]:
                        iftrue = True
                        for l in range(1, len(entitysplit)):
                            #if sensplit[k + l] != entitysplit[l]:
                            if sensplitlow[k + l] != entitysplit[l] and (entitysplit[0] not in sensplitlow[k]):
                                iftrue = False
                                break
                        if iftrue:
                            trueindex = k
                            break
                if trueindex == -100:
                    continue
                else:
                    for k in range(trueindex, trueindex + len(entitysplit)):
                        if k == trueindex:
                            predres[k] = 'B-' + type # typedic[type]
                        else:
                            predres[k] = 'I-' + type # typedic[type]
        restar.append(tarres)
        respred.append(predres)
    return restar, respred

def save_test(all_target, all_pred, all_x, args):
    target_file = f"{args.save_test_dir}/{args.log_name}_target.txt"
    pred_file = f"{args.save_test_dir}/{args.log_name}_pred.txt"
    source_file = f"{args.save_test_dir}/{args.log_name}_sent.txt"
    for _file,_data in zip([target_file, pred_file, source_file], [all_target, all_pred, all_x]):
        with open(_file, 'a') as wf:
            for line in _data:
                wf.write(line+'\n')


def label_init_stats(args, label_name_embs):
    def compute_overlap(toks1, toks2):
        assert len(toks1) == len(toks2)
        return len([tok for tok in toks1 if tok in toks2]) / len(toks1)
    l2l = {}
    default_keys = label_name_embs.keys()
    line = ''
    line += f'{" ":20}  '
    for label_name in default_keys:
        line += f'{label_name:<20}  '
    logger.info(line)
    
    # compute overlap between label pairs
    for label_name, v in label_name_embs.items():
        tokens = v[2]
        l2l[label_name] = {}
        for _label_name, _v in label_name_embs.items():
            _tokens = _v[2]
            overlap = compute_overlap(tokens, _tokens)
            l2l[label_name][_label_name] = overlap
    
    for key in default_keys:
        line = f'{key:<20}  '
        for _key in default_keys:
            line += f'{l2l[key][_key]:<20.04f}  '
        logger.info(line)

    #import pdb;pdb.set_trace()
    

def get_seq_label_tokens(train_dataset):
    l_tokens = defaultdict(Counter)
    for (input_, target_) in train_dataset.data:
        inputres = train_dataset.tokenizer.batch_encode_plus([input_], padding=False, truncation=False, return_tensors="pt")
        target_ = set([t.split(' ! ')[1] for t in target_.rstrip().split(' ;') if t])
        for target in target_:
            l_tokens[target].update(inputres['input_ids'].squeeze().tolist())
    return l_tokens


def get_label_tokens(args, train_dataset):
    if args.dataset == 'fewNERD':
        return get_seq_label_tokens(train_dataset)
    l_tokens = {}
    for (input_, target_) in train_dataset.data:
        inputres = train_dataset.tokenizer.batch_encode_plus([input_], padding=False, truncation=False, return_tensors="pt")
        target_ = target_.rstrip()
        if target_ not in l_tokens:
            l_tokens[target_] = Counter()
        l_tokens[target_].update(inputres['input_ids'].squeeze().tolist())
    return l_tokens


@torch.no_grad()
def similarity_matrix(label_embedding_src, label_embedding_tgt, output_file=None):
    def mean_pw_cosine_similarity(input_a, input_b):
        normalized_input_a = torch.nn.functional.normalize(input_a)  
        normalized_input_b = torch.nn.functional.normalize(input_b)
        res = torch.mm(normalized_input_a, normalized_input_b.T)
        return res.mean()

    similarity_dic = {} # {label_name: similarity}

    for label_name, emb in label_embedding_src.items():
        similarity_dic[label_name] = {}
        sum_similarity = 0
        for tgt_label_name, tgt_emb in label_embedding_tgt.items():
            sim = mean_pw_cosine_similarity(emb, tgt_emb).cpu().detach().numpy()
            sum_similarity += sim
            similarity_dic[label_name][tgt_label_name] = sim
        # normalise similarity
        for k,v in similarity_dic[label_name].items():
            similarity_dic[label_name][k] = v / sum_similarity
    
    with open(output_file, 'w') as wf:
        wf.write(f'{" ":20}\t')
        for label_name in label_embedding_tgt.keys():
            wf.write(f'{label_name:<20}\t')
        wf.write('\n')
        for key in label_embedding_src.keys():
            wf.write(f'{key:<20}\t')
            for _key in label_embedding_tgt.keys():
                wf.write(f'{similarity_dic[key][_key]:<20.04f}\t')
            wf.write('\n')


def raw(label_name): # drop dataset tag if have
    if ':' in label_name: # dataset tagged label, remove dataset tag as name
        real_name = label_name[3:]
    else:
        real_name = label_name
    return real_name


@torch.no_grad()
def get_mix_prompt_l_embedding_v2(args, model, tokenizer, labels_to_update, prev_labels, is_random=False):
    '''transfer_v2: label emb transfer by label name similarity
    Args:
        is_random: if True, random similarity mapping
    '''
    def mean_pw_cosine_similarity(input_a, input_b):
        if is_random:
            return torch.rand(1, device=input_a.device)
        normalized_input_a = torch.nn.functional.normalize(input_a)  
        normalized_input_b = torch.nn.functional.normalize(input_b)
        res = torch.mm(normalized_input_a, normalized_input_b.T)
        return res.mean()
    def get_embs(toks, t5_embedding):
        encoderes = tokenizer.batch_encode_plus([toks], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1].to(t5_embedding.weight.device)
        embeddingres = t5_embedding(touse).clone().detach()
        return embeddingres
    def log():
        logger.info(f"prev labels: {prev_labels}")
        logger.info(f"labels_to_update {labels_to_update}")

    if args.verbose:
        log()

    label_name_embs = {}
    t5_embedding = model.model.get_input_embeddings()
    for label_name in labels_to_update:
        cur_name_emb = model.prompt_fix_dict[label_name]
        label_name_embs[label_name] = [cur_name_emb, None]
        # compute similarity to every prev label prompt
        similarity_dic = {} # {label_name: similarity}
        sum_similarity = 0
        for prev_label_name in prev_labels:
            prev_name_emb = get_embs(raw(prev_label_name), t5_embedding)# model.prompt_fix_dict[prev_label_name]
            sim = mean_pw_cosine_similarity(cur_name_emb, prev_name_emb)
            similarity_dic[prev_label_name] = sim
            sum_similarity += sim
        
        if args.forward_transfer_similarity_type == 'top3': # mean of top 3
            sim_list = [(sim, label_name) for label_name,sim in similarity_dic.items()]
            top_sim_list = heapq.nlargest(3, sim_list)
            sum_similarity = sum([sim for sim,_ in top_sim_list])
            for sim, prev_label_name in top_sim_list:
                alpha = sim / sum_similarity
                weighted_emb = alpha * model.prompt_dict[prev_label_name].clone().detach()
                if label_name_embs[label_name][1] is None:
                    label_name_embs[label_name][1] = weighted_emb
                else:
                    label_name_embs[label_name][1] += weighted_emb
            similarity_dic = {l:sim for sim,l in top_sim_list}
        if args.verbose:
            sim_report = {k:(v.data/sum_similarity).item() for k,v in similarity_dic.items()}
            sim_report = sorted(sim_report.items(), key=lambda item: item[1], reverse=True)
            logger.info(f'get_mix_prompt_l_embedding_v2, label: {label_name}, similarity : {sim_report}')
    return label_name_embs


def get_mix_prompt_l_embedding_v1(args, model, tokenizer, label_prompt_length, l_tokens, prev_labels, current_wild_stage):
    '''sample top tokens for current stage labels (only update new labels), return t5 embedding
    Args:
        l_tokens: dict {label_name: {num_times: token}}, tokens appeared in data of that label
        prev_labels: set {label_name}
    '''
    def sample_topk_tokens(args, topk, t5_embedding, label, alltokens):
        sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)
        top1000 = []
        for one in sortedalltoken:
            if one[0] == 2:
                continue
            else:
                if len(top1000) < 1000:
                    top1000.append(one)
                else:
                    break
        vocab = tokenizer.get_vocab()
        topk_emb = []
        touse = random.sample(top1000, label_prompt_length)
        reuse_cnt = 0
        if args.forward_transfer_type == 'sample_label' and prev_labels:
            rlabel_name = random.choice(list(prev_labels.keys()))
        for cur_idx, tok in enumerate(touse):
            prev_ref = []
            if args.forward_transfer_type == 'token_replace' and prev_labels:
                for label_name, tokens in prev_labels.items():
                    if tok[0] in tokens:
                        idx = tokens.index(tok[0])
                        prev_ref.append(model.prompt_dict[label_name][idx].clone().detach().unsqueeze(0))
            elif args.forward_transfer_type == 'sample_label' and prev_labels:
                prev_ref.append(model.prompt_dict[rlabel_name][cur_idx].clone().detach().unsqueeze(0))
            if len(prev_ref) > 0:
                topk_emb.append(torch.mean(torch.stack(prev_ref), dim=0))
                reuse_cnt += 1
            else:
                topk_emb.append(t5_embedding.weight[tok[0]].clone().detach().unsqueeze(0))
        if args.verbose:
            logger.info(f'get_mix_prompt_l_embedding_v1, label: {label}, sampled tokens: {[tok[0] for tok in touse]}')
            logger.info(f'reuse cnt: {reuse_cnt}')
        return torch.cat(topk_emb, 0), [tok[0] for tok in touse]

    def get_embs(toks, t5_embedding):
        encoderes = tokenizer.batch_encode_plus([toks], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1].to(t5_embedding.weight.device)
        embeddingres = t5_embedding(touse).clone().detach()
        return embeddingres

    def _no_tag_(name):
        '''remove tag in label name: e.g. ag:world -> world '''
        return name.split(':')[1] if ':' in name else name
    
    label_mapping = {_no_tag_(k):k for k in current_wild_stage}

    t5_embedding = model.model.get_input_embeddings()
    label_name_embs = {}
    for label_name, tokens in l_tokens.items():
        label_name = label_mapping[label_name]
        if label_name in prev_labels: # already initialised label, pass
            continue
        label_name_embs[label_name] = [None, None, None]
        embeddingres = get_embs(raw(label_name), t5_embedding)
        label_name_embs[label_name][0] = embeddingres
        label_name_embs[label_name][1], label_name_embs[label_name][2] = sample_topk_tokens(args, label_prompt_length, t5_embedding, label_name, tokens)
    return label_name_embs


def get_mix_prompt_embedding(args, model, tokenizer, label_prompt_length):
    ''' random init token embedding for label prompts
    '''
    def sample_top_k_tokens(args, topk, t5_embedding):
        if args.toptokens == 'c4':
            with open('allnumber.pickle', 'rb') as fr:
                alltokens = pickle.load(fr)
            sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)
        top5000 = []
        for one in sortedalltoken:
            if one[0] == 2:
                continue
            else:
                if len(top5000) < 5000:
                    top5000.append(one)
                else:
                    break
        vocab = tokenizer.get_vocab()
        while True:
            topk_emb = []
            touse = random.sample(top5000, topk)
            for tok in touse:
                topk_emb.append(t5_embedding.weight[tok[0]].clone().detach().unsqueeze(0))
            yield torch.cat(topk_emb, 0)

    def get_embs(toks, t5_embedding):
        encoderes = tokenizer.batch_encode_plus([toks], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        embeddingres = t5_embedding(touse).clone().detach()
        return embeddingres
    
    t5_embedding = model.model.get_input_embeddings()
    embs_generator = sample_top_k_tokens(args, label_prompt_length, t5_embedding)
    
    if args.dataset == 'fewNERD':
        label_names = ['art music', 'art written art', 'art film', 'art other', 'art broadcast', 'person soldier', 'person athlete', 'person politician', 'person director', 'person scholar', 'person actor', 'person artist or author', 'person other', 'location GPE', 'location island', 'location bodies of water', 'location park', 'location other', 'location way', 'location mountain', 'building hotel', 'building airport', 'building hospital', 'building other', 'building library', 'building sports', 'building restaurant', 'building theater', 'other astronomy', 'other educational degree', 'other language', 'other disease', 'other livingthing', 'other award', 'other chemical', 'other law', 'other biology', 'other currency', 'other god', 'other medical', 'organization political party', 'organization media', 'organization government', 'organization religion', 'organization sports team', 'organization company', 'organization show', 'organization other', 'organization sports league', 'organization education', 'event other', 'event war', 'event protest', 'event disaster', 'event sports', 'product other', 'product weapon', 'product ship', 'product train', 'product software', 'product food', 'product game', 'product airplane', 'product car']
        task_embeddingres = get_embs("name entity recognition", t5_embedding)
    elif args.dataset == 'huffpost':
        label_names = ['queer voices','weird news','comedy','style','home and living','style and beauty','entertainment','weddings','politics','wellness','travel','healthy living','parenting','parents','latino voices','food and drink','tech','science','black voices','divorce','worldpost','business','religion','world news','crime','green','fifty','good news','sports','education','money','arts','impact','taste','environment','the worldpost','women','college','media','culture and arts','arts and culture']
        task_embeddingres = get_embs("news classification huffpost", t5_embedding)
    elif args.dataset == 'fewrel':
        label_names = ['sibling', 'place served by trainsport hub', 'architect', 'location', 'spouse', 'licensed to broadcast to', 'member of', 'tributary', 'location of formation', 'taxon rank', 'occupant', 'notable work', 'followed by', 'developer', 'said to be the same as', 'competition class', 'voice type', 'language of work or name', 'residence', 'part of', 'sports season of', 'performer', 'position held', 'located on terrain feature', 'member of political party', 'owned by', 'country of origin', 'position palyed on team', 'occupation', 'director', 'winner', 'mountain range', 'operator', 'participant of', 'father', 'distributor', 'applies to jurisdiction', 'after a work by', 'movement', 'instrument', 'platform', 'manufacturer', 'record label', 'subsidiary', 'country', 'original network', 'mouth of the watercourse', 'constellation', 'located in or next to body of water', 'headquarters location', 'country of citizenship', 'mother', 'work location', 'participant', 'field of work', 'screenwriter', 'compose', 'nominated for', 'genre', 'participating teams', 'instance of', 'religion', 'military rank', 'league', 'located in the administrative territorial entity', 'child', 'original language of work', 'sport', 'publisher', 'characters', 'head of government', 'successful candidate', 'crosses', 'follows', 'military branch', 'operating system', 'contains administrative territorial entity', 'has part', 'heritage designation', 'main subject']
        task_embeddingres = get_embs("relation extraction", t5_embedding)
        
    label_name_embs = {k:[None, None] for k in label_names}
    
    for label_name, v in label_name_embs.items():
        embeddingres = get_embs(raw(label_name), t5_embedding)
        label_name_embs[label_name][0] = embeddingres
        label_name_embs[label_name][1] = next(embs_generator)
    
    return label_name_embs


def get_prompt_embedding(args, model,tokenizer,prompt_length,alllabel=None):
    ''' random init token embedding for soft prompt tuning
    '''
    t5_embedding = model.model.get_input_embeddings()
    promptinitembedding = torch.FloatTensor(prompt_length, t5_embedding.weight.size(1))
    startindex = 0
    if args.dataset == 'fewNERD':
        alllabel = ['art music', 'art written art', 'art film', 'art other', 'art broadcast', 'person soldier', 'person athlete', 'person politician', 'person director', 'person scholar', 'person actor', 'person artist or author', 'person other', 'location GPE', 'location island', 'location bodies of water', 'location park', 'location other', 'location way', 'location mountain', 'building hotel', 'building airport', 'building hospital', 'building other', 'building library', 'building sports', 'building restaurant', 'building theater', 'other astronomy', 'other educational degree', 'other language', 'other disease', 'other livingthing', 'other award', 'other chemical', 'other law', 'other biology', 'other currency', 'other god', 'other medical', 'organization political party', 'organization media', 'organization government', 'organization religion', 'organization sports team', 'organization company', 'organization show', 'organization other', 'organization sports league', 'organization education', 'event other', 'event war', 'event protest', 'event disaster', 'event sports', 'product other', 'product weapon', 'product ship', 'product train', 'product software', 'product food', 'product game', 'product airplane', 'product car']
    elif args.dataset == 'huffpost':
        alllabel = ["news classification huffpost", 'queer voices','weird news','comedy','style','home and living','style and beauty','entertainment','weddings','politics','wellness','travel','healthy living','parenting','parents','latino voices','food and drink','tech','science','black voices','divorce','worldpost','business','religion','world news','crime','green','fifty','good news','sports','education','money','arts','impact','taste','environment','the worldpost','women','college','media','culture and arts','arts and culture']
    elif args.dataset == 'fewrel':
        alllabel = ['sibling', 'place served by trainsport hub', 'architect', 'location', 'spouse', 'licensed to broadcast to', 'member of', 'tributary', 'location of formation', 'taxon rank', 'occupant', 'notable work', 'followed by', 'developer', 'said to be the same as', 'competition class', 'voice type', 'language of work or name', 'residence', 'part of', 'sports season of', 'performer', 'position held', 'located on terrain feature', 'member of political party', 'owned by', 'country of origin', 'position palyed on team', 'occupation', 'director', 'winner', 'mountain range', 'operator', 'participant of', 'father', 'distributor', 'applies to jurisdiction', 'after a work by', 'movement', 'instrument', 'platform', 'manufacturer', 'record label', 'subsidiary', 'country', 'original network', 'mouth of the watercourse', 'constellation', 'located in or next to body of water', 'headquarters location', 'country of citizenship', 'mother', 'work location', 'participant', 'field of work', 'screenwriter', 'compose', 'nominated for', 'genre', 'participating teams', 'instance of', 'religion', 'military rank', 'league', 'located in the administrative territorial entity', 'child', 'original language of work', 'sport', 'publisher', 'characters', 'head of government', 'successful candidate', 'crosses', 'follows', 'military branch', 'operating system', 'contains administrative territorial entity', 'has part', 'heritage designation', 'main subject']

    for one in alllabel:
        encoderes = tokenizer.batch_encode_plus([one], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        embeddingres = t5_embedding(touse.to(t5_embedding.weight.device)).clone().detach()
        if embeddingres.shape[0] > 1:
            embeddingres = torch.mean(embeddingres, 0, keepdim=True)
        promptinitembedding[startindex] = embeddingres
        startindex += 1
        if startindex >= prompt_length:
            break
        
    if args.toptokens == 'c4':
        with open('allnumber.pickle', 'rb') as fr:
            alltokens = pickle.load(fr)
        sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)

    top5000 = []
    for one in sortedalltoken:
        if one[0] == 2:
            continue
        else:
            if len(top5000) < 5000:
                top5000.append(one)
            else:
                break
    vocab = tokenizer.get_vocab()
    
    randomtokennum = prompt_length - len(alllabel)
    if randomtokennum > 0:
        touse = random.sample(top5000, randomtokennum)
        for one in touse:
            promptinitembedding[startindex] = t5_embedding.weight[one[0]].clone().detach()
            startindex += 1
    
    return promptinitembedding
