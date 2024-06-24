import torch 
import tqdm 

from semantic_graph_utils import build_detailed_tree, merge_month, count_nodes, draw_graph

verb_pos = ['VBZ', 'VBN', 'VBD', 'VBP', 'VB', 'VBG']
prep_pos = ['PP', 'IN', 'TO']
subj_and_obj = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass'] + ['dobj', 'pobj', 'iobj']
conj = ['conj', 'parataxis']
modifier_pos = ['JJ', 'FW', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
modifiers = ['amod', 'nn', 'mwe', 'advmod', 'quantmod', 'npadvmod', 'advcl', 'poss', 
             'possessive', 'neg', 'auxpass', 'aux', 'det', 'dep', 'predet', 'num']

prep_pos = ['PP', 'IN', 'TO']
modefier_pos = ['JJ', 'FW', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
modifiers = ['amod', 'nn', 'mwe', 'num', 'quantmod', 'dep', 'number', 'auxpass', 'partmod', 'poss', 
             'possessive', 'neg', 'advmod', 'npadvmod', 'advcl', 'aux', 'det', 'predet', 'appos']
prune_list = ['punct', 'cc', 'preconj']

subj_and_obj = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass'] + ['dobj', 'pobj', 'iobj']
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']




def build_tree(sent):
    dep, sequence, title = sent['dependency_parse'], sent['coreference'], sent['title']
    root = [i for i in range(len(dep)) if dep[i]['head'] == -1]
    heads_dep = [w['dep'] for w in dep if w['head'] == root[0]]

    word_type = 'V' if dep[root[0]]['pos'] in verb_pos or 'cop' not in heads_dep else 'A'
    tree = build_detailed_tree(sequence, dep, root[0], word_type)

    return {'words': sequence, 'tree': tree, 'title': title}


def prune(node, sequence):
    ## collect child nodes
    nouns = node['noun'] if 'noun' in node else []
    verbs = node['verb'] if 'verb' in node else []
    attributes = node['attribute'] if 'attribute' in node else []
    ## prune and update child node sets
    Ns, Vs, As = [], [], []
    for child in nouns + verbs + attributes:
        if child['pos'] not in prep_pos and child['dep'] in prune_list:
            Ns += child['noun'] if 'noun' in child else []
            Vs += child['verb'] if 'verb' in child else []
            As += child['attribute'] if 'attribute' in child else []
        else:
            Ns += [child] if child in nouns else []
            Vs += [child] if child in verbs else []
            As += [child] if child in attributes else []
    ## do pruning and merging on child nodes
    Ns = [prune(n, sequence) for n in Ns]
    Vs = [prune(v, sequence) for v in Vs]
    As = [prune(a, sequence) for a in As]
    ## do merging
    slf = {k:v for k,v in node.items() if k not in ['noun', 'verb', 'attribute']}
    if As:
        slf['attribute'] = As
    slf = merge(slf, sequence)
    ## get final node
    wrap = {'dep':slf['dep'], 'word':slf['word'], 'index':slf['index'], 'pos':slf['pos'], 
            'type':slf['type'], 'noun':Ns, 'verb':Vs}
    if 'attribute' in slf:
        wrap['attribute'] = slf['attribute']    
    if not Ns:
        del wrap['noun']
    if not Vs:
        del wrap['verb']        
    return wrap

def rearrange(node, sequence):
    ## collect child node sets and do tree-rearranging on child nodes
    noun, verb = None, None
    slf = {k:v for k,v in node.items() if k not in ['noun', 'verb']}
    noun = [rearrange(n, sequence) for n in node['noun']] if 'noun' in node else []
    verb = [rearrange(v, sequence) for v in node['verb']] if 'verb' in node else []
    if 'attribute' in node:
        slf['attribute'] = [rearrange(a, sequence) for a in node['attribute']]
    ## redirect grandchild nodes to the current node
    ## rule: redirect the parallel words to their real parents
    if noun and node['type'] != 'V':    # those nodes of type 'V' will be rearranged later
        for id_n in range(len(noun)):
            if noun[id_n]['dep'] in subj_and_obj and 'verb' not in noun[id_n] and 'noun' in noun[id_n]:
                new_nouns, rearrg_nouns = [], []
                dep_list = [n['dep'] == 'conj' for n in noun[id_n]['noun']]     # whether a parallel word
                for i, grandchild in enumerate(noun[id_n]['noun']):
                    if dep_list[i]:
                        grandchild['dep'] = noun[id_n]['dep']
                        rearrg_nouns.append(grandchild)
                    else:
                        new_nouns.append(grandchild)
                if len(new_nouns) > 0:
                    noun[id_n]['noun'] = new_nouns
                else:
                    del noun[id_n]['noun']
                noun += rearrg_nouns
    ## merge preposition and its only child node (pobject) as one node [node type = 'M' (modifier)]
    if noun and (not verb) and ('attribute' not in node):
        if len(noun) == 1 and node['dep'] == 'prep' and noun[0]['dep'] in subj_and_obj:
            if ('noun' not in noun[0]) and ('verb' not in noun[0]):
                indexes = node['index'] + noun[0]['index']
                indexes.sort(key=lambda x: x)
                wrap = {'dep':noun[0]['dep'], 'word':[sequence[i] for i in indexes], 'index':indexes,
                        'pos':noun[0]['pos'], 'type': 'M'}
                if 'attribute' in noun[0]:
                    wrap['attribute'] = noun[0]['attribute']
                return wrap
            ## if has more than one nodes, do redirecting
            elif 'verb' not in noun[0] and 'noun' in noun[0]:
                new_nouns, gg_nouns = [], []
                dep_list = [n['dep'] == 'conj' for n in noun[0]['noun']]
                for i, grandchild in enumerate(noun[0]['noun']):
                    if dep_list[i]:
                        grandchild['dep'] = noun[0]['dep']
                        gg_nouns.append(grandchild)
                    else:
                        new_nouns.append(grandchild)
                if len(new_nouns) > 0:
                    noun[0]['noun'] = new_nouns
                else:
                    del noun[0]['noun']
                noun += gg_nouns
    ## for node which represents time/date (i.e., contain month word),
    #  merge it with all its attribute child nodes
    if 'attribute' in slf and any([w in months for w in slf['word']]):
        slf['index'], slf['word'] = merge_month(slf, sequence)
        del slf['attribute']
    ## get final node
    wrap = {'dep':slf['dep'], 'word':slf['word'], 'index':slf['index'], 'pos':slf['pos'], 
            'type':slf['type'], 'noun':noun, 'verb':verb}
    if 'attribute' in slf:
        wrap['attribute'] = slf['attribute']
    if not noun:
        del wrap['noun']
    if not verb:
        del wrap['verb']
    return wrap

def get_graph(tree):
    ## collect nodes
    nodes = []
    count_nodes(nodes, tree)
    ## draw edges
    edges = [['' for _ in nodes] for _ in nodes]
    draw_graph(edges, tree)

    return {'nodes': nodes, 'edges': edges}


def merge(corpus):

    def reindex(sequence):
        '''do reindexing
        because we have coreference resolution, which means
        there may be more than one words in the so-called 'one' word in fact
        '''
        cnt, new_seq = 0, []
        dicts = [[] for _ in sequence]
        for i, w in enumerate(sequence):
            wrd_cnt = max(len(w.strip().split(' ')), 1)
            dicts[i] = [i for i in range(cnt, cnt + wrd_cnt)]
            cnt += wrd_cnt
            new_seq += w.strip().split(' ')
        length = len(new_seq)
        return dicts, length, new_seq

if __name__ == '__main__':
    
    graphs = []
    for idx, sample in tqdm(enumerate(data), desc='   - (Building Graphs) -   '):
        corpus = sample['evidence']
        evidence = []
        for sent in corpus:
            sent = build_tree(sent)
            sent = {'sequence':sent['words'], 'tree':prune(sent['tree'], sent['words'])}
            sent = {'sequence': sent['sequence'], 'tree': rearrange(sent['tree'], sent['sequence'])}
            evidence.append({'sequence':sent['sequence'], 'graph':get_graph(sent['tree'])})
        graph = merge(evidence)
        graphs.append(graph)