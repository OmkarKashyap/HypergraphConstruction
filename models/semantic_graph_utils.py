

verb_pos = ['VBZ', 'VBN', 'VBD', 'VBP', 'VB', 'VBG']
prep_pos = ['PP', 'IN', 'TO']
subj_and_obj = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass'] + ['dobj', 'pobj', 'iobj']
conj = ['conj', 'parataxis']
modifier_pos = ['JJ', 'FW', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
modifiers = ['amod', 'nn', 'mwe', 'advmod', 'quantmod', 'npadvmod', 'advcl', 'poss', 
             'possessive', 'neg', 'auxpass', 'aux', 'det', 'dep', 'predet', 'num']

subj_and_obj = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass'] + ['dobj', 'pobj', 'iobj']
others_dep = ['poss', 'npadvmod', 'appos', 'nn']
conj = ['conj', 'cc', 'preconj', 'parataxis']
verb_pos = ['VBZ', 'VBN', 'VBD', 'VBP', 'VB', 'VBG', 'IN', 'TO', 'PP']
noun_pos = ['NN', 'NNP', 'NNS', 'NNPS']
subj = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']

def build_detailed_tree(sequence, all_dep, root, word_type):

    def is_noun(node):
        return node['dep'] in subj_and_obj or (all_dep[root]['dep'] in subj_and_obj and node['dep'] == 'conj')
    
    def is_verb(node):
        return (node['dep'] == 'cop' and word_type == 'A') or (word_type == 'V' and node['dep'] == 'conj')
    ##=== initialize tree-node ===##
    element = all_dep[root]
    word_type = 'V' if element['pos'] in verb_pos else 'A'
    node = {'word': [sequence[root]], 'index': [root], 'type': word_type, 'dep': element['dep'], 'pos': element['pos']}
    ##=== classify child node sets ===##
    children = [(i, elem) for i, elem in enumerate(all_dep) if elem['head'] == root]
    nouns = [child for child in children if is_noun(child[1])]
    if len(nouns) > 0:
        node['noun'] = [build_detailed_tree(sequence, all_dep, child[0], 'A') for child in nouns]
    verbs = [child for child in children if is_verb(child[1])]
    if len(verbs) > 0:
        node['verb'] = [build_detailed_tree(sequence, all_dep, child[0], 'V') for child in verbs]
    attributes = [child for child in children if child not in nouns + verbs]
    if len(attributes) > 0:
        node['attribute'] = [build_detailed_tree(sequence, all_dep, child[0], 'A') for child in attributes]
    ##=== do node-merging ===##
    if 'attribute' in node:
        node = merge_node(node, sequence)
        
    return node


def merge_month(node, sequence):    
    indexes = [idx for idx in node['index']]

    if 'attribute' in node:
        for a in node['attribute']:
            index, _ = merge_month(a, sequence)
            indexes += index

    indexes.sort(key=lambda x:x)
    words = [sequence[i] for i in indexes]

    return indexes, words


def count_nodes(nodes, tree):
    ## initialize node
    str_words = ' '.join(tree['word'])
    node = {'type': tree['type'], 'dep': tree['dep'], 'pos':tree['pos'], 'word': str_words, 'index':tree['index']}
    ## merge almost the same nodes as one node if meet requirements
    for idx, exist in enumerate(nodes):
        # requirement one: has common words and has the same type
        if all([w in exist['word'].split(' ') for w in node['word'].split(' ')]):
            if all([w in node['word'].split(' ') for w in exist['word'].split(' ')]) and node['type'] == exist['type']:
                # requirement two: noun-like nodes
                if node['pos'] in noun_pos or node['dep'] in subj_and_obj + others_dep:
                    # requirement three: has upper case and not quite short
                    if any([w.isupper() for w in node['word']]) and len(node['word'].split(' ')) > 1:
                        # requirement four: enough high level of overlapping
                        if len(node['word'].split(' ')) / len(exist['word'].split(' ')) > 0.9:
                            tree['node_num'] = idx
                            break
    ## added as new node if not meet above requirements
    if 'node_num' not in tree:
        tree['node_num'] = len(nodes)
        nodes.append(node)
    ## collect child nodes
    if 'noun' in tree:
        for child in tree['noun']:
            count_nodes(nodes, child)
    if 'verb' in tree:
        for child in tree['verb']:
            count_nodes(nodes, child)
    if 'attribute' in tree:
        for child in tree['attribute']:
            count_nodes(nodes, child)


def draw_graph(graph, tree):
    index = tree['node_num']
    ## copy existed edges in tree
    children = tree['noun'] if 'noun' in tree else []
    children += tree['verb'] if 'verb' in tree else []
    children += tree['attribute'] if 'attribute' in tree else []
    for child in children:
        if child['dep'] != 'punct' or 'noun' in child or 'verb' in child or 'attribute' in child:
            idx = child['node_num']
            graph[idx][index] = child['dep']
            draw_graph(graph, child)
    ## redirect to make it sure that:
    #  1. entity --> predicate <-- entity in each [entity, predicate, entity] triple
    #  2. those parallel words connect to their real parents
    if 'noun' in tree and 'verb' in tree:
        ## for 'V' type node
        if tree['type'] == 'V' or tree['pos'] in verb_pos:
            for verb in tree['verb']:
                v_idx = verb['node_num']
                is_replace = False  # whether to redirect
                for noun in tree['noun']:
                    if noun['dep'] in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']:
                        is_replace = True
                        n_idx = noun['node_num']
                        graph[n_idx][v_idx] = noun['dep']
                if is_replace and graph[v_idx][index] in conj:
                    graph[v_idx][index] = ''
        ## for 'A'/'M' type node
        else:
            verbs = []
            for v in tree['verb']:
                # collect verbs to do redirecting with
                if v['word'] not in [vrb['word'] for vrb in verbs] or 'noun' in v or 'attribute' in v or 'verb' in v:
                    verbs.append(v)
            tree['verb'] = verbs
            for verb in tree['verb']:
                v_idx = verb['node_num']
                for noun in tree['noun']:
                    n_idx = noun['node_num']
                    # for parallel words
                    if graph[n_idx][index] in conj:
                        if 'verb' not in noun:
                            graph[v_idx][n_idx] = verb['dep']
                            graph[n_idx][index] = ''
                        else:
                            nv_idx = noun['verb'][0]['node_num']
                            for nn in [tmp for tmp in tree['noun'] if tmp['dep'] in subj]:
                                graph[nn['node_num']][nv_idx] = nn['dep']
                                graph[n_idx][index] = ''
                    # for [entity, predicate, entity] triple where predicate is corpula
                    if noun['dep'] in subj_and_obj and noun['dep'] != 'pobj':
                        graph[n_idx][v_idx] = noun['dep']
                        graph[n_idx][index] = ''
                        
                        
                        
def merge_node(raw, sequence):
    node = {k: v for k, v in raw.items()}
    attribute = raw['attribute']

    attr1, attr2 = [], []   # attr1: ok to merge
    indexes = [idx for idx in node['index']]
    for a in attribute:
        if 'attribute' in a or 'noun' in a or 'verb' in a:
            attr2.append(a)
        elif (a['dep'] in modifiers or a['pos'] in modifier_pos) and a['pos'] not in prep_pos:
            attr1.append(a)
            indexes += [idx for idx in a['index']]
        else:
            attr2.append(a)
    
    if len(attr1) > 0:
        indexes.sort(key=lambda x:x)
        flags = [index not in indexes[:idx] for idx, index in enumerate(indexes)]
        if len(indexes) == indexes[-1] - indexes[0] + 1 and all(flags):     # need to be consecutive modifiers
            node['word'] = [sequence[i] for i in indexes]
            node['index'] = indexes
            if len(attr2) > 0:
                node['attribute'] = [a for a in attr2]
            else:
                del node['attribute']
    
    return node