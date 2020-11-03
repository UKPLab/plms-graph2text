"""
Preprocess AMR and surface forms.

Options:

- linearise: for seq2seq learning
  - simplify: simplify graphs and lowercase surface
  - anon: same as above but with anonymisation
  
- triples: for graph2seq learning
  - anon: anonymise NEs

"""
import sys
from amr import AMR, Var, Concept, AMRNumber
import re
import json
import argparse
from collections import Counter
from collections import defaultdict
from copy import deepcopy

from ne_clusters import NE_CLUSTER, QUANT_CLUSTER

##########################

class AMRTree(object):
    """
    Used for printing only
    """
    def __init__(self, label):
        self._label = label
        self._children = []

    def __str__(self):
        if len(self.children) > 0:
            str_chs = ' '.join([str(ch) for ch in self._children])
            if len(self.children) > 1:
                return self._label + ' ( ' + str_chs + ' ) '
            else:
                return self._label + ' ' + str_chs
        else:
            return self._label
        

##########################

def simplify(tokens, v2c):
    new_tokens = []
    for tok in tokens:
        # ignore instance-of
        if tok.startswith('('):
            new_tokens.append('(')
            continue
        elif tok == '/':
            continue
        # predicates, we remove any alignment information and parenthesis
        elif tok.startswith(':'):

            new_tok = tok.strip(')')
            new_tok = new_tok.split('~')[0]
            new_tokens.append(new_tok)

            count_ = tok.count(')')
            for _ in range(count_):
                new_tokens.append(')')

        # concepts/reentrancies, treated similar as above
        else:
            new_tok = tok.strip(')')
            new_tok = new_tok.split('~')[0]
            # now we check if it is a concept or a variable (reentrancy)
            if Var(new_tok) in v2c:
                # reentrancy: replace with concept
                new_tok = v2c[Var(new_tok)]._name
            # remove sense information
            elif re.search(SENSE_PATTERN, new_tok):
                new_tok = new_tok[:-3]
            # remove quotes
            elif new_tok[0] == '"' and new_tok[-1] == '"':
                new_tok = new_tok[1:-1]
            new_tokens.append(new_tok)

            count_ = tok.count(')')
            for _ in range(count_):
                new_tokens.append(')')

    return new_tokens

##########################

def get_name(v, v2c):
    try:
        # Remove sense info from concepts if present
        c = v2c[v]
        if re.search(SENSE_PATTERN, c._name):
            return c._name[:-3]
        else:
            return c._name
    except: # constants: remove quotes if present
        r = str(v).lower()
        if r[0] == '"' and r[-1] == '"':
            return r[1:-1]
        else:
            return r

##########################

def get_nodes(graph):
    v_ids = {}
    rev_v_ids = []
    for concept in graph.concepts():
        v = concept[0]
        #c = concept[1]
        #c_ids[c] = str(len(c_ids))
        #rev_c_ids.append(c)
        v_ids[v] = str(len(v_ids))
        rev_v_ids.append(v)
        
    # Add constant nodes as well
    for constant in graph.constants():
        v_ids[constant] = str(len(v_ids))
        rev_v_ids.append(constant)
    return v_ids, rev_v_ids
        
##########################

def get_nodes2(graph):
    v_ids = {}
    rev_v_ids = []
    filtered = [t for t in graph.triples() if type(t[2]) != Var]
    #v2cs = [t for t in filtered if t[1] == ':instance-of']
    #constants = [t[2] for t in filtered if t[1] != ':instance-of']
    for triple in filtered:
        # Concepts: we map Vars
        if triple[1] == ':instance-of':
            v = triple[0]
        # Constants: we map the actual constant
        else:
            v = triple[2]
        if v not in v_ids:
            # Need this check so we do not add double constants
            v_ids[v] = str(len(v_ids))
            rev_v_ids.append(v)            
    return v_ids, rev_v_ids
        
##########################

def get_triples(graph, v_ids, rev_v_ids, v2c):
    triples = []
    for triple in graph.triples():
        # Ignore top and instance-of
        # TODO: remove wikification
        if triple[1] == ':top' or triple[1] == ':instance-of':
            continue
        predicate = triple[1]
        try:
            v1 = triple[0]
            c1 = v2c[v1]                        
        except: # If it is not a concept it is a constant:
            v1 = triple[0]
        try:
            v2 = triple[2]
            c2 = v2c[v2]
        except:
            v2 = triple[2]
        triples.append((v_ids[v1], v_ids[v2], predicate))
        if args.add_reverse:
            # Add reversed edges if requested
            if predicate.endswith('-of'):
                rev_predicate = predicate[:-3]
            else:
                rev_predicate = predicate + '-of'
                triples.append((v_ids[v2], v_ids[v1], rev_predicate))
                        
    # Add self-loops
    for v in v_ids:
        triples.append((v_ids[v], v_ids[v], 'self'))
    return triples

##########################

def anonymize_nes(graph, triples, output_triples, v2c, anon_ids, anon_map, anon_surf):
    """
    Anonymize NEs. We replace the node with a clustered concept and delete
    corresponding subgraphs, including wiki.
    """
    name_triples = [t for t in triples if t[1] == ':name']
    for name_t in name_triples:
        conc = name_t[0]
        name = name_t[2]

        # update concept name
        clusterized = NE_CLUSTER.setdefault(v2c[conc]._name, 'other')
        cluster_id = anon_ids[clusterized]
        new_conc_name = clusterized + '_' + str(cluster_id)
        anon_ids[clusterized] += 1
        v2c[conc] = Concept(new_conc_name)

        # get :op predicates, sorted by indexes
        op_tuples = [t for t in triples if t[0] == name and t[1] != ':instance-of']
        op_tuples = sorted(op_tuples, key=lambda x: x[1])

        # update mapping, removing quotes
        anon_map[new_conc_name] = ' '.join([str(op[2])[1:-1] for op in op_tuples])
        
        # update surface form
        # sometimes an :op is not aligned (implicit), therefore the if statement
        alignments = [graph.alignments()[t] for t in op_tuples if t in graph.alignments()]
        align_indexes = [a.split('.')[1] for a in alignments]
        # need to do this because some :ops are many-to-1
        indexes = []
        for a_index in align_indexes:
            if ',' in a_index:
                for i in a_index.split(','):
                    indexes.append(int(i))
            else:
                indexes.append(int(a_index))    
        #align_indexes = [int(alignments[t].split('.')[1]) for t in op_tuples if '~' in alignments[t]]
        for i, index in enumerate(indexes):
            if i == 0:
                anon_surf[index] = new_conc_name
            else:
                anon_surf[index] = ''
                
        # update the instance triple and remove other triples from graph
        for triple in triples:
            try:
                if triple[0] == name:
                    try:
                        output_triples.remove(triple)
                    except ValueError:
                        # Sometimes we have multiple NEs refering to the same graph,
                        # which means we already removed the graph in the first instance
                        pass
                elif triple[0] == conc and triple[1] == ':name' and triple[2] == name:
                    output_triples.remove(triple)
                elif triple[0] == conc and triple[1] == ':wiki':
                    output_triples.remove(triple)                
                elif triple[0] == conc and triple[1] == ':instance-of':
                    output_t_index = output_triples.index(triple)
                    output_triples[output_t_index]= (triple[0], triple[1], Concept(new_conc_name))
            except ValueError:
                # weird things
                #import ipdb; ipdb.set_trace()
                pass
                
                
    return output_triples, anon_ids, anon_map, anon_surf

###############

def anonymize_dates(graph, triples, output_triples, v2c, anon_ids, anon_map, anon_surf):
    """
    Anonymize dates. We use different tokens for days, months and years. The surface side
    also has different tokens for months and days if they are numbers or names.
    """
    date_vars = [t[0] for t in triples if t[2] == Concept('date-entity')]
    #date_vars = [v for v in v2c if v2c[v]._name == 'date-entity']
    for date_var in date_vars:

        # Get triples where this var appears
        date_triples = [t for t in triples if t[0] == date_var]

        # Anonymize days, months and years. Ignore others for now.
        date_triples = sorted(date_triples, key=lambda x: x[1])
        for date_t in date_triples:
            if date_t[1] in [':day', ':month', ':year']:

                # Get alignment for surface form
                try:
                    alignment = graph.alignments()[date_t]
                except:
                    # Alignment bug, ignore node and move to the next one
                    continue
                a_index = alignment.split('.')[1]
                indexes = []
                if ',' in a_index:
                    for i in a_index.split(','):
                        indexes.append(int(i))
                else:
                    indexes.append(int(a_index))

                stripped = date_t[1][1:]
                new_conc_name = stripped + '_' + str(anon_ids[stripped])
                anon_ids[stripped] += 1
                anon_map[new_conc_name] = str(date_t[2])


                # Update the corresponding triple
                output_t_index = output_triples.index(date_t)
                output_triples[output_t_index]= (date_t[0], date_t[1], Concept(new_conc_name))
                    
                # Update surface form
                for i, index in enumerate(indexes):
                    if i == 0:
                        curr_token = anon_surf[index]
                        if curr_token.isdigit() or 'day' in new_conc_name or 'year' in new_conc_name:
                            anon_surf[index] = new_conc_name + '_number'
                        else:
                            if '_' in curr_token:
                                # token was already preprocessed due to weird double alignment
                                try:
                                    next_index = indexes[i+1]
                                except IndexError:
                                    # revisiting previous token, ignore and move to the next one
                                    #import ipdb; ipdb.set_trace()
                                    print(curr_token)
                                    continue
                                next_token = anon_surf[next_index]
                                #print(next_token)
                                if next_token.isdigit() or 'day' in new_conc_name or 'year' in new_conc_name:
                                    anon_surf[next_index] = new_conc_name + '_number'
                                else:
                                    anon_surf[next_index] = new_conc_name + '_name'
                            else:
                                #print(curr_token)
                                anon_surf[index] = new_conc_name + '_name'
                    #else:
                    #    anon_surf[index] = ''

    return output_triples, anon_ids, anon_map, anon_surf

###############

def anonymize_quants(graph, triples, output_triples, v2c, anon_ids, anon_map, anon_surf):
    """
    Anonymize quantities. Similar procedure with NEs but without deleting subgraphs.
    There are three different cases that requires different treatments
    """
    quant_triples = [t for t in triples if t[1] == ':quant']
    for quant_t in quant_triples:
        conc = quant_t[0]
        quant = quant_t[2]

        # 1st case: :quant links to another concept. In this case we ignore it.
        # TODO: sometimes the quantity appears inside as an :opX predicate. We do
        # not deal with these cases here.
        if type(quant) == Var:
            continue

        # 2nd case: :quant links a non-quantity concept to a number. In this case
        # we replace the number with an anonymization token. To do this, we ...
        elif v2c[conc]._name not in QUANT_CLUSTER:
            if quant_t in graph.alignments():
                a_index = graph.alignments()[quant_t].split('.')[1]
                indexes = []
                if ',' in a_index:
                    for i in a_index.split(','):
                        indexes.append(int(i))
                else:
                    indexes.append(int(a_index))
            new_quant_name = 'quantity_' + str(anon_ids['quantity'])
            anon_ids['quantity'] += 1
            anon_map[new_quant_name] = quant._value
            try:
                output_t_index = output_triples.index(quant_t)
                output_triples[output_t_index][2]._value = new_quant_name
            except ValueError:
                # Constant already updated, can ignore
                pass
            if quant_t in graph.alignments():
                for i, index in enumerate(indexes):
                    if i == 0:
                        anon_surf[index] = new_quant_name
                    else:
                        anon_surf[index] = ''

        # 3rd case: :quant links a quantity concept to a number. In this case
        # we replace the *entire tuple* with an anonymization token.
        else:
            # Sometimes quantities are not aligned
            if quant_t in graph.alignments():
                a_index = graph.alignments()[quant_t].split('.')[1]
                indexes = []
                if ',' in a_index:
                    for i in a_index.split(','):
                        indexes.append(int(i))
                else:
                    indexes.append(int(a_index)) 
            new_quant_name = 'quantity_' + str(anon_ids['quantity'])
            anon_ids['quantity'] += 1
            anon_map[new_quant_name] = quant._value
            if quant_t in graph.alignments():
                for i, index in enumerate(indexes):
                    if i == 0:
                        anon_surf[index] = new_quant_name
                    else:
                        anon_surf[index] = ''

            # update concept name and remove triples
            v2c[conc] = Concept(new_quant_name)
            try:
                output_triples.remove(quant_t)
            except:
                for triple in output_triples:
                    if triple[0] == quant_t[0] and triple[1] == quant_t[1]:
                        output_triples.remove(triple)
            for triple in triples:
                if triple[0] == conc and triple[1] == ':instance-of':
                    output_t_index = output_triples.index(triple)
                    output_triples[output_t_index]= (triple[0], triple[1], Concept(new_quant_name))

    return output_triples, anon_ids, anon_map, anon_surf

##########################

def anonymize(graph, surf):
    
    # Get triples with :name predicate
    triples = graph.triples()
    new_graph = deepcopy(graph)
    #output_triples = deepcopy(triples)
    output_triples = new_graph.triples()
    v2c = new_graph.var2concept()
    
    anon_ids = {'person': 0,
                'organization': 0,
                'location': 0,
                'other': 0,
                'quantity': 0,
                'day': 0,
                'month': 0,
                'year': 0}
    anon_map = {}
    anon_surf = surf.split()
    output_triples, anon_ids, anon_map, anon_surf = anonymize_dates(graph,
                                                                    triples,
                                                                    output_triples,
                                                                    v2c,
                                                                    anon_ids,
                                                                    anon_map,
                                                                    anon_surf)
    output_triples, anon_ids, anon_map, anon_surf = anonymize_nes(graph,
                                                                  triples,
                                                                  output_triples,
                                                                  v2c,
                                                                  anon_ids,
                                                                  anon_map,
                                                                  anon_surf)
    output_triples, anon_ids, anon_map, anon_surf = anonymize_quants(graph,
                                                                     triples,
                                                                     output_triples,
                                                                     v2c,
                                                                     anon_ids,
                                                                     anon_map,
                                                                     anon_surf)

    anon_surf = ' '.join(anon_surf).lower().split() # remove extra spaces
    return output_triples, v2c, anon_surf, anon_map

##########################

def get_line_graph(graph, surf, new_tokens, anon=False, scope=False):
    triples = []
    nodes = {}
    rev_nodes = []
    uniq = 0
    nodes_to_print = []
    graph_triples = graph.triples()

    if anon:
        # preprocess triples and surface
        #import ipdb; ipdb.set_trace()
        graph_triples, v2c, anon_surf, anon_map = anonymize(graph, surf)
        anon_surf = ' '.join(anon_surf)
        #import ipdb; ipdb.set_trace()
        nodes_scope = print_simplified(graph_triples, v2c)
        nodes_scope = ' '.join(nodes_scope)
    else:
        graph_triples = graph.triples()
        v2c = graph.var2concept()
        anon_surf = surf
        anon_map = None
        nodes_scope = None
    for triple in graph_triples:
        src, edge, tgt = triple
        # ignore these nodes
        if edge == ':top':
            # store this to add scope later
            top_node = get_name(tgt, v2c)
            continue
        if edge == ':instance-of' or edge == ':wiki':
            continue
        # process nodes, populating the ids
        if src not in nodes:
            nodes[src] = len(nodes)
            rev_nodes.append(src)
            src_id = nodes[src]
            triples.append((src_id, src_id, 's'))
            nodes_to_print.append(get_name(src, v2c))
        edge_uniq = edge + '_' + str(uniq)
        uniq += 1
        nodes[edge_uniq] = len(nodes)
        rev_nodes.append(edge_uniq)
        edge_id = nodes[edge_uniq]
        triples.append((edge_id, edge_id, 's'))
        nodes_to_print.append(edge)
        if tgt not in nodes:
            nodes[tgt] = len(nodes)
            rev_nodes.append(tgt)
            tgt_id = nodes[tgt]
            triples.append((tgt_id, tgt_id, 's'))
            nodes_to_print.append(get_name(tgt, v2c))
        # process triples
        src_id = nodes[src]
        edge_id = nodes[edge_uniq]
        tgt_id = nodes[tgt]
        triples.append((src_id, edge_id, 'd'))
        triples.append((edge_id, src_id, 'r'))
        triples.append((edge_id, tgt_id, 'd'))
        triples.append((tgt_id, edge_id, 'r'))
    #print(nodes_to_print)
    #print(triples)
    if scope and nodes_to_print != []:
        nodes_to_print, triples = add_scope_markers(nodes_to_print, triples, top_node)
    #print(nodes_to_print)
    #print(triples)

    if nodes_to_print == []:
        # single node graph, first triple is ":top", second triple is the node
        triple = graph.triples()[1]
        nodes_to_print.append(get_name(triple[0], v2c))
        triples.append((0, 0, 's'))
    return nodes_to_print, triples, anon_surf, anon_map, nodes_scope

##########################

def add_scope_markers(nodes, triples, top_node):
    """
    Add scope markers to the graph representation.
    """
    triples = set(triples)
    # First, build adjacency list
    adj_list = build_adj_list(triples)

    # We need a visited node list because some graphs have cycles...
    visited = [False] * len(nodes)

    # Start from the top node
    top_node_id = nodes.index(top_node)
    #new_nodes.append(top_node)

    # Perform recursion
    stack = []
    add_marker(nodes, triples, adj_list, top_node_id, stack, visited)

    return nodes, sorted(list(triples))

##########################

def add_marker(nodes, triples, adj_list, node_id, stack, visited):
    """
    If node is a predicate, we add the scope markers if it
    has grandchildren.
    """
    #print(stack)
    node = nodes[node_id]
    visited[node_id] = True
    is_predicate = node.startswith(':') and len(node) > 2
    if is_predicate:
        print('plpl')
        #print(nodes[node_id])
        # check if it has grandchildren, predicates have only one child
        #import ipdb;ipdb.set_trace()
        try:
            child_id = list(adj_list[node_id])[0]
        except:
            import ipdb;ipdb.set_trace()
        has_gchild = has_child(child_id, adj_list, visited)
        #print(has_gchild)
        # only add markers if it has grandchildren
        if has_gchild:
            print('ok')
            # add markers
            nodes.append('(')
            open_id = len(nodes) - 1
            nodes.append(')')
            close_id = len(nodes) - 1
            stack.append(close_id) # stack node_id
            # update triples
            triples.add((open_id, open_id, 's'))
            triples.add((node_id, open_id, 'd'))
            triples.add((open_id, node_id, 'r'))
            triples.add((open_id, child_id, 'd'))
            triples.add((child_id, open_id, 'r'))

            triples.add((close_id, close_id, 's'))
            triples.add((node_id, close_id, 'd'))
            triples.add((close_id, node_id, 'r'))

            triples.remove((node_id, child_id, 'd'))
            triples.remove((child_id, node_id, 'r'))
        # proceed to next node, predicates have only one child
        add_marker(nodes, triples, adj_list, child_id, stack, visited)
        # finished processing predicate, can pop close scope symbol from stack
        if has_gchild:
            if len(stack) > 1:
                triples.add((stack[-2], stack[-1], 'd'))
                triples.add((stack[-1], stack[-2], 'r'))
            stack.pop()
    else:
        # node is not predicate
        # if it has children we simply iterate over them
        if has_child(node_id, adj_list, visited):
            for child_id in adj_list[node_id]:
                add_marker(nodes, triples, adj_list, child_id, stack, visited)
        # leaf node: complete the scopes, adding any additional edges according
        # to the stack
        else:
            #print(stack)
            if len(stack) > 0:
                curr_id = stack[-1]
                triples.add((curr_id, node_id, 'd'))
                triples.add((node_id, curr_id, 'r'))
            #for close_id in stack:
            #    triples.add((close_id, node_id, 'default'))
            #    triples.add((node_id, close_id, 'reverse'))
        
##########################

def has_child(node_id, adj_list, visited):
    """
    Check if a node has children. If it has but that node
    was already visited we assume it does not have any children.
    """
    if adj_list[node_id] == set():
        return False
    else:
        # still have to check for cycles...
        result = True
        for child in adj_list[node_id]:
            if visited[child]:
                result = False
    return result
    
##########################

def build_adj_list(triples):
    adj_list = defaultdict(set)
    for triple in triples:
        if triple[2] == 'default':
            adj_list[triple[0]].add(triple[1])
    #print(adj_list)
    return adj_list
                                       

##########################

def print_simplified(graph_triples, v2c):
    """
    Given a modified graph, prints the linearised, simplified version with scope markers.
    Taken from AMR code.
    """       
    s = []
    stack = []
    instance_fulfilled = None
    concept_stack_depth = {None: 0} # size of the stack when the :instance-of triple was encountered for the variable

    # Traverse the graph and build initial string
    for h, r, d in graph_triples + [(None,None,None)]:
        if r==':top':
            s.append('(')
            s.append(get_name(d, v2c))
            stack.append((h, r, d))
            instance_fulfilled = False
        elif r==':instance-of':
            instance_fulfilled = True
            concept_stack_depth[h] = len(stack)
        else:
            while len(stack)>concept_stack_depth[h]:
                h2, r2, d2 = stack.pop()
                if instance_fulfilled is False:
                    s.pop()
                    s.pop()
                    s.append(get_name(d2, v2c))
                else:
                    s.append(')')
                instance_fulfilled = None
            if d is not None:
                s.append(r)
                s.append('(')
                s.append(get_name(d, v2c))
                stack.append((h, r, d))
                instance_fulfilled = False

    #import ipdb; ipdb.set_trace()
    # Remove extra parenthesis when there's one token only between them
    final_s = []
    skip = False
    for i, token in enumerate(s[:-2]):
        if token == '(':
            if s[i+2] == ')':
                skip = True
            if not skip:
                final_s.append(token)
        elif token == ')':
            if not skip:
                final_s.append(token)
            skip = False
        else:
            final_s.append(token)
    # remove extra set of parenthesis
    final_s.append(s[-2])
    if len(s) == 3:
        # corner case: single node with two parenthesis
        return s[1:2]
    #print(s)
    return final_s[1:]
##########################

def main(args):

    # First, let's read the graphs and surface forms
    with open(args.input_amr) as f:
        amrs = f.readlines()
    with open(args.input_surface) as f:
        surfs = f.readlines()

    if args.triples_output is not None:
        triples_out = open(args.triples_output, 'w')
        
    # Iterate
    anon_surfs = []
    anon_maps = []
    nodes_list_scope = []
    i = 0
    cont_error = 0
    total_g = 0
    with open(args.output, 'w') as out, open(args.output_surface, 'w') as surf_out:
        for amr, surf in zip(amrs, surfs):
            try:
                total_g += 1
                graph = AMR(amr, surf.split())

            except:
                cont_error += 1
                print('error', cont_error, '/', total_g)
                continue
            
            # Get variable: concept map for reentrancies
            #v2c = graph.var2concept()

            if args.mode == 'LIN':
                # Linearisation mode for seq2seq

                v2c = graph.var2concept()

                tokens = amr.split()
                try:
                    new_tokens = simplify(tokens, v2c)
                except:
                    continue
                out.write(' '.join(new_tokens) + '\n')

            elif args.mode == 'GRAPH':
                # Triples mode for graph2seq
                #import ipdb; ipdb.set_trace()
                # Get concepts and generate IDs
                v_ids, rev_v_ids = get_nodes2(graph)

                v2c = graph.var2concept()

                # Triples
                triples = get_triples(graph, v_ids, rev_v_ids, v2c)

                # Print concepts/constants and triples
                #cs = [get_name(c) for c in rev_c_ids]

                cs = [get_name(v, v2c) for v in rev_v_ids]
                out.write(' '.join(cs) + '\n')
                triples_out.write(' '.join(['(' + ','.join(adj) + ')' for adj in triples]) + '\n')

            elif args.mode == 'LINE_GRAPH':
                # Similar to GRAPH, but with edges as extra nodes
                #import ipdb; ipdb.set_trace()
                #print(i)
                i += 1
                #if i == 98:
                #    import ipdb; ipdb.set_trace()

                v2c = graph.var2concept()

                tokens = amr.split()
                new_tokens = simplify(tokens, v2c)

                nodes, triples, anon_surf, anon_map, nodes_scope = get_line_graph(graph, surf, new_tokens, anon=args.anon, scope=args.scope)
                out.write(' '.join(nodes) + '\n')
                triples_out.write(' '.join(['(%d,%d,%s)' % adj for adj in triples]) + '\n')
                #surf = ' '.join(new_surf)
                anon_surfs.append(anon_surf)
                anon_maps.append(json.dumps(anon_map))
                nodes_list_scope.append(nodes_scope)
                
            # Process the surface form
            #surf_out.write(surf.lower())
            surf_out.write(surf)
    if args.anon:
        with open(args.anon_surface, 'w') as f:
            for anon_surf in anon_surfs:
                f.write(anon_surf + '\n')
        with open(args.map_output, 'w') as f:
            for anon_map in anon_maps:
                f.write(anon_map + '\n')
        with open(args.nodes_scope, 'w') as f:
            for nodes_scope in nodes_list_scope:
                f.write(nodes_scope + '\n')

###########################
            
if __name__ == "__main__":
    
    # Parse input
    parser = argparse.ArgumentParser(description="Preprocess AMR into linearised forms with multiple preprocessing steps (based on Konstas et al. ACL 2017)")
    parser.add_argument('input_amr', type=str, help='input AMR file')
    parser.add_argument('input_surface', type=str, help='input surface file')
    parser.add_argument('output', type=str, help='output file, either AMR or concept list')
    parser.add_argument('output_surface', type=str, help='output surface file')
    parser.add_argument('--mode', type=str, default='GRAPH', help='preprocessing mode',
                        choices=['GRAPH','LIN','LINE_GRAPH'])
    parser.add_argument('--anon', action='store_true', help='anonymise NEs and dates')
    parser.add_argument('--scope', action='store_true', help='add scope markers to graph')
    parser.add_argument('--add-reverse', action='store_true', help='whether to add reverse edges in the graph output')
    parser.add_argument('--triples-output', type=str, default=None, help='triples output for graph2seq')
    parser.add_argument('--map-output', type=str, default=None, help='mapping output file, if using anonymisation')
    parser.add_argument('--anon-surface', type=str, default=None, help='anonymized surface output file, if using anonymisation')
    parser.add_argument('--nodes-scope', type=str, default=None, help='anonymized AMR graph file, with scope marking, used in baseline seq2seq models')

    args = parser.parse_args()

    assert (args.triples_output is not None) or (args.mode != 'GRAPH'), "Need triples output for graph mode"
    assert (args.map_output is not None) or (not args.anon), "Need map output for anon mode"
    assert (args.anon_surface is not None) or (not args.anon), "Need anonymized surface output for anon mode"

    SENSE_PATTERN = re.compile('-[0-9][0-9]$')
    
    main(args)
