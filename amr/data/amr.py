#!/usr/bin/env python2.7
#coding=utf-8
'''
Parser for Abstract Meaning Represention (AMR) annotations in Penman format.
A *parsing expression grammar* (PEG) for AMRs is specified in amr.peg
and the AST is built by the Parsimonious library (https://github.com/erikrose/parsimonious).
The resulting graph is represented with the AMR class.
When called directly, this script runs some cursory unit tests.

If the AMR has ISI-style inline alignments, those are stored in the AMR object as well.

TODO: Include the smatch evaluation code
(released at http://amr.isi.edu/evaluation.html under the MIT License).

@author: Nathan Schneider (nschneid@inf.ed.ac.uk)
@since: 2015-05-05
'''
from __future__ import print_function

import re
from collections import defaultdict, Counter

from nltk.parse import DependencyGraph
from parsimonious.exceptions import ParseError
from parsimonious.grammar import Grammar
import os

def clean_grammar_file(s):
    return re.sub('\n[ \t]+', ' ', re.sub(r'#.*','',s.replace('\t',' ').replace('`','_backtick')))

with open(os.path.dirname(os.path.realpath(__file__)) + '/amr.peg') as inF:
    grammar = Grammar(clean_grammar_file(inF.read()))


class Var(object):
    def __init__(self, name):
        self._name = name
    def is_constant(self):
        return False
    def __repr__(self):
        return 'Var('+self._name+')'
    def __str__(self):
        return self._name
    def __call__(self, align_key='', append=False):
        return self._name+(align_key if append else '')
    def __eq__(self, that):
        return type(that)==type(self) and self._name==that._name
    def __hash__(self):
        return hash(repr(self))

class Concept(object):
    RE_FRAME_NUM = re.compile(r'-\d\d$')
    def __init__(self, name):
        self._name = name
    def is_constant(self):
        return False
    def is_frame(self):
        return self.RE_FRAME_NUM.search(self._name) is not None
    def __repr__(self):
        return 'Concept('+self._name+')'
    def __str__(self, align_key='', **kwargs):
        return self._name+align_key
    def __call__(self, *args, **kwargs):
        return self.__str__(*args, **kwargs)
    def __eq__(self, that):
        return type(that)==type(self) and self._name==that._name
    def __hash__(self):
        return hash(repr(self))

class AMRConstant(object):
    def __init__(self, value):
        self._value = value
    def is_constant(self):
        return True
    def is_frame(self):
        return False
    def __repr__(self):
        return 'Const('+self._value+')'
    def __str__(self, align_key='', **kwargs):
        return self._value+align_key
    def __call__(self, *args, **kwargs):
        return self.__str__(*args, **kwargs)
    def __eq__(self, that):
        return type(that)==type(self) and self._value==that._value
    def __hash__(self):
        return hash(repr(self))

class AMRString(AMRConstant):
    def __str__(self, align_key='', **kwargs):
        return '"'+self._value+'"'+align_key
    def __repr__(self):
        return '"'+self._value+'"'

class AMRNumber(AMRConstant):
    def __repr__(self):
        return 'Num('+self._value+')'


class AMRError(Exception):
    pass

class AMRSyntaxError(Exception):
    pass

class AMR(DependencyGraph):
    '''
    An AMR annotation. Constructor parses the Penman notation.
    Does not currently provide functionality for manipulating the AMR structure,
    but subclassing from DependencyGraph does provide the contains_cycle() method.

    >>> s = """                                    \
    (b / business :polarity -                      \
       :ARG1-of (r / resemble-01                   \
                   :ARG2 (b2 / business            \
                             :mod (s / show-04)))) \
    """
    >>> a = AMR(s)
    >>> a
    (b / business :polarity -
        :ARG1-of (r / resemble-01
            :ARG2 (b2 / business
                :mod (s / show-04))))
    >>> a.reentrancies()
    Counter()
    >>> a.contains_cycle()
    False

    >>> a = AMR("(h / hug-01 :ARG0 (y / you) :ARG1 y :mode imperative)")
    >>> a
    (h / hug-01
        :ARG0 (y / you)
        :ARG1 y
        :mode imperative)
    >>> a.reentrancies()
    Counter({Var(y): 1})
    >>> a.contains_cycle()
    False

    >>> a = AMR('(h / hug-01 :ARG1 (p / person :ARG0-of h))')
    >>> a
    (h / hug-01
        :ARG1 (p / person
            :ARG0-of h))
    >>> a.reentrancies()
    Counter({Var(h): 1})
    >>> a.triples()     #doctest:+NORMALIZE_WHITESPACE
    [(Var(TOP), ':top', Var(h)), (Var(h), ':instance-of', Concept(hug-01)),
     (Var(h), ':ARG1', Var(p)), (Var(p), ':instance-of', Concept(person)),
     (Var(p), ':ARG0-of', Var(h))]
    >>> sorted(a.contains_cycle(), key=str)
    [Var(h), Var(p)]

    >>> a = AMR('(h / hug-01 :ARG0 (y / you) :mode imperative \
    :ARG1 (p / person :ARG0-of (w / want-01 :ARG1 h)))')
    >>> # Hug someone who wants you to!
    >>> sorted(a.contains_cycle(), key=str)
    [Var(h), Var(p), Var(w)]

    >>> a = AMR('(w / wizard    \
    :name (n / name :op1 "Albus" :op2 "Percival" :op3 "Wulfric" :op4 "Brian" :op5 "Dumbledore"))')
    >>> a
    (w / wizard
        :name (n / name
            :op1 "Albus"
            :op2 "Percival"
            :op3 "Wulfric"
            :op4 "Brian"
            :op5 "Dumbledore"))

    # with automatic alignments
    # at_0 a_1 glance_2 i_3 can_4 distinguish_5 china_6 from_7 arizona_8 ._9
    >>> a = AMR('(p / possible~e.4 :domain~e.1 (d / distinguish-01~e.5 :arg0 (i / i~e.3) \
    :arg1 (c / country :wiki~e.7 "china"~e.6 :name (n / name :op1 "china"~e.6))          \
    :arg2 (s / state :wiki~e.7 "arizona"~e.8 :name (n2 / name :op1 "arizona"~e.8))       \
    :manner~e.0 (g / glance-01~e.2 :arg0 i)))')
    >>> a
    (p / possible~e.4
        :domain~e.1 (d / distinguish-01~e.5
            :arg0 (i / i~e.3)
            :arg1 (c / country
                :wiki~e.7 "china"~e.6
                :name (n / name
                    :op1 "china"~e.6))
            :arg2 (s / state
                :wiki~e.7 "arizona"~e.8
                :name (n2 / name
                    :op1 "arizona"~e.8))
            :manner~e.0 (g / glance-01~e.2
                :arg0 i)))
    >>> sorted(a.alignments().items(), key=str)  #doctest:+NORMALIZE_WHITESPACE
    [((Var(c), ':wiki', "china"), 'e.6'),
     ((Var(d), ':instance-of', Concept(distinguish-01)), 'e.5'),
     ((Var(g), ':instance-of', Concept(glance-01)), 'e.2'),
     ((Var(i), ':instance-of', Concept(i)), 'e.3'),
     ((Var(n), ':op1', "china"), 'e.6'),
     ((Var(n2), ':op1', "arizona"), 'e.8'),
     ((Var(p), ':instance-of', Concept(possible)), 'e.4'),
     ((Var(s), ':wiki', "arizona"), 'e.8')]
    >>> sorted(a.role_alignments().items(), key=str)  #doctest:+NORMALIZE_WHITESPACE
    [((Var(c), ':wiki', "china"), 'e.7'),
     ((Var(d), ':manner', Var(g)), 'e.0'),
     ((Var(p), ':domain', Var(d)), 'e.1'),
     ((Var(s), ':wiki', "arizona"), 'e.7')]
    >>> print(a(alignments=False))
    (p / possible
        :domain (d / distinguish-01
            :arg0 (i / i)
            :arg1 (c / country
                :wiki "china"
                :name (n / name
                    :op1 "china"))
            :arg2 (s / state
                :wiki "arizona"
                :name (n2 / name
                    :op1 "arizona"))
            :manner (g / glance-01
                :arg0 i)))

    >>> a = AMR("(h / hug-01~e.2 :polarity~e.1 -~e.1 :ARG0 (y / you~e.3) :ARG1 y \
                 :mode~e.0 imperative~e.5 :result (s / silly-01~e.4 :ARG1 y))", \
                "Do n't hug yourself silly !".split())
    >>> a
    (h / hug-01~e.2[hug] :polarity~e.1[n't] -~e.1[n't]
        :ARG0 (y / you~e.3[yourself])
        :ARG1 y
        :mode~e.0[Do] imperative~e.5[!]
        :result (s / silly-01~e.4[silly]
            :ARG1 y))

    >>> a = AMR("(h / hug-01~e.3 :ARG0 (a / and~e.1 :op1 (y / you~e.0) :op2 (i / i~e.2)) \
                 :ARG1 (a2 / and~e.5 :op1 (s / she~e.4) :op2 (h2 / he~e.6)))", \
                "You and I hug her and him".split())
    >>> a
    (h / hug-01~e.3[hug]
        :ARG0 (a / and~e.1[and]
            :op1 (y / you~e.0[You])
            :op2 (i / i~e.2[I]))
        :ARG1 (a2 / and~e.5[and]
            :op1 (s / she~e.4[her])
            :op2 (h2 / he~e.6[him])))

    >>> a = AMR("(a / and~e.5 :op1 (e / eat-01~e.2 :ARG0 (c / cat~e.1,8 :poss~e.0,8 (i / i~e.0,8)) \
                 :time~e.3 (d / dawn~e.4)) \
                 :op2 (e2 / eat-01~e.2 :ARG0 (c2 / cat~e.1,6 :poss~e.6 (y / you~e.6)) \
                 :time~e.7 (a2 / after~e.7 :op1 e~e.9)))", \
                "my cat eats at dawn and yours after mine does".split())
    >>> a
    (a / and~e.5[and]
        :op1 (e / eat-01~e.2[eats]
            :ARG0 (c / cat~e.1,8[cat,mine]
                :poss~e.0,8[my,mine] (i / i~e.0,8[my,mine]))
            :time~e.3[at] (d / dawn~e.4[dawn]))
        :op2 (e2 / eat-01~e.2[eats]
            :ARG0 (c2 / cat~e.1,6[cat,yours]
                :poss~e.6[yours] (y / you~e.6[yours]))
            :time~e.7[after] (a2 / after~e.7[after]
                :op1 e~e.9[does])))

    >>> a = AMR('(r / reduce-01~e.8 :ARG0 (t / treat-04~e.0 :ARG1~e.1 (c / cell-line~e.3,4 \
                :mod (d2 / disease :name (n3 / name :op1 "CRC"~e.2))) :ARG2~e.5 (s / small-molecule \
                :name (n / name :op1 "U0126"~e.6))) :ARG1 (l / level~e.11 :quant-of (n6 / nucleic-acid \
                :name (n4 / name :op1 "mRNA"~e.10) :ARG0-of (e2 / encode-01 :ARG1 (p / protein \
                :name (n5 / name :op1 "serpinE2"~e.9))))) :manner~e.7 (m / marked~e.7) \
                :ARG0-of (i / indicate-01~e.13 :ARG1~e.14 (l2 / likely-01~e.19 \
                :ARG1 (d / depend-01~e.20 :ARG0 (e3 / express-03~e.15 :ARG2 p~e.17) \
                :ARG1~e.21 (a / activity-06~e.23 :ARG0 (e / enzyme \
                :name (n2 / name :op1 "ERK"~e.22)))))))', \
                "Treatment of CRC cell lines with U0126 markedly reduced serpinE2 mRNA levels , indicating that expression of serpinE2 is likely dependent of ERK activity".split())
    >>> a
    (r / reduce-01~e.8[reduced]
        :ARG0 (t / treat-04~e.0[Treatment]
            :ARG1~e.1[of] (c / cell-line~e.3,4[cell,lines]
                :mod (d2 / disease
                    :name (n3 / name
                        :op1 "CRC"~e.2[CRC])))
            :ARG2~e.5[with] (s / small-molecule
                :name (n / name
                    :op1 "U0126"~e.6[U0126])))
        :ARG1 (l / level~e.11[levels]
            :quant-of (n6 / nucleic-acid
                :name (n4 / name
                    :op1 "mRNA"~e.10[mRNA])
                :ARG0-of (e2 / encode-01
                    :ARG1 (p / protein
                        :name (n5 / name
                            :op1 "serpinE2"~e.9[serpinE2])))))
        :manner~e.7[markedly] (m / marked~e.7[markedly])
        :ARG0-of (i / indicate-01~e.13[indicating]
            :ARG1~e.14[that] (l2 / likely-01~e.19[likely]
                :ARG1 (d / depend-01~e.20[dependent]
                    :ARG0 (e3 / express-03~e.15[expression]
                        :ARG2 p~e.17[serpinE2])
                    :ARG1~e.21[of] (a / activity-06~e.23[activity]
                        :ARG0 (e / enzyme
                            :name (n2 / name
                                :op1 "ERK"~e.22[ERK])))))))
    '''

    def __init__(self, anno, tokens=None):
        '''
        Given a Penman annotation string for a single rooted AMR, construct the data structure.
        Triples are stored internally in an order that preserves the layout of the
        annotation (even though this doesn't matter for the "pure" graph).
        (Whitespace is normalized, however.)

        Will raise an AMRSyntaxError if notationally malformed, or an AMRError if
        there is not a 1-to-1 mapping between (unique) variables and concepts.
        Will not check details such as the appropriateness of relation/role names
        or constants. Does not currently read or store metadata about the AMR.
        '''
        self._v2c = {}
        self._triples = []
        self._constants = set()
        self._alignments = {}
        self._role_alignments = {}
        self._tokens = tokens

        self.nodes = defaultdict(lambda: {'address': None,
                                          'type': None,
                                          'head': None,
                                          'rel': None,
                                          'word': None,
                                          'deps': []})
        # Emulate the DependencyGraph (superclass) data structures somewhat.
        # There are some differences, e.g., in AMR it is possible for a node to have
        # multiple dependents with the same relation; so here, 'deps' is simply a list
        # of dependents, not a mapping from relation types to dependents.
        # In typical depenency graphs, 'word' is a word in the sentence
        # and 'address' is its index; here, both point to the object representing
        # the node's AMR variable, concept, or constant.

        TOP = Var('TOP')
        self.nodes[TOP]['address'] = self.nodes[TOP]['word'] = TOP
        self.nodes[TOP]['type'] = 'TOP'
        if anno:
            self._anno = anno
            msg = ''
            try:
                p = grammar.parse(anno)
            except ParseError as e:
                msg += '\n' + str(e)
                p = None
            if p is None:
                raise AMRSyntaxError('Well-formedness error in annotation:\n'+anno.strip()+msg)
            self._analyze(p)

    def triples(self, head=None, rel=None, dep=None, normalize_inverses=False, normalize_mod=False):  # overrides superclass implementation
        '''
        Returns a list of head-relation-dependent triples in the AMR.
        Can be filtered by specifying a value (or iterable of allowed values) for:
          - 'head': head variable(s)
          - 'rel': relation label(s) (string(s) starting with ":"), or "core" for all :ARGx roles,
            or "non-core" for all other relations. See also role_triples().
          - 'dep': dependent variable(s)/concept(s)/constant(s)
        Boolean options:
          - 'normalize_inverses': transform (h,':REL-of',d) relations to (d,':REL',h)
          - 'normalize_mod': transform ':mod' to ':domain-of' (before normalizing inverses,
            if applicable)

        >>> a = AMR('(h / hug-01 :ARG1 (p / person :ARG0-of h))')
        >>> a.triples(head=Var('h'))
        [(Var(h), ':instance-of', Concept(hug-01)), (Var(h), ':ARG1', Var(p))]
        >>> a.triples(head=Var('p'), rel=':instance-of')
        [(Var(p), ':instance-of', Concept(person))]
        >>> a.triples(rel=[':top',':instance-of'])
        [(Var(TOP), ':top', Var(h)), (Var(h), ':instance-of', Concept(hug-01)), (Var(p), ':instance-of', Concept(person))]
        >>> a.triples(rel='core')
        [(Var(h), ':ARG1', Var(p)), (Var(p), ':ARG0-of', Var(h))]
        >>> a.triples(rel='core', normalize_inverses=True)
        [(Var(h), ':ARG1', Var(p)), (Var(h), ':ARG0', Var(p))]
        '''
        tt = (trip for trip in self._triples)
        if normalize_mod:
            tt = ((h,':domain-of',d) if r==':mod' else (h,r,d) for h,r,d in tt)
        if normalize_inverses:
            tt = ((y,r[:-3],x) if r.endswith('-of') else (x,r,y) for x,r,y in tt)
        if head:
            tt = ((h,r,d) for h,r,d in tt if h in (head if hasattr(head,'__iter__') else (head,)))
        if rel:
            if rel=='core':
                tt = ((h,r,d) for h,r,d in tt if r.startswith(':ARG'))
            elif rel=='non-core':
                tt = ((h,r,d) for h,r,d in tt if not r.startswith(':ARG'))
            else:
                tt = ((h,r,d) for h,r,d in tt if r in (rel if hasattr(rel,'__iter__') else (rel)))
        if dep:
            tt = ((h,r,d) for h,r,d in tt if d in (dep if hasattr(dep,'__iter__') else (dep,)))
        return list(tt)

    def role_triples(self, **kwargs):
        '''
        Same as triples(), but limited to roles (excludes :instance-of, :instance, and :top relations).

        >>> a = AMR('(h / hug-01 :ARG1 (p / person :ARG0-of h))')
        >>> a.role_triples()
        [(Var(h), ':ARG1', Var(p)), (Var(p), ':ARG0-of', Var(h))]
        >>> a.role_triples(head=Var('h'))
        [(Var(h), ':ARG1', Var(p))]
        '''
        tt = [(h,r,d) for h,r,d in self.triples(**kwargs) if r not in (':instance',':instance-of',':top')]
        return tt

    def constants(self):
        return self._constants

    def concept(self, variable):
        return self._v2c[variable]

    def concepts(self):
        return self._v2c.items()

    def var2concept(self):
        return dict(self._v2c)

    def alignments(self):
        return dict(self._alignments)

    def role_alignments(self):
        return dict(self._role_alignments)

    def tokens(self):
        return self._tokens

    def reentrancies(self):
        '''Counts the number of times each variable is mentioned in the annotation
        beyond the one where it receives a concept. Non-reentrant variables are not
        included in the output.'''
        c = defaultdict(int)
        for h, r, d in self.triples():
            if isinstance(d, Var):
                c[d] += 1
            elif isinstance(d, Concept):
                c[h] -= 1
        return Counter(c) + Counter()   # the addition removes non-positive entries

    #def __repr__(self):
    #    return 'AMR(v2c='+repr(self._v2c)+', triples='+repr(self._triples)+', constants='+repr(self._constants)+')'

    def __call__(self, *args, **kwargs):
        return self.__str__(*args, **kwargs)

    def __str__(self, alignments=True, tokens=True, compressed=False, indent=' '*4):
        '''
        Assumes triples are stored in a sensible order (reflecting how they are encountered in a valid AMR).

        >>> a = AMR('(p / person :ARG0-of (h / hug-01 :ARG0 p :ARG1 p) :mod (s / strange))')
        >>> # people who hug themselves and are strange
        >>> print(str(a))
        (p / person
            :ARG0-of (h / hug-01
                :ARG0 p
                :ARG1 p)
            :mod (s / strange))
        '''
        def alignment_str(align_key):
            s = '~' + align_key
            if tokens:  # alignment key is like "e.10" (single token offset) or "e.10,11" (multiple)
                s += '[' + ','.join(tokens[int(woffset)] for woffset in align_key.split('.')[1].split(',')) + ']'
            return s
        
        s = ''
        stack = []
        instance_fulfilled = None
        align = role_align = {}
        if alignments:
            if tokens:
                tokens = self.tokens()
            align = {k: alignment_str(align_key) for k,align_key in self._alignments.items()}
            role_align = {k: alignment_str(align_key) for k,align_key in self._role_alignments.items()}
        concept_stack_depth = {None: 0} # size of the stack when the :instance-of triple was encountered for the variable
        for h, r, d in self.triples()+[(None,None,None)]:
            align_key = align.get((h, r, d), '')
            role_align_key = role_align.get((h, r, d), '')
            if r==':top':
                s += '(' + d()
                stack.append((h, r, d))
                instance_fulfilled = False
            elif r==':instance-of':
                s += ' / ' + d(align_key)
                instance_fulfilled = True
                concept_stack_depth[h] = len(stack)
            elif h==stack[-1][2] and r==':polarity':   # polarity gets to be on the same line as the concept
                s += ' ' + r + role_align_key + ' ' + d(align_key)
            else:
                while len(stack)>concept_stack_depth[h]:
                    h2, r2, d2 = stack.pop()
                    if instance_fulfilled is False:
                        # just a variable or constant with no concept hanging off of it
                        # so we have an extra paren to get rid of
                        align_key2 = align.get((h2, r2, d2), '')
                        s = s[:-len(d2(align_key2)) - 1] + d2(align_key2, append=not instance_fulfilled)
                    else:
                        s += ')'
                    instance_fulfilled = None
                if d is not None:
                    s += '\n' + indent*len(stack) + r + role_align_key + ' (' + d(align_key)
                    stack.append((h, r, d))
                    instance_fulfilled = False
        return s

    def __repr__(self):
        return str(self)

    def _analyze(self, p):
        '''Analyze the AST produced by parsimonious.'''
        v2c = {}    # variable -> concept
        allvars = set() # all vars mentioned in the AMR
        elts = {}  # for interning variables, concepts, constants, etc.
        consts = set()  # all constants used in the AMR

        def intern_elt(x):
            return elts.setdefault(x, x)

        def walk(n):    # (v / concept...)
            triples = []
            deps = []
            v = None
            for ch in n.children:
                t = ch.expr_name
                if t=='BAREVAR':
                    v = intern_elt(Var(ch.text))
                    allvars.add(v)
                elif t=='CONCEPT':
                    assert v is not None
                    if v in v2c:
                        raise AMRError('Variable has multiple concepts: '+str(v)+'\n'+self._anno)
                    concept_node, alignment_node = ch.children
                    c = intern_elt(Concept(concept_node.text))
                    v2c[v] = c
                    self.add_node({'address': c, 'word': c, 'type': 'CONCEPT',
                                   'rel': ':instance-of', 'head': v, 'deps': []})
                    deps.append(c)
                    triple = (v, ':instance-of', c)
                    triples.append(triple)
                    if alignment_node.text:
                        self._alignments[triple] = alignment_node.text[1:]
                elif t=='' and ch.children:
                    for ch2 in ch.children:
                        _part, RELpart, _part, Ypart = ch2.children
                        rel, relalignment = RELpart.children
                        rel = rel.text
                        assert rel is not None
                        assert len(Ypart.children)==1
                        q = Ypart.children[0]
                        tq = q.expr_name
                        n2 = None
                        triples2 = []
                        deps2 = []
                        qalign = None
                        if tq=='X':
                            n2, triples2, deps2 = walk(q)
                        elif tq=='NAMEDCONST':
                            qleft, qalign = q.children
                            n2 = intern_elt(AMRConstant(qleft.text))
                            consts.add(n2)
                        elif tq=='VAR':
                            qleft, qalign = q.children
                            n2 = intern_elt(Var(qleft.text))
                            allvars.add(n2)
                        elif tq=='STR':
                            quote1, qstr, quote2, qalign = q.children
                            n2 = intern_elt(AMRString(qstr.text))
                            consts.add(n2)
                        elif tq=='NUM':
                            qleft, qalign = q.children
                            n2 = intern_elt(AMRNumber(qleft.text))
                            consts.add(n2)
                        assert n2 is not None
                        self.add_node({'address': n2, 'word': n2, 'type': tq,
                                       'rel': rel, 'head': v})
                        self.nodes[n2]['deps'].extend(deps2)
                        deps.append(n2)
                        triple = (v, rel, n2)
                        triples.append(triple)
                        if qalign and qalign.text:
                            self._alignments[triple] = qalign.text[1:]
                        if relalignment.text:
                            self._role_alignments[triple] = relalignment.text[1:]
                        triples.extend(triples2)
            return v, triples, deps

        assert p.expr_name=='ALL'

        n = None
        for ch in p.children:
            if ch.expr_name=='X':
                assert n is None    # only one top-level node per AMR
                n, triples, deps = walk(ch)
                self.add_node({'address': n, 'word': n, 'type': 'VAR',
                               'rel': ':top', 'head': intern_elt(Var('TOP'))})
                self.nodes[n]['deps'].extend(deps)
                triples = [(intern_elt(Var('TOP')), ':top', n)] + triples

        if allvars - set(v2c.keys()):
            raise AMRError('Unbound variable(s): ' + ','.join(map(str,allvars - set(v2c.keys())))+'\n'+self._anno)

        # All is well, so store the resulting data
        self._v2c = v2c
        self._triples = triples
        self._constants = consts



good_tests = [
'''(h / hot)''',
'''(h / hot :mode expressive)''',
'''(h / hot :mode "expressive")''',
'''(h / hot :domain h)''',
'''(h / hot :domain h~e.0)''',
'''  (  h  /  hot   :mode  expressive  )   ''',
'''  (  h
/
hot
:mode
expressive
)   ''',
'''(h / hot
     :mode expressive
     :mod (e / emoticon
          :value ":)"))''',
'''(n / name :op1 "Washington")''',
'''(s / state :name (n / name :op1 "Washington"))''',
'''(s / state :name (n / name :op1 "Ohio"))
''',
'''(s / state :name (n / name :op1 "Washington")
    )
''',
'''(s / state
:name (n / name :op1 "Washington"))''',
'''(s / state :name (n / name :op1 "Washington")
    :wiki "http://en.wikipedia.org/wiki/Washington_(state)")
''',
'''(f / film :name (n / name :op1 "Victor/Victoria")
    :wiki "http://en.wikipedia.org/wiki/Victor/Victoria_(1995_film)")
''',
'''(g / go-01 :polarity -
      :ARG0 (b / boy))''',
'''(a / and
:op1 (l / love-01 :ARG0 (b / boy) :ARG1 (g / girl))
:op2 (l2 / love-01 :ARG0 g :ARG1 b)
)''',
'''(d / date-entity :month 2 :day 29 :year 2012 :time "16:30" :timezone "PST"
           :weekday (w / wednesday))''',
'''(a / and :op1 (d / day :quant 40) :op2 (n / night :quant 40))''',
'''(e / earthquake
           :location (c / country-region :wiki "Tōhoku_region"
                 :name (n / name :op1 "Tohoku"))
           :quant (s / seismic-quantity :quant 9.3)
           :time (d / date-entity :year 23 :era "Heisei"
                 :calendar (c2 / country :wiki "Japan"
                       :name (n2 / name :op1 "Japan"))))''',
''' (d / date-entity :polite +
           :time (a / amr-unknown))'''
]

sembad_tests = [    # not a syntax error, but malformed in terms of variables
'''(h / hot :mod (h / hot))''',
'''(h / hot :mod q)'''
]

bad_tests = [
'''h / hot :mode expressive''',
'''(h~e.0 / hot :domain h)''',
'''(hot :mode expressive)''',
'''(h/hot :mode expressive)''',
'''(h / hot :mode )''',
'''(h / hot :mode expressive''',
'''(h / hot :mode (expressive))''',
'''(h / hot :mode (e / ))''',
'''((h / hot :mode expressive)''',
'''(h / hot :mode expressive)

x''',
'''(s / state :name (n / name :op1 "  Washington  "))''',
'''(s / state :name (n / name :op1 "Washington")

    )
''',
'''(s / state

:name (n / name :op1 "Washington"))''',
'''(e / earthquake
           :location (c / country-region :wiki "Tōhoku_region"
                 :name (n / name :op1 "Tohoku"))
           :quant (s / seismic-quantity :quant 9.3.1)
           :time (d / date-entity :year 23 :era "Heisei"
                 :calendar (c2 / country :wiki "Japan"
                       :name (n2 / name :op1 "Japan"))))'''
]

def test():
    for good in good_tests:
        try:
            AMR(good)
        except AMRSyntaxError:
            print('Should be valid!')
            print(good)
        except AMRError:
            print('Should be valid!')
            print(good)

    for sembad in sembad_tests:
        try:
            AMR(sembad)
        except AMRSyntaxError:
            print('Parse should work!')
            print(sembad)
        except AMRError:
            pass    # should trigger exception
        else:
            print('Should be invalid!')
            print(sembad)

    for bad in bad_tests:
        try:
            AMR(bad)
        except AMRSyntaxError:
            pass
        else:
            print('Parse should fail!')
            print(bad)

if __name__=='__main__':
    test()
    import doctest
    doctest.testmod()
