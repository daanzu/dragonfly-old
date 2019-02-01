#
# This file is part of Dragonfly.
# (c) Copyright 2007, 2008 by Christo Butcher
# Licensed under the LGPL.
#
#   Dragonfly is free software: you can redistribute it and/or modify it 
#   under the terms of the GNU Lesser General Public License as published 
#   by the Free Software Foundation, either version 3 of the License, or 
#   (at your option) any later version.
#
#   Dragonfly is distributed in the hope that it will be useful, but 
#   WITHOUT ANY WARRANTY; without even the implied warranty of 
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
#   Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public 
#   License along with Dragonfly.  If not, see 
#   <http://www.gnu.org/licenses/>.
#

"""
    This file implements the compiler class for Kaldi
"""


#---------------------------------------------------------------------------

import hashlib, json, logging, math, os.path, subprocess, tempfile
from collections import OrderedDict, namedtuple

from ..base                     import CompilerBase, CompilerError
from ...grammar.rule_base       import Rule
from ...grammar.elements_basic  import Impossible, Literal

import pyparsing as pp

_log = logging.getLogger("engine.compiler")


#---------------------------------------------------------------------------
# Utilities

_trace_level=0
def trace_compile(func):
    return func
    def dec(self, element, src_state, dst_state, grammar, fst):
        global _trace_level
        s = '%s %s: compiling %s' % (grammar.name, '==='*_trace_level, element)
        l = 140-len(s)
        s += ' '*l + '| %-20s %s -> %s' % (id(fst), src_state, dst_state)
        grammar._log_load.error(s)
        _trace_level+=1
        ret = func(self, element, src_state, dst_state, grammar, fst)
        _trace_level-=1
        grammar._log_load.error('%s %s: compiling %s.' % (grammar.name, '...'*_trace_level, element))
        return ret
    return dec

class FSTFileCache(object):

    def __init__(self, filename):
        self.filename = filename
        try:
            self.load()
        except Exception as e:
            _log.warning("%s: failed to load cache; initializing empty", self)
            self.cache = dict()

    def load(self):
        with open(self.filename, 'r') as f:
            self.cache = json.load(f)

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.cache, f)

    def hash(self, data):
        return hashlib.md5(data).hexdigest()

    def add(self, filename, data):
        self.cache[filename] = self.hash(data)

    def contains(self, filename, data):
        return (filename in self.cache) and (self.cache[filename] == self.hash(data))

InternalGrammar = namedtuple('InternalGrammar', 'name')
InternalRule = namedtuple('InternalRule', 'name gstring')


#---------------------------------------------------------------------------

class WFST(object):
    eps = '<eps>'
    zero = float('inf')
    one = 0.0

    def __init__(self):
        self._next_state_id = 1
        self._arc_table = []
        self._state_table = []
        self.start_state = self.add_state()

    def add_state(self, weight=None, initial=False, final=False):
        id = self._next_state_id
        self._next_state_id += 1
        if weight is None:
            weight = self.one if final else self.zero
        else:
            weight = -math.log(weight)
        self._state_table.append((id, weight))
        if initial:
            self.add_arc(self.start_state, id, self.eps)
        return id

    def add_arc(self, src_state, dst_state, label, olabel=None, weight=None):
        if label is None: label = self.eps
        if olabel is None: olabel = label
        weight = self.one if weight is None else -math.log(weight)
        self._arc_table.append([src_state, dst_state, label, olabel, weight])

    @property
    def fst_text(self, eps2disambig=True):
        eps_replacement = '#0' if eps2disambig else self.eps
        text = ''.join("%s %s %s %s %s\n" % (src_state, dst_state, ilabel if ilabel != self.eps else eps_replacement, olabel, weight)
            for (src_state, dst_state, ilabel, olabel, weight) in self._arc_table)
        text += ''.join("%s %s\n" % (id, weight) for (id, weight) in self._state_table if weight is not self.zero)
        return text

    def equalize_weights(self):
        # FIXME: dead code
        # breakeven 10-13?
        for state_id, _ in self._state_table:
            arcs = [arc for arc in self._arc_table if arc[0] == state_id]
            if arcs:
                weight = -math.log(1.0 / len(arcs))
                for arc in arcs:
                    arc[4] = weight

class KaldiRule(object):
    def __init__(self, compiler, grammar, rule, id):
        self.compiler = compiler
        self.grammar = grammar
        self.rule = rule
        self._id = id
        self._fst = WFST()
        self.matcher = None
        self._active = True
        self._dictation = "<Dictation()>" in rule.gstring()

    def __str__(self):
        return "KaldiRule(%s, %s, %s)" % (self.id, self.grammar.name, self.rule.name)

    id = property(lambda self: self._id)  # int
    dictation = property(lambda self: self._dictation)

    active = property(lambda self: self._active, doc="The ground truth of whether rule is active for decoding")
    @active.setter
    def active(self, value): self._active = value

    filename = property(lambda self: os.path.join(self.compiler.grammar_file_dir,
        self.grammar.name.replace(' ','!') + '!!' + self.rule.name.replace(' ','!') + '.fst'))

    def compile_file(self):
        KaldiCompiler._log.debug("%s: Compiling exported rule %s to %s" % (self, self.rule.name, self.filename))
        if self.filename in self.compiler.fst_filenames_set:
            raise CompilerError("KaldiRule fst filename collision")
        self.compiler.fst_filenames_set.add(self.filename)

        fst_text = self._fst.fst_text
        if self.compiler.fst_cache.contains(self.filename, fst_text) and os.path.exists(self.filename):
            # KaldiCompiler._log.debug("%s: Skipped full compilation thanks to FSTFileCache" % self)
            return
        else:
            KaldiCompiler._log.warning("%s: FSTFileCache useless; has %s not %s" % (self,
                self.compiler.fst_cache.hash(self.compiler.fst_cache.cache[self.filename]) if self.filename in self.compiler.fst_cache.cache else None,
                self.compiler.fst_cache.hash(fst_text)))
        with open(self.filename + '.txt', 'w') as f:
            f.write(fst_text)

        # subprocess.check_call([r"C:\Work\Speech\kaldi\kaldi-windows-deps\openfst\build_output\x64\Release\bin\fstcompile.exe",
        #     "--isymbols=%s/words.txt" % self.compiler.grammar_file_dir, "--osymbols=%s/words.txt" % self.compiler.grammar_file_dir,
        #     self.filename+'.txt', self.filename])
        # NOTE: windows output redirection corrupts binary data! pipes fail too!

        # subprocess.check_call(
        #     r"{dir}\fstcompile --isymbols={dir}\words.txt --osymbols={dir}\words.txt {filename}.txt"
        #     r" | {dir}\fstrelabel --relabel_ipairs={dir}/g.irelabel"
        #     r" | {dir}\fstarcsort"
        #     # r" | {dir}\fstconvert --fst_type=const"
        #     r" | {dir}\fstconvert --fst_type=const - {filename}"
        #     # " > {filename}"
        #     .format(dir=self.compiler.grammar_file_dir, filename=self.filename),
        #     shell=True, bufsize=999999)

        # run_pipeline = lambda cmd, **kwargs: subprocess.Popen(cmd.format(dir=self.compiler.grammar_file_dir, filename=self.filename), bufsize=9999999, **kwargs)
        # p = subprocess.Popen(r"{dir}\fstcompile --isymbols={dir}\words.txt --osymbols={dir}\words.txt {filename}.txt".format(dir=self.compiler.grammar_file_dir, filename=self.filename), stdout=subprocess.PIPE)
        # p = subprocess.Popen(r"{dir}\fstrelabel --relabel_ipairs={dir}/g.irelabel".format(dir=self.compiler.grammar_file_dir, filename=self.filename), stdin=p.stdout)
        # p1 = run_pipeline(r"{dir}\fstcompile --isymbols={dir}\words.txt --osymbols={dir}\words.txt {filename}.txt", stdout=subprocess.PIPE)
        # p2 = run_pipeline(r"{dir}\fstrelabel --relabel_ipairs={dir}/g.irelabel", stdin=p1.stdout, stdout=subprocess.PIPE)
        # p3 = run_pipeline(r"{dir}\fstarcsort", stdin=p2.stdout, stdout=subprocess.PIPE)
        # p4 = run_pipeline(r"{dir}\fstconvert --fst_type=const - {filename}", stdin=p3.stdout)

        run = lambda cmd, **kwargs: subprocess.check_call(cmd.format(dir=self.compiler.grammar_file_dir, filename=self.filename), **kwargs)
        p1 = run("{dir}/fstcompile --isymbols={dir}/words.txt --osymbols={dir}/words.txt {filename}.txt {filename}")
        p2 = run("{dir}/fstrelabel --relabel_ipairs={dir}/g.irelabel {filename} {filename}")
        p3 = run("{dir}/fstarcsort {filename} {filename}")
        # p4 = run("{dir}/fstconvert --fst_type=const {filename} {filename}")
        self.compiler.fst_cache.add(self.filename, fst_text)


#---------------------------------------------------------------------------

class KaldiCompiler(CompilerBase):

    def __init__(self):
        CompilerBase.__init__(self)
        self.kaldi_rules_dict = OrderedDict()  # maps rule -> kaldi_rule
        self.kaldi_rule_id_dict = OrderedDict()  # maps kaldi_rule.id -> kaldi_rule
        self.fst_filenames_set = set()
        self.grammar_file_dir = 'tmp_kaldi'
        if not os.path.exists(self.grammar_file_dir): os.mkdir(self.grammar_file_dir)
        # self.grammar_file_dir = tempfile.mkdtemp(prefix='tmp_kaldi_', dir='.')
        self._num_kaldi_rules = 0
        self._grammar_rule_states_dict = dict()  # disabled!
        self.fst_cache = FSTFileCache(os.path.join(self.grammar_file_dir, 'fst_cache.json'))
        self._lexicon_words = []
        self.internal_grammar = InternalGrammar('!kaldi_engine_internal')

    num_kaldi_rules = property(lambda self: self._num_kaldi_rules)

    def load_words(self, words_file, unigram_probs_file=None):
        with open(words_file, 'r') as file:
            word_id_pairs = [line.strip().split() for line in file]
        self._lexicon_words = set([word for word, id in word_id_pairs if word not in "<eps> !SIL <UNK> #0 <s> </s>".split()])
        if unigram_probs_file:
            with open(unigram_probs_file, 'r') as file:
                word_count_pairs = [line.strip().split() for line in file]
            word_count_pairs = [(word, int(count)) for word, count in word_count_pairs[:30000] if word in self._lexicon_words]
            total = sum(count for word, count in word_count_pairs)
            self._lexicon_word_probs = {word: (float(count) / total) for word, count in word_count_pairs}
        # FIXME
        # self._lexicon_words = "alpha bravo charlie".split()
        return self._lexicon_words

    # def compile_null_grammar(self):
    #     kaldi_rule = KaldiRule(self, None, None)
    #     fst = kaldi_rule._fst
    #     state = fst.add_state(initial=True, final=True)
    #     # fst.add_arc(state, state, '<UNK>', fst.eps)
    #     kaldi_rule.compile_file()
    #     return kaldi_rule

    def compile_universal_grammar(self, words=None):
        rule = InternalRule('universal', lambda: '')
        kaldi_rule = KaldiRule(self, self.internal_grammar, rule, -1)
        if words is None: words = self._lexicon_words
        fst = kaldi_rule._fst
        backoff_state = fst.add_state(initial=True, final=True)
        for word in words:
            # state = fst.add_state()
            # fst.add_arc(backoff_state, state, word)
            # fst.add_arc(state, backoff_state, None)
            fst.add_arc(backoff_state, backoff_state, word)
        kaldi_rule.compile_file()
        return kaldi_rule

    def _construct_dictation_rule(self, fst, src_state, dst_state, words=None):
        """matches zero or more words"""
        if words is None: words = self._lexicon_words
        backoff_state = fst.add_state()
        fst.add_arc(src_state, backoff_state, None)
        fst.add_arc(backoff_state, dst_state, None)
        for word in words:
            fst.add_arc(backoff_state, backoff_state, word)

    def _construct_dictation_plus_rule(self, fst, src_state, dst_state, words=None, start_weight=None):
        """matches one or more words"""
        if words is None: words = self._lexicon_words
        word_probs = self._lexicon_word_probs
        backoff_state = fst.add_state()
        fst.add_arc(src_state, backoff_state, None, weight=start_weight)
        for word, prob in word_probs.items():
            state = fst.add_state()
            fst.add_arc(backoff_state, state, word, weight=prob)
            fst.add_arc(state, backoff_state, None)
            fst.add_arc(state, dst_state, None)

    def _construct_dictation_one_rule(self, fst, src_state, dst_state, words=None, start_weight=None):
        """matches one words"""
        if words is None: words = self._lexicon_words
        word_probs = self._lexicon_word_probs
        backoff_state = fst.add_state()
        fst.add_arc(src_state, backoff_state, None, weight=start_weight)
        for word, prob in word_probs.items():
            state = fst.add_state()
            fst.add_arc(backoff_state, state, word, weight=prob)
            # fst.add_arc(state, backoff_state, None)
            fst.add_arc(state, dst_state, None)

    #-----------------------------------------------------------------------
    # Methods for recognition.

    def prepare_for_recognition(self):
        self.fst_cache.save()

    def parse(self, kaldi_rule, output):
        try:
            parse_results = kaldi_rule.matcher.parseString(output, parseAll=True)
        except pp.ParseException:
            return None
        parsed_output = ' '.join(parse_results)
        if parsed_output.lower() != output:
            self._log.error("parsed_output(%r).lower() != output(%r)" % (parse_results, output))
        kaldi_rule_id = int(parse_results.getName())
        assert kaldi_rule_id == kaldi_rule.id
        return parsed_output

    #-----------------------------------------------------------------------
    # Methods for compiling grammars.

    def compile_grammar(self, grammar, engine):
        self._log.debug("%s: Compiling grammar %s." % (self, grammar.name))

        kaldi_rules_dict = OrderedDict()
        for rule in grammar.rules:
            if rule.exported:
                kaldi_rule_id = self._num_kaldi_rules
                self._num_kaldi_rules += 1
                kaldi_rule = KaldiRule(self, grammar, rule, kaldi_rule_id)
                matcher, _, _ = self._compile_rule(rule, grammar, kaldi_rule._fst)
                kaldi_rule.matcher = matcher.setName(str(kaldi_rule_id)).setResultsName(str(kaldi_rule_id))
                kaldi_rule.compile_file()
                self.kaldi_rule_id_dict[kaldi_rule_id] = kaldi_rule
                self.kaldi_rules_dict[rule] = kaldi_rule
                kaldi_rules_dict[rule] = kaldi_rule

        return kaldi_rules_dict

    def _compile_rule(self, rule, grammar, fst, export=True):
        # Determine whether this rule has already been compiled.
        if (grammar.name, rule.name) in self._grammar_rule_states_dict:
            self._log.debug("%s: Already compiled rule %s%s." % (self, rule.name, ' [EXPORTED]' if export else ''))
            return self._grammar_rule_states_dict[(grammar.name, rule.name)]
        else:
            self._log.debug("%s: Compiling rule %s%s." % (self, rule.name, ' [EXPORTED]' if export else ''))

        src_state = fst.add_state(initial=export)
        dst_state = fst.add_state(final=export)
        matcher = self.compile_element(rule.element, src_state, dst_state, grammar, fst)
        matcher = matcher.setName(rule.name).setResultsName(rule.name)

        # self._grammar_rule_states_dict[(grammar.name, rule.name)] = (matcher, src_state, dst_state)
        return (matcher, src_state, dst_state)

    #-----------------------------------------------------------------------
    # Methods for compiling elements.

    translation_dict = {
        # missing from dictionary
        'backspace': 'back space',
        'backtick': 'back tick',
        'caret': 'carat',
        'dimness': 'dim ness',
        'juliett': 'juliet',
        'outliner': 'outline',
        'semicolon': 'semi-colon',
        'xray': 'x-ray',
    }
    untranslation_dict = {v: k for k, v in translation_dict.items()}
    translation_dict.update({
        'tasco': 'task view', # 'askew'
        # fix hacks due to msasr
        'risky': 'whiskey',
    })

    def untranslate_output(self, output):
        for old, new in self.untranslation_dict.iteritems():
            output = output.replace(old, new)
        return output

    def translate_words(self, words):
        new_words = []
        for word in words:
            # word = word.lower()
            for old, new in self.translation_dict.iteritems():
                word = word.replace(old, new)
            new_words.extend(word.split())
        return new_words

    def compile_element(self, element, *args, **kwargs):
        # Look for a compiler method to handle the given element.
        for element_type, compiler in self.element_compilers:
            if isinstance(element, element_type):
                return compiler(self, element, *args, **kwargs)
        # Didn't find a compiler method for this element type.
        raise NotImplementedError("Compiler %s not implemented for element type %s." % (self, element))

    @trace_compile
    def _compile_sequence(self, element, src_state, dst_state, grammar, fst):
        # "insert" new states for individual children elements
        states = [src_state] + [fst.add_state() for i in range(len(element.children)-1)] + [dst_state]
        matchers = []
        for i, child in enumerate(element.children):
            s1 = states[i]
            s2 = states[i + 1]
            matchers.append(self.compile_element(child, s1, s2, grammar, fst))
        return pp.And(tuple(matchers))

    @trace_compile
    def _compile_alternative(self, element, src_state, dst_state, grammar, fst):
        matchers = []
        for child in element.children:
            matchers.append(self.compile_element(child, src_state, dst_state, grammar, fst))
        return pp.Or(tuple(matchers))

    @trace_compile
    def _compile_optional(self, element, src_state, dst_state, grammar, fst):
        matcher = self.compile_element(element.children[0], src_state, dst_state, grammar, fst)
        fst.add_arc(src_state, dst_state, None)
        return pp.Optional(matcher)

    @trace_compile
    def _compile_literal(self, element, src_state, dst_state, grammar, fst):
        # "insert" new states for individual words
        words = element.words
        matcher = pp.CaselessLiteral(' '.join(words))
        words = self.translate_words(words)
        states = [src_state] + [fst.add_state() for i in range(len(words)-1)] + [dst_state]
        for i, word in enumerate(words):
            s1 = states[i]
            s2 = states[i + 1]
            fst.add_arc(s1, s2, word.lower())
        return matcher

    @trace_compile
    def _compile_rule_ref(self, element, src_state, dst_state, grammar, fst):
        matcher, rule_src_state, rule_dst_state = self._compile_rule(element.rule, grammar, fst, export=False)
        fst.add_arc(src_state, rule_src_state, None)
        fst.add_arc(rule_dst_state, dst_state, None)
        return matcher

    # @trace_compile
    # def _compile_list_ref(self, element, src_state, dst_state, grammar, fst):
    #     list_rule_name = "__list_%s" % element.list.name
    #     rule_handle = grammar_handle.Rules.FindRule(list_rule_name)
    #     if not rule_handle:
    #         grammar.add_list(element.list)
    #         flags = constants.SRADynamic
    #         rule_handle = grammar_handle.Rules.Add(list_rule_name, flags, 0)
    #     src_state.AddRuleTransition(dst_state, rule_handle)

    @trace_compile
    def _compile_dictation(self, element, src_state, dst_state, grammar, fst):
        # fst.add_arc(src_state, dst_state, None)
        self._construct_dictation_one_rule(fst, src_state, dst_state, start_weight=1)  # unweighted=0.01
        return pp.OneOrMore(pp.Word(pp.alphas + pp.alphas8bit + pp.printables))

    # def _get_dictation_rule(self, grammar, fst):
    #     """
    #         Retrieve the special dictation rule.

    #         If it already exists within this grammar, return it.
    #         If it does not yet exist, create it.
    #     """
    #     rule_handle = grammar_handle.Rules.FindRule("dgndictation")
    #     if rule_handle:
    #         # self._log.error("%s: dictation rule already present." % self)
    #         return rule_handle
    #     # self._log.error("%s: building dictation rule ." % self)

    #     flags = 0
    #     # flags = constants.SRADynamic
    #     rule_handle = grammar_handle.Rules.Add("dgndictation", flags, 0)
    #     # grammar.add_rule(

    #     src_state = rule_handle.InitialState
    #     dst_state = None
    #     src_state.AddSpecialTransition(dst_state, 2)
    #     states = [src_state.Rule.AddState() for i in range(16)]
    #     src_state.AddSpecialTransition(states[0], 2)
    #     for s in states:
    #         s.AddSpecialTransition(dst_state, 2)
    #     for s1, s2 in zip(states[:-1], states[1:]):
    #         s1.AddSpecialTransition(s2, 2)

    #     return rule_handle

    # @trace_compile
    # def _compile_impossible(self, element, src_state, dst_state, grammar, fst):
    #     src_state.AddWordTransition(dst_state, "no can do")
    #     return
