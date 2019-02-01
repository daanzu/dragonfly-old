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
Kaldi engine classes
============================================================================

"""


#---------------------------------------------------------------------------

import time, collections, subprocess, threading, os
from six import string_types, integer_types

from ..base                     import EngineBase, EngineError
from .compiler                  import KaldiCompiler
from .kaldiwrapper              import KaldiGmmDecoder, KaldiOtfGmmDecoder
from .audio                     import VADAudio, AudioStore
from .dictation                 import KaldiDictationContainer
from .recobs                    import KaldiRecObsManager
from .testing                   import debug_timer
from ...grammar.state           import State
from ...windows.window          import Window


#===========================================================================

class KaldiEngine(EngineBase):
    """ Speech recognition engine back-end for Kaldi recognizer. """

    _name = "kaldi"
    DictationContainer = KaldiDictationContainer

    #-----------------------------------------------------------------------

    def __init__(self):
        EngineBase.__init__(self)
        self._compiler = None
        self._decoder = None
        self._audio = None
        self._speaker = None
        self._recognition_observer_manager = KaldiRecObsManager(self)

    def connect(self):
        """ Connect to back-end SR engine. """
        if self._decoder:
            return

        self._log.debug("Loading KaldiEngine in process %s." % os.getpid())
        # subprocess.call(['vsjitdebugger', '-p', str(os.getpid())]); time.sleep(10)
        # time.sleep(30)

        self._compiler = KaldiCompiler()

        exp = r"C:\Work\Speech\training\kaldi_tmp\exp\\"
        model_dir = exp + r"tri3b_mmi_online\\"
        graph_dir = exp + r"tri3b_cnc\\"
        model_conf_file = r"testing\kaldi_decoding.conf"
        words_file = r"C:\Work\Speech\training\kaldi_tmp\data/lang_cnc/words.txt"
        graph_file = r"C:\Work\Speech\training\kaldi_tmp\exp/tri3b_cnc/graph/HCLG.fst"
        graph_file = r"C:\Work\Speech\training\kaldi_tmp\exp/tri3b_test/graph/HCLG.fst"
        # graph_file = r"C:\Work\Speech\training\kaldi_tmp\data/lang_cnc/otf/HCLrGr.fst"
        words_file = r"C:\Work\Speech\training\kaldi_tmp\data/lang_dfly/words.txt"
        graph_file = r"C:\Work\Speech\training\kaldi_tmp\data/lang_dfly/otf/HCLrGr.fst"
        # graph_file = r"C:\Work\Speech\training\kaldi_tmp\data/lang_dfly/otf/HCLG.fst"
        # graph_file = r"C:\Work\Speech\training\kaldi_tmp\exp/tri3b_dfly/graph/HCLG.fst"
        # self._decoder = KaldiGmmDecoder(words_file=words_file, graph_file=graph_file, model_conf_file=model_conf_file)

        # hcl_fst_file = r"C:\Work\Speech\training\kaldi_tmp\data/lang_dfly/otf/HCL.fst"
        # grammar_fst_files = [r"C:\Work\Speech\training\kaldi_tmp\data/lang_dfly/G.fst"]
        hcl_fst_file = r"C:\Work\Speech\training\kaldi_tmp\data/lang_dfly/otf/HCLr.fst"
        grammar_fst_files = [r"C:\Work\Speech\training\kaldi_tmp\data/lang_dfly/otf/Gr.fst"]
        grammar_fst_files = [
            # self._compiler.grammar_file_dir + '/test1__SimpleKeyboardRule.fst',
            # self._compiler.grammar_file_dir + '/test1__TestRule.fst',
            # self._compiler.grammar_file_dir + '/coding!!python.fst',
            # self._compiler.grammar_file_dir + '/global!!KeyboardRule.fst',
            # self._compiler.grammar_file_dir + '/sleep!!SleepRule.fst',
            # self._compiler.grammar_file_dir + '/terminal!!tmux.fst',
        ]
        self._decoder = KaldiOtfGmmDecoder(words_file=words_file, model_conf_file=model_conf_file, hcl_fst_file=hcl_fst_file, grammar_fst_files=grammar_fst_files)

        # words = self._compiler.load_words(words_file)
        unigram_probs_file = r"tmp_kaldi/unigram.txt"
        words = self._compiler.load_words(words_file, unigram_probs_file=unigram_probs_file)
        kaldi_rule = self._compiler.compile_universal_grammar()
        # self._decoder.add_grammar_fst(kaldi_rule.filename)

        self._audio = VADAudio()
        self._audio_iter = self._audio.vad_collector(300)
        auto_save_func = lambda audio, text, grammar_name, rule_name: True and text == 'foxtrot'
        self.audio_store = AudioStore(self._audio, save_dir="tmp_kaldi/retain/", maxlen=5, auto_save_func=auto_save_func) if True else None

        self._any_exclusive_grammars = False

    def disconnect(self):
        """ Disconnect from back-end SR engine. """
        self._compiler = None
        self._audio.destroy()
        self._audio = None

    #-----------------------------------------------------------------------
    # Methods for working with grammars.

    def _load_grammar(self, grammar):
        """ Load the given *grammar*. """
        self._log.debug("Loading grammar %s." % grammar.name)
        if not self._decoder:
            self.connect()
        grammar.engine = self

        # Dependency checking.
        memo = []
        for r in grammar._rules:
            for d in r.dependencies(memo):
                grammar.add_dependency(d)

        kaldi_rules_dict = self._compiler.compile_grammar(grammar, self)
        wrapper = GrammarWrapper(grammar, kaldi_rules_dict, self)
        for kaldi_rule in kaldi_rules_dict.values():
            self._decoder.add_grammar_fst(kaldi_rule.filename)
        return wrapper

    def _unload_grammar(self, grammar, wrapper):
        """ Unload the given *grammar*. """
        raise NotImplementedError("Method not implemented for engine %s." % self)  # FIXME

    def activate_grammar(self, grammar):
        """ Activate the given *grammar*. """
        self._log.debug("Activating grammar %s." % grammar.name)
        self._get_grammar_wrapper(grammar).active = True

    def deactivate_grammar(self, grammar):
        """ Deactivate the given *grammar*. """
        self._log.debug("Deactivating grammar %s." % grammar.name)
        self._get_grammar_wrapper(grammar).active = False

    def activate_rule(self, rule, grammar):
        """ Activate the given *rule*. """
        self._log.debug("Activating rule %s in grammar %s." % (rule.name, grammar.name))
        self._compiler.kaldi_rules_dict[rule].active = True

    def deactivate_rule(self, rule, grammar):
        """ Deactivate the given *rule*. """
        self._log.debug("Deactivating rule %s in grammar %s." % (rule.name, grammar.name))
        self._compiler.kaldi_rules_dict[rule].active = False

    def update_list(self, lst, grammar):
        raise NotImplementedError("Method not implemented for engine %s." % self)  # FIXME
        grammar_handle = self._get_grammar_wrapper(grammar).handle
        list_rule_name = "__list_%s" % lst.name
        rule_handle = grammar_handle.Rules.FindRule(list_rule_name)

        rule_handle.Clear()
        src_state = rule_handle.InitialState
        dst_state = None
        for item in lst.get_list_items():
            src_state.AddWordTransition(dst_state, item)

        grammar_handle.Rules.Commit()

    def set_exclusiveness(self, grammar, exclusive):
        self._log.debug("Setting exclusiveness of grammar %s to %s." % (grammar.name, exclusive))
        self._get_grammar_wrapper(grammar).exclusive = exclusive
        if exclusive:
            self._get_grammar_wrapper(grammar).active = True
        self._any_exclusive_grammars = any(gw.exclusive for gw in self._grammar_wrappers.values())

    #-----------------------------------------------------------------------
    # Miscellaneous methods.

    def mimic(self, words):
        """ Mimic a recognition of the given *words*. """
        raise NotImplementedError("Method not implemented for engine %s." % self)  # FIXME
        if isinstance(words, string_types):
            phrase = words
        else:
            phrase = " ".join(words)
        self._recognizer.EmulateRecognition(phrase)

    def speak(self, text):
        """ Speak the given *text* using text-to-speech. """
        raise NotImplementedError("Method not implemented for engine %s." % self)  # FIXME
        self._speaker.Speak(text)

    def _get_language(self):
        return "en"

    def grammars_process_begin(self):
        # FIXME: hack!
        return False
        # window = Window.get_foreground()
        # for grammar_wrapper in self._grammar_wrappers.itervalues():
        #     grammar_wrapper.grammar.process_begin(window.executable, window.title, window.handle)

    def check_foreground_window(self):
        # FIXME: hack!
        return False

    def do_recognition(self, timeout=None, single=False):
        self._log.debug("do_recognition: timeout %s" % timeout)
        self._compiler.prepare_for_recognition()
        if timeout != None:
            end_time = time.time() + timeout
            timed_out = True
        phrase_started = False

        while (not timeout) or (time.time() < end_time):
            block = next(self._audio_iter)

            if block is not None:
                if not phrase_started:
                    with debug_timer(self._log, "computing activity"):
                        kaldi_rules_activity = self._compute_kaldi_rules_activity()
                    phrase_started = True
                else:
                    kaldi_rules_activity = None
                self._decoder.decode(block, False, kaldi_rules_activity)
                if self.audio_store: self.audio_store.add_block(block)

            else:
                # end of phrase
                self._decoder.decode('', True)
                output, likelihood = self._decoder.get_output()
                output = self._compiler.untranslate_output(output)
                kaldi_rule = self._parse_recognition(output)
                self._log.debug("End of utterence: likelihood %f, rule %s, %r" % (likelihood, kaldi_rule, output))
                if self.audio_store and kaldi_rule: self.audio_store.finalize(output, kaldi_rule.grammar.name, kaldi_rule.rule.name)
                phrase_started = False
                timed_out = False
                if single:
                    break

        return not timed_out

    def wait_for_recognition(self, timeout=None):
        return do_recognition(timeout=timeout, single=True)

    #-----------------------------------------------------------------------
    # Internal processing methods.

    def _compute_kaldi_rules_activity(self, phrase_start=True):
        self._active_kaldi_rules = []
        self._kaldi_rules_activity = [False] * self._compiler.num_kaldi_rules
        for grammar_wrapper in self._grammar_wrappers.values():
            if phrase_start:
                grammar_wrapper.phrase_start_callback()
            if grammar_wrapper.active and (not self._any_exclusive_grammars or (self._any_exclusive_grammars and grammar_wrapper.exclusive)):
                for kaldi_rule in grammar_wrapper.kaldi_rules_dict.values():
                    if kaldi_rule.active:
                        self._active_kaldi_rules.append(kaldi_rule)
                        self._kaldi_rules_activity[kaldi_rule.id] = True
        self._log.debug("active kaldi rules: %s", [kr.filename for kr in self._active_kaldi_rules])
        self._active_kaldi_rules.sort(key=lambda kr: 100 if kr.dictation else 0)
        return self._kaldi_rules_activity

    def _parse_recognition(self, output):
        if output == '':
            self._log.warning("attempted to parse empty recognition")
            return None
        detect_ambiguity = False
        results = []

        with debug_timer(self._log, "kaldi_rule parse time"):
            for kaldi_rule in self._active_kaldi_rules:
                self._log.debug("attempting to parse %r with %s", output, kaldi_rule)
                parsed_output = self._compiler.parse(kaldi_rule, output)
                if parsed_output is None:
                    continue
                # self._log.debug("success %d", kaldi_rule_id)
                # pass kaldi_rule, parsed_output to below
                results.append((kaldi_rule, parsed_output))
                if not detect_ambiguity:
                    break

        if not results:
            self._log.error("unable to parse recognition: %r" % output)
            return None
        if len(results) > 1:
            self._log.warning("ambiguity in recognition: %r" % output)
            results.sort(key=lambda res: 100 if res[0].dictation else 0)
        kaldi_rule, parsed_output = results[0]

        words = tuple(word for word in parsed_output.split())
        grammar_wrapper = self._get_grammar_wrapper(kaldi_rule.grammar)
        with debug_timer(self._log, "dragonfly parse time"):
            grammar_wrapper.recognition_callback(words, kaldi_rule.rule)

        return kaldi_rule


#===========================================================================

class GrammarWrapper(object):

    def __init__(self, grammar, kaldi_rules_dict, engine):
        self.grammar = grammar
        self.kaldi_rules_dict = kaldi_rules_dict
        self.engine = engine

        self.active = True
        self.exclusive = False

        # Register callback functions which will handle recognizer events.
        # c.OnPhraseStart = self.phrase_start_callback
        # c.OnRecognition = self.recognition_callback
        # if hasattr(grammar, "process_recognition_other"):
        #     c.OnRecognitionForOtherContext = self.recognition_other_callback
        # if hasattr(grammar, "process_recognition_failure"):
        #     c.OnFalseRecognition = self.recognition_failure_callback

    def phrase_start_callback(self):
        window = Window.get_foreground()
        self.grammar.process_begin(window.executable, window.title, window.handle)

    def recognition_callback(self, words, rule):
        try:
            # Prepare the words and rule names for the element parsers
            rule_names = (rule.name,)
            rule_names = ('dgndictation',)
            words_rules = tuple((word, 0) for word in words)

            # Attempt to parse the recognition
            func = getattr(self.grammar, "process_recognition", None)
            if func:
                if not func(words):
                    return

            state = State(words_rules, rule_names, self.engine)
            state.initialize_decoding()
            for result in rule.decode(state):
                if state.finished():
                    root = state.build_parse_tree()
                    with debug_timer(self.engine._log, "rule execution time"):
                        rule.process_recognition(root)
                    return

        except Exception as e:
            self.engine._log.error("Grammar %s: exception: %s" % (self.grammar._name, e), exc_info=True)

        # If this point is reached, then the recognition was not processed successfully
        self.engine._log.error("Grammar %s: failed to decode rule %s recognition %r." % (self.grammar._name, rule.name, words))

    # def recognition_other_callback(self, StreamNumber, StreamPosition):
    #         func = getattr(self.grammar, "process_recognition_other", None)
    #         if func:
    #             func(words=False)
    #         return

    # def recognition_failure_callback(self, StreamNumber, StreamPosition, Result):
    #         func = getattr(self.grammar, "process_recognition_failure", None)
    #         if func:
    #             func()
    #         return
