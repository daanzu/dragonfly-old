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
SAPI 5 engine class
============================================================================

"""


#---------------------------------------------------------------------------

import time
from datetime import datetime
import win32con
from ctypes import *

from win32com.client           import Dispatch, getevents, constants
from win32com.client.gencache  import EnsureDispatch
from pywintypes                import com_error

from ..base                    import EngineBase, EngineError
from .compiler                 import Sapi5Compiler
from .dictation                import Sapi5DictationContainer
from .recobs                   import Sapi5RecObsManager
#from .timer                    import NatlinkTimerManager
from ...grammar.state          import State
from ...windows.window         import Window


#===========================================================================

class POINT(Structure):
    _fields_ = [('x', c_long),
                ('y', c_long)]

class MSG(Structure):
    _fields_ = [('hwnd', c_int),
                ('message', c_uint),
                ('wParam', c_int),
                ('lParam', c_int),
                ('time', c_int),
                ('pt', POINT)]


#===========================================================================

class Sapi5Engine(EngineBase):
    """ Speech recognition engine back-end for SAPI 5; dannzu edition. """

    _name = "sapi5"
    DictationContainer = Sapi5DictationContainer

    #-----------------------------------------------------------------------

    def __init__(self, private=True, retain=None, response_speed=None, complex_response_speed=None):
        EngineBase.__init__(self)

        if private: EnsureDispatch("SAPI.SpInprocRecognizer")
        else: EnsureDispatch("SAPI.SpSharedRecognizer")
        EnsureDispatch("SAPI.SpVoice")
        self._recognizer  = None
        self._speaker     = None
        self._compiler    = None

        self._private = bool(private)
        self._retain_dir = retain
        self._response_speed = response_speed
        self._complex_response_speed = complex_response_speed
        self._last_window = None

        self._recognition_observer_manager = Sapi5RecObsManager(self)
        self.my_recognition_observer = None
        # self._timer_manager = NatlinkTimerManager(0.02, self)

    def connect(self):
        """ Connect to back-end SR engine. """
        if self._private:
            self._recognizer = Dispatch("SAPI.SpInprocRecognizer")
            audio = self._recognizer.GetAudioInputs().Item(0)
            self._log.info("Audio input: %s %s" % (audio.GetDescription(), audio.Id.split('.')[-1]))
            self._recognizer.AudioInput = audio
            self._log.info("Profile: %s" % self._recognizer.Profile.GetDescription())
            if self._response_speed: self._recognizer.SetPropertyNumber("ResponseSpeed", self._response_speed)  # default: 150
            if self._complex_response_speed: self._recognizer.SetPropertyNumber("ComplexResponseSpeed", self._complex_response_speed)  # default: 500
        else:
            self._recognizer = Dispatch("SAPI.SpSharedRecognizer")
        self._speaker     = Dispatch("SAPI.SpVoice")
        self._compiler    = Sapi5Compiler()

    def disconnect(self):
        """ Disconnect from back-end SR engine. """
        self._recognizer  = None
        self._speaker     = None
        self._compiler    = None

    #-----------------------------------------------------------------------
    # Methods for working with grammars.

    def _load_grammar(self, grammar):
        """ Load the given *grammar*. """
        self._log.debug("Loading grammar %s." % grammar.name)
        if not self._recognizer:
            self.connect()

        grammar.engine = self

        # Dependency checking.
        memo = []
        for r in grammar._rules:
            for d in r.dependencies(memo):
                grammar.add_dependency(d)

        # Create recognition context, compile grammar, and create
        #  the grammar wrapper object for managing this grammar.
        context = self._recognizer.CreateRecoContext()
        if self._retain_dir: context.RetainedAudio = constants.SRAORetainAudio
        handle = self._compiler.compile_grammar(grammar, context)
        wrapper = GrammarWrapper(grammar, handle, context, self)

        handle.State = constants.SGSEnabled
        for rule in grammar.rules:
            self._log.debug("Activating rule %s." % rule.name)
            handle.CmdSetRuleState(rule.name, constants.SGDSActive)
#        self.activate_grammar(grammar)
#        for l in grammar.lists:
#            l._update()
        handle.CmdSetRuleState("_FakeRule", constants.SGDSActive)

        return wrapper

    def _unload_grammar(self, grammar, wrapper):
        """ Unload the given *grammar*. """
        try:
            wrapper.handle.State = constants.SGSDisabled
        except Exception, e:
            self._log.exception("Failed to unload grammar %s: %s."
                                % (grammar, e))

    def activate_grammar(self, grammar):
        """ Activate the given *grammar*. """
        self._log.debug("Activating grammar %s." % grammar.name)
        grammar_handle = self._get_grammar_wrapper(grammar).handle
        grammar_handle.State = constants.SGSEnabled

    def deactivate_grammar(self, grammar):
        """ Deactivate the given *grammar*. """
        self._log.debug("Deactivating grammar %s." % grammar.name)
        grammar_handle = self._get_grammar_wrapper(grammar).handle
        grammar_handle.State = constants.SGSDisabled

    def activate_grammars(self):
        for grammar_wrapper in self._grammar_wrappers.itervalues():
            grammar_wrapper.handle.State = constants.SGSEnabled

    def deactivate_grammars(self):
        for grammar_wrapper in self._grammar_wrappers.itervalues():
            grammar_wrapper.handle.State = constants.SGSDisabled

    def activate_rule(self, rule, grammar):
        """ Activate the given *rule*. """
        self._log.debug("Activating rule %s in grammar %s."
                        % (rule.name, grammar.name))
        # if grammar.name == "wemo":
        #     print "activate_rule(%s,%s)" % (rule.name, grammar.name)
        grammar_handle = self._get_grammar_wrapper(grammar).handle
        grammar_handle.CmdSetRuleState(rule.name, constants.SGDSActive)

    def deactivate_rule(self, rule, grammar):
        """ Deactivate the given *rule*. """
        self._log.debug("Deactivating rule %s in grammar %s."
                        % (rule.name, grammar.name))
        grammar_handle = self._get_grammar_wrapper(grammar).handle
        grammar_handle.CmdSetRuleState(rule.name, constants.SGDSInactive)

    def update_list(self, lst, grammar):
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
        self._log.debug("Setting exclusiveness of grammar %s to %s."
                        % (grammar.name, exclusive))
        grammar_handle = self._get_grammar_wrapper(grammar).handle
        grammar_handle.State = constants.SGSExclusive
#        grammar_handle.SetGrammarState(constants.SPGS_EXCLUSIVE)

    def check_foreground_window(self):
        window = Window.get_foreground()
        curr_window = (window.executable, window.title, window.handle)
        if curr_window != self._last_window:
            self._log.info("New foreground window: (%s, %s, %d)" % curr_window)
            self._last_window = curr_window
            for grammar_wrapper in self._grammar_wrappers.itervalues():
                grammar_wrapper.grammar.process_begin(window.executable, window.title, window.handle)
            return curr_window
        return None

    def grammars_process_begin(self):
        window = Window.get_foreground()
        for grammar_wrapper in self._grammar_wrappers.itervalues():
            grammar_wrapper.grammar.process_begin(window.executable, window.title, window.handle)


    #-----------------------------------------------------------------------
    # Miscellaneous methods.

    def mimic(self, words):
        """ Mimic a recognition of the given *words*. """
        if isinstance(words, basestring):
            phrase = words
        else:
            phrase = " ".join(words)
        self._recognizer.EmulateRecognition(phrase)

    def speak(self, text):
        """ Speak the given *text* using text-to-speech. """
        self._speaker.Speak(text)

    def _get_language(self):
        return "en"

    def wait_for_recognition(self, timeout=None):
        NULL = c_int(win32con.NULL)
        if timeout != None:
            begin_time = time.time()
            timed_out = False
            windll.user32.SetTimer(NULL, NULL, int(timeout * 1000), NULL)
    
        message = MSG()
        message_pointer = pointer(message)

        while (not timeout) or (time.time() - begin_time < timeout):
            self._log.error("loop")
            if windll.user32.GetMessageW(message_pointer, NULL, 0, 0) == 0:
                msg = str(WinError())
                self._log.error("GetMessageW() failed: %s" % msg)
                raise EngineError("GetMessageW() failed: %s" % msg)

            if message.message == win32con.WM_TIMER:
                self._log.error("loop, timeout")
                # A timer message means this loop has timed out.
                timed_out = True
                break
            else:
                self._log.error("loop, dispatch")
                # Process other messages as normal.
                windll.user32.TranslateMessage(message_pointer)
                windll.user32.DispatchMessageW(message_pointer)

        return not timed_out


#---------------------------------------------------------------------------
# Utility generator function for iterating over COM collections.

def collection_iter(collection):
    if not collection:
        return
    for index in xrange(0, collection.Count):
        yield collection.Item(index)


#---------------------------------------------------------------------------

class GrammarWrapper(object):

    def __init__(self, grammar, handle, context, engine):
        self.grammar = grammar
        self.handle = handle
        self.engine = engine
        self.context = context

        # Register callback functions which will handle recognizer events.
        base = getevents("SAPI.SpSharedRecoContext")
        class ContextEvents(base): pass
        c = ContextEvents(context)
        c.OnPhraseStart = self.phrase_start_callback
        c.OnRecognition = self.recognition_callback
        if hasattr(grammar, "process_recognition_other"):
            c.OnRecognitionForOtherContext = self.recognition_other_callback
        if hasattr(grammar, "process_recognition_failure"):
            c.OnFalseRecognition = self.recognition_failure_callback

    def phrase_start_callback(self, stream_number, stream_position):
        window = Window.get_foreground()
        # print "phrase_start_callback(%s): %s // %s // %d" % (self.grammar._name, window.executable, window.title, window.handle)
        self.grammar.process_begin(window.executable, window.title,
                                   window.handle)

    def recognition_callback(self, StreamNumber, StreamPosition, RecognitionType, Result):
        try:
            newResult = Dispatch(Result)
            phrase_info = newResult.PhraseInfo
            rule_name = phrase_info.Rule.Name

            #---------------------------------------------------------------
            # Build a list of rule names for each element.

            # First populate it with the top level rule name.
            element = phrase_info.Rule
            name = element.Name
            start = element.FirstElement
            count = element.NumberOfElements
            rule_names = [name] * count

            # Walk the tree of child rules and put their names in the list.
            stack = [collection_iter(phrase_info.Rule.Children)]
            while stack:
                try: element = stack[-1].next()
                except StopIteration: stack.pop(); continue
                name = element.Name
                start = element.FirstElement
                count = element.NumberOfElements
                rule_names[start:start + count] = [name] * count
                if element.Children:
                    stack.append(collection_iter(element.Children))

            #---------------------------------------------------------------
            # Prepare the words and rule names for the element parsers.

            replacements = [False] * len(rule_names)
            if phrase_info.Replacements:
                for replacement in collection_iter(phrase_info.Replacements):
                    begin = replacement.FirstElement
                    end = begin + replacement.NumberOfElements
                    replacements[begin] = replacement.Text
                    for index in range(begin + 1, end):
                        replacements[index] = True

            results = []
            rule_set = list(set(rule_names))
            elements = phrase_info.Elements
            for index in range(len(rule_names)):
                element = elements.Item(index)
                rule_id = rule_set.index(rule_names[index])
                replacement = replacements[index]
                info = [element.LexicalForm, rule_id,
                        element.DisplayText, element.DisplayAttributes,
                        replacement]
                results.append(info)

            from IPython import embed; embed()

            #---------------------------------------------------------------
            # Retain audio
            if self.engine._retain_dir:
                try:
                    file_stream = Dispatch("SAPI.SpFileStream")
                    # Note: application can also retrieve smaller portions of the audio stream by specifying a starting phrase element and phrase element length
                    audio_stream = newResult.Audio()
                    if audio_stream:
                        # not a mimic, so have audio data
                        # write audio data
                        file_stream.Format = audio_stream.Format
                        filename = self.engine._retain_dir + "retain_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + ".wav"
                        file_stream.Open(filename, constants.SSFMCreateForWrite)
                        try:
                            buf = audio_stream.GetData()
                            written = file_stream.Write(buf)
                        finally:
                            file_stream.Close()
                        # write info
                        words = ' '.join([r[2] for r in results])
                        audio_length = int(newResult.Times.Length) / 1e7
                        with open(self.engine._retain_dir+"retain.csv", "a") as csvfile:
                            csvfile.write(','.join([filename, str(audio_length), self.grammar._name, rule_name, words]) + '\n')
                except:
                    self.engine._log.exception("exception retaining audio")

            #---------------------------------------------------------------
            # Attempt to parse the recognition.

            func = getattr(self.grammar, "process_recognition", None)
            if func:
                words = [r[2] for r in results]
                if not func(words):
                    return

            s = State(results, rule_set, self.engine)
            for r in self.grammar._rules:
                if r.name != rule_name:
                    continue
                self.engine._log.log(15, "recognition_callback: Grammar %s: Rule %s:\n\t%s", self.grammar._name, r.name, s.words())
                s.initialize_decoding()
                for result in r.decode(s):
                    if s.finished():
                        root = s.build_parse_tree()
                        r.process_recognition(root)
                        if self.engine.my_recognition_observer is not None: self.engine.my_recognition_observer.on_recognition(s.words())
                        return

        except Exception, e:
            self.engine._log.error("Grammar %s: exception: %s"
                                   % (self.grammar._name, e), exc_info=True)

        #-------------------------------------------------------------------
        # If this point is reached, then the recognition was not
        #  processed successfully..

        self.engine._log.error("Grammar %s: failed to decode"
                               " recognition %r."
                               % (self.grammar._name,
                                  [r[0] for r in results]))

    def recognition_other_callback(self, StreamNumber, StreamPosition):
            func = getattr(self.grammar, "process_recognition_other", None)
            if func:
                # Note that SAPI 5.3 doesn't offer access to the actual
                #  recognition contents during a
                #  OnRecognitionForOtherContext event.
                func(words=False)
            return

    def recognition_failure_callback(self, StreamNumber, StreamPosition, Result):
            func = getattr(self.grammar, "process_recognition_failure", None)
            if func:
                func()
            return
