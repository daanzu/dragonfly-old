import logging, os.path, time
import numpy as np
from cffi import FFI
from ..base import EngineError

# TERMINOLOGY NOTE: a kaldi "grammar" can be (usually is) just a single dragonfly rule!

_log = logging.getLogger("engine.kaldiwrapper")

class KaldiDecoderBase(object):
    """docstring for KaldiDecoderBase"""

    def __init__(self):
        self._reset_decode_time()

    _library_binary = r"C:\Work\Speech\kaldi\kaldi-windows\kaldiwin_vs2017_OPENBLAS\x64\Debug\dragonfly.dll"
    _library_binary = r"C:\Work\Speech\kaldi\kaldi-windows\kaldiwin_vs2017_OPENBLAS\x64\Release\dragonfly.dll"
    # _library_binary = r"tmp_kaldi/dragonfly.dll"
    _library_binary = r"dragonfly/engines/backend_kaldi/dragonfly.dll"

    def _reset_decode_time(self):
        self._decode_time = 0
        self._decode_real_time = 0
        self._decode_times = []

    def _start_decode_time(self, num_frames):
        self.decode_start_time = time.clock()
        self._decode_real_time += 1000.0 * num_frames / self.sample_rate

    def _stop_decode_time(self, finalize=False):
        this = (time.clock() - self.decode_start_time) * 1000.0
        self._decode_time += this
        self._decode_times.append(this)
        if finalize:
            rtf = 1.0 * self._decode_real_time / self._decode_time
            pct = 100.0 * this / self._decode_time
            _log.debug("decoded at %.2f RTF, for %d ms audio, spending %d ms, of which %d ms (%d%%) in finalization",
                rtf, self._decode_real_time, self._decode_time, this, pct)
            _log.debug("    decode times: %s", ' '.join("%d" % t for t in self._decode_times))
            self._reset_decode_time()

class KaldiGmmDecoder(KaldiDecoderBase):
    """docstring for KaldiGmmDecoder"""

    def __init__(self, graph_dir=None, words_file=None, graph_file=None, model_conf_file=None):
        super(KaldiGmmDecoder, self).__init__()
        self._ffi = FFI()
        self._ffi.cdef("""
            int test();
            void* init_gmm(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
                char* word_syms_filename_cp, char* fst_in_str_cp, char* config_cp);
            bool decode_gmm(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize);
            bool get_output_gmm(void* model_vp, char* output, int32_t output_length, double* likelihood_p);
        """)
        self._lib = self._ffi.dlopen(self._library_binary)

        if words_file is None and graph_dir is not None: words_file = graph_dir + r"graph\words.txt"
        if graph_file is None and graph_dir is not None: graph_file = graph_dir + r"graph\HCLG.fst"
        self.words_file = os.path.normpath(words_file)
        self.graph_file = os.path.normpath(graph_file)
        self.model_conf_file = os.path.normpath(model_conf_file)
        self._model = self._lib.init_gmm(7.0, 7000, 200, 8.0, words_file, graph_file, model_conf_file)
        self.sample_rate = 16000

    def decode(self, frames, finalize, grammars_activity=None):
        if not isinstance(frames, np.ndarray): frames = np.frombuffer(frames, np.int16)
        frames = frames.astype(np.float32)
        frames_char = self._ffi.from_buffer(frames)
        frames_float = self._ffi.cast('float *', frames_char)

        self._start_decode_time(len(frames))
        result = self._lib.decode_gmm(self._model, self.sample_rate, len(frames), frames_float, finalize)
        self._stop_decode_time(finalize)

        if not result:
            raise RuntimeError("decoding error")
        return finalize

    def get_output(self, max_output_length=1024):
        output_p = self._ffi.new('char[]', max_output_length)
        likelihood_p = self._ffi.new('double *')
        result = self._lib.get_output_gmm(self._model, output_p, max_output_length, likelihood_p)
        output_str = self._ffi.string(output_p)
        likelihood = likelihood_p[0]
        return output_str, likelihood

class KaldiOtfGmmDecoder(KaldiDecoderBase):
    """docstring for KaldiOtfGmmDecoder"""

    def __init__(self, graph_dir=None, words_file=None, model_conf_file=None, hcl_fst_file=None, grammar_fst_files=None):
        super(KaldiOtfGmmDecoder, self).__init__()
        self._ffi = FFI()
        self._ffi.cdef("""
            int test();
            void* init_otf_gmm(float beam, int32_t max_active, int32_t min_active, float lattice_beam,
                char* word_syms_filename_cp, char* config_cp,
                char* hcl_fst_filename_cp, char** grammar_fst_filenames_cp, int32_t grammar_fst_filenames_len);
            bool add_grammar_fst_otf_gmm(void* model_vp, char* grammar_fst_filename_cp);
            bool decode_otf_gmm(void* model_vp, float samp_freq, int32_t num_frames, float* frames, bool finalize,
                bool* grammars_activity, int32_t grammars_activity_size);
            bool get_output_otf_gmm(void* model_vp, char* output, int32_t output_length, double* likelihood_p);
        """)
        self._lib = self._ffi.dlopen(self._library_binary)

        if words_file is None and graph_dir is not None: words_file = graph_dir + r"graph\words.txt"
        if hcl_fst_file is None and graph_dir is not None: hcl_fst_file = graph_dir + r"graph\HCLr.fst"
        if grammar_fst_files is None and graph_dir is not None: grammar_fst_files = [graph_dir + r"graph\Gr.fst"]
        self.words_file = os.path.normpath(words_file)
        self.model_conf_file = os.path.normpath(model_conf_file)
        self.hcl_fst_file = os.path.normpath(hcl_fst_file)
        grammar_fst_filenames_cps = [self._ffi.new('char[]', os.path.normpath(f)) for f in grammar_fst_files]
        grammar_fst_filenames_cp = self._ffi.new('char*[]', grammar_fst_filenames_cps)
        self._model = self._lib.init_otf_gmm(7.0, 7000, 200, 8.0, words_file, model_conf_file,
            hcl_fst_file, self._ffi.cast('char**', grammar_fst_filenames_cp), len(grammar_fst_files))
        self.sample_rate = 16000
        self.num_grammars = len(grammar_fst_files)

    def add_grammar_fst(self, grammar_fst_file):
        grammar_fst_file = os.path.normpath(grammar_fst_file)
        _log.debug("%s: adding grammar_fst_file: %s", self, grammar_fst_file)
        result = self._lib.add_grammar_fst_otf_gmm(self._model, grammar_fst_file)
        if not result:
            raise EngineError("error adding grammar")
        self.num_grammars += 1

    def decode(self, frames, finalize, grammars_activity=None):
        # grammars_activity = [True] * self.num_grammars
        # grammars_activity = np.random.choice([True, False], len(grammars_activity)).tolist(); print grammars_activity; time.sleep(5)
        if grammars_activity is None: grammars_activity = []
        else: _log.debug("decode: grammars_activity = %s", ''.join('1' if a else '0' for a in grammars_activity))
        # if len(grammars_activity) != self.num_grammars:
        #     raise EngineError("wrong len(grammars_activity)")

        if not isinstance(frames, np.ndarray): frames = np.frombuffer(frames, np.int16)
        frames = frames.astype(np.float32)
        frames_char = self._ffi.from_buffer(frames)
        frames_float = self._ffi.cast('float *', frames_char)

        self._start_decode_time(len(frames))
        result = self._lib.decode_otf_gmm(self._model, self.sample_rate, len(frames), frames_float, finalize,
            grammars_activity, len(grammars_activity))
        self._stop_decode_time(finalize)

        if not result:
            raise EngineError("decoding error")
        return finalize

    def get_output(self, max_output_length=1024):
        output_p = self._ffi.new('char[]', max_output_length)
        likelihood_p = self._ffi.new('double *')
        result = self._lib.get_output_otf_gmm(self._model, output_p, max_output_length, likelihood_p)
        output_str = self._ffi.string(output_p)
        likelihood = likelihood_p[0]
        return output_str, likelihood

class CodeTimer:
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        self.start = time.clock()
    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (time.clock() - self.start) * 1000.0
        logging.debug("code spent %s ms in block %s", self.took, self.name)


def stash():

    # with open(r"C:\Work\Speech\kaldi\kaldi-windows\src\dragonfly\dragonfly.h") as f: ffi.cdef(f.read())

    # libgcc_s_seh-1.dll*
    # libgfortran-3.dll*
    # libopenblas.dll*
    # libquadmath-0.dll*

    exp = r"C:\Work\Speech\training\kaldi_tmp\exp\\"
    model_dir = exp + r"tri3b_mmi_online\\"
    graph_dir = exp + r"tri3b_cnc\\"
    config = r"testing\kaldi_decoding.conf"

    import collections
    import contextlib
    import wave
    import struct
    def read_wave(path):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000)
            num_frames = wf.getnframes()
            pcm_data = wf.readframes(num_frames)
            duration = num_frames / sample_rate
            return pcm_data, sample_rate, num_frames

    pcm_data, sample_rate, num_frames = read_wave("tmp/alpha-bravo-charlie.wav")

    # np.frombuffer(data, np.int16).astype(np.float32)
    # samples = struct.unpack_from('<%dh' % num_frames, pcm_data)
    # samples = np.array(samples, dtype=np.float32)


    from IPython import embed; embed()
