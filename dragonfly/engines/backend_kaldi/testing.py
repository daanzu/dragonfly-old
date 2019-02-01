import time
from contextlib import contextmanager

debug_timer_enabled = True
debug_timer_stack = []

@contextmanager
def debug_timer(log, desc, enabled=True):
    start_time = time.clock()
    debug_timer_stack.append(start_time)
    spent_time_func = lambda: time.clock() - start_time
    yield spent_time_func
    start_time_adjusted = debug_timer_stack.pop()
    if enabled and debug_timer_enabled:
        log.debug("%s %d ms", desc, (time.clock() - start_time_adjusted) * 1000)
    if debug_timer_stack:
        debug_timer_stack[-1] += spent_time_func()
