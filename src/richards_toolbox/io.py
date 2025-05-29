import contextlib
import os


@contextlib.contextmanager
def suppress_all_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)