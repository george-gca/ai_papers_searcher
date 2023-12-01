import logging
import time
from collections.abc import Callable
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, ClassVar

try:
    from colorama import Fore
    COLORIZE = True
except Exception:
    COLORIZE = False


_logger = logging.getLogger(__name__)


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator"""

    bigger_than: float = 1.
    color: str = 'red'
    enabled: bool = True
    logger: None | Callable[[str], None] = _logger.info
    name: None | str = None
    text: str = "Elapsed time: "
    timers: ClassVar[dict[str, float]] = dict()
    _start_time: None | float = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, 0)

        if COLORIZE:
            match self.color.lower():
                case 'red':
                    self.color = Fore.RED
                case 'green':
                    self.color = Fore.GREEN
                case 'yellow':
                    self.color = Fore.YELLOW
                case 'blue':
                    self.color = Fore.BLUE
                case 'magenta':
                    self.color = Fore.MAGENTA
                case 'cyan':
                    self.color = Fore.CYAN
                case 'white':
                    self.color = Fore.WHITE

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if elapsed_time > self.bigger_than and self.logger:
            if self.name:
                text = f'{self.name} {self.text.lower()}'
            else:
                text = self.text

            if COLORIZE:
                self.logger(f'{text + self.color + self._format_interval(elapsed_time) + Fore.RESET}')
            else:
                self.logger(f'{text + self._format_interval(elapsed_time)}')

        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        if self.enabled:
            self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        if self.enabled:
            self.stop()

    def _format_interval(self, t):
        """
        Formats a number of seconds as a clock time, [Dd ][Hh ][Mm ]S.SSSs

        Parameters
        ----------
        t  : int
            Number of seconds.

        Returns
        -------
        out  : str
            [Dd ][Hh ][Mm ]S.SSSs
        """
        mins, s = divmod(int(t), 60)
        if mins:
            h, m = divmod(mins, 60)
            if h:
                d, h = divmod(h, 24)
                if d:
                    return f'{d:d}d {h:02d}h {m:02d}m {s:02d}s'
                else:
                    return f'{h:d}h {m:02d}m {s:02d}s'
            else:
                return f'{m:d}m {s:02d}s'
        else:
            return f'{t:0.3f}s'
