try:
    import h5py
    import curses
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    import time
    from typing import Union

    DATASET = h5py._hl.dataset.Dataset

except ModuleNotFoundError:
    print('Cannot run the terminal application. Missing dependencies.')

class H5TerminalApp:
    def __init__(self, path: str):
        self._path: str = path
        self._current_dir = '/'

        # Arrays for storing datasets
        self.arr1: Union[np.ndarray, None] = None
        self.arr2: Union[np.ndarray, None] = None

        # Open h5 file
        try:
            self.h5 = h5py.File(path)
        except FileNotFoundError:
            print(f'Small data not found at path: {path}')
            return

    def run(self):
        # Curses init
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)

        self._cols = curses.COLS
        self._rows = curses.LINES

        self.y_offset: int = int(self._rows*0.10)
        self.title_bar = curses.newwin(self.y_offset, self._cols, 0, 0)
        self.main_window = curses.newwin(self._rows - self.y_offset,
                                         self._cols,
                                         self.y_offset,
                                         0)

        # Used to change which keys are shown if not all can be shown at once
        self._rowshift: int = 0
        self._shiftmax: int = self._calc_shiftmax()

        self.main_window.box()

        name = self._path.split('/')[-1]
        self.title = f'HDF5 Explorer - {name}'

        curses.start_color()
        curses.use_default_colors()

        # Color pair n with foreground f and background b (n, f, b)
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)

        # Set background colors of the app
        self.main_window.bkgd(curses.color_pair(1))
        self.title_bar.bkgd(curses.color_pair(1))

        self.cursor = Cursor(y = 1,
                             x = 4,
                             ylim = [1, self._rows - self.y_offset - 2],
                             xlim = [0, self._cols - 1])

        # Display initial text.
        self._update_title_bar()
        self._refresh(self.title_bar)

        self.update_maintext = True
        self._update_main_window(None)
        self._refresh(self.main_window)

        # Main event loop
        self.main_loop()

    def main_loop(self):
        """! Main event loop.

        Loop condition assigns any keypress to a variable, and breaks if 'q'.
        Other keypresses will update screen display, otherwise continues to
        display the current text.
        """
        while ((keypress := self.main_window.getch()) != ord('q')):
            self._update_main_window(keypress)
            self._refresh(self.main_window)

    def _update_title_bar(self):
        curses.curs_set(0)
        x = (self._cols // 2) - (len(self.title) // 2)

        # Add title
        self.title_bar.addstr(1, x, self.title, curses.color_pair(1))

    def _update_main_window(self, keypress):
        curses.curs_set(1)
        keys: list = list(self.h5[self._current_dir].keys())
        if self.update_maintext:
            self.main_window.clear()
            self.main_window.box()
            for i in range(len(keys)):
                if i < self.cursor.ylim[1]:
                    self.main_window.addstr(i + 1, 4, keys[i + self._rowshift])
            self.update_maintext = False
        self._parse_keypress(keypress)

    def _parse_keypress(self, keypress):
        """! Updates display or performs action based on keypress.

        Keypresses:
        'w' : Move cursor up
        's' : Move cursor down
        'a' : Go up one level in hdf5 hierarchy (if not root)
        'd' : Go down one level in hdf5 hierarchy at current position if not on
              a dataset.
        '1' : Store dataset in array 1. No action if cursor is over a group.
        '2' : Store dataset in array 2. No action if cursor is over a group.
        'p' : Plot function. Behaviour depends on stored datasets and following
              keypresses.
        """
        if keypress == ord('d'):
            new_path = self._cursor_over_path()
            if type(self.h5[new_path]) == DATASET:
                pass
            else:
                self._current_dir = new_path
                self.update_maintext = True

            ymax: int = self._calc_ymax()
            ylims: list = [1, ymax]
            self.cursor.update_limits(ylims, self.cursor.xlim)
            self._rowshift = 0
            self._shiftmax = self._calc_shiftmax()

        elif keypress == ord('a'):
            new_path = '/'
            if self._current_dir == '/':
                pass
            else:
                for item in self._current_dir.split('/')[1:-1]:
                    new_path += item
                self._current_dir = new_path
                self.update_maintext = True
                ymax: int = self._calc_ymax()
                ylims: list = [1, ymax]
                self.cursor.update_limits(ylims, self.cursor.xlim)
                self._rowshift = 0
                self._shiftmax = self._calc_shiftmax()

        elif keypress == ord('s'):
            if self.cursor.down() and self._rowshift < self._shiftmax:
                self._rowshift += 1
                self.update_maintext = True

        elif keypress == ord('w'):
            if self.cursor.up() and self._rowshift > 0:
                self._rowshift -= 1
                self.update_maintext = True

        elif keypress == ord('p'):
            pass

        elif keypress == ord('1'):
            new_path = self._cursor_over_path()
            if type(self.h5[new_path]) == DATASET:
                self.arr1 = self.h5[new_path][()]

        elif keypress == ord('2'):
            new_path = self._cursor_over_path()
            if type(self.h5[new_path]) == DATASET:
                self.arr2 = self.h5[new_path][()]

        self.main_window.move(self.cursor.row, self.cursor.col)
        curses.doupdate()

    def _calc_ymax(self) -> int:
        """! Calculate the max row value based on current position in hdf5.

        @return ymax (int) Max row based on current position in hdf5 path.
        """
        ymax: int = min(len(self.h5[self._current_dir]),
                        self._rows - self.y_offset - 2)
        return ymax

    def _calc_shiftmax(self) -> int:
        """! Calculate the max row shift based on current position in hdf5.

        @return shiftmax (int) Maximum amount of row shift.
        """
        ymax = self._calc_ymax()
        rows_needed = len(self.h5[self._current_dir])

        if rows_needed > ymax:
            return rows_needed - ymax
        else:
            return 0

    def _cursor_over_path(self) -> str:
        """! Return the path in the hdf5 file that the cursor is over.

        @return new_path (str) The path that the cursor is currently over.
        """
        keys = self.h5[self._current_dir].keys()
        item = list(keys)[self.cursor.row - 1]
        new_path = self._current_dir
        if new_path == '/':
            new_path += item
        else:
            new_path += '/' + item
        return new_path

    def _refresh(self, window):
        window.timeout(1)
        window.refresh()

    def _cleanup(self, window):
        window.clear()
        window.refresh()

    def __exit__(self, *errors):
        self._cleanup(self.title_bar)
        self._cleanup(self.main_window )
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()

    def __del__(self):
        self.__exit__()


class Cursor:
    """! Holds the cursor position and limits on where it can move."""
    def __init__(self, y, x, ylim, xlim):
        self.row = y
        self.col = x

        self.ylim = ylim
        self.xlim = xlim

    def left(self):
        """! Move cursor position to the left by one."""
        if self.col > self.xlim[0]:
            self.col -= 1

    def right(self):
        """! Move cursor position to the right by one."""
        if self.col < self.xlim[1]:
            self.col += 1

    def up(self) -> int:
        """! Move cursor position up by one.

        @return (int) Returns 1 if cannot move up anymore, otherwise 0.
        """
        if self.row > self.ylim[0]:
            self.row -= 1
            return 0
        else:
            return 1

    def down(self) -> int:
        """! Move cursor position down by one.

        @return (int) Returns 1 if cannot move down anymore, otherwise 0.
        """
        if self.row < self.ylim[1]:
            self.row += 1
            return 0
        else:
            return 1

    def update_limits(self, ylim: list, xlim: list):
        """! Update the bounds of permitted x, y cursor positions.

        If the cursor will be out of the new bounds, it will be moved to within
        them.

        @param ylim (list) Length two list, [lower, upper], of bounds on y.
        @param xlim (list) Length two list, [lower, upper], of bounds on x.
        """
        self.ylim = ylim
        self.xlim = xlim

        if self.row < self.ylim[0]:
            self.row = self.ylim[0]
        elif self.row > self.ylim[1]:
            self.row = self.ylim[1]

        if self.col < self.xlim[0]:
            self.col = self.xlim[0]
        elif self.col > self.xlim[1]:
            self.col = self.xlim[1]

if __name__ == '__main__':
    app = H5TerminalApp(sys.argv[1])
    app.run()
