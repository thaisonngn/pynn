
import curses
import threading

useTerminalPrint = False

class TerminalPrint:
    def __init__(self):
        if useTerminalPrint:
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()

        self.lock = threading.Lock()

        self.data = {}

    def add_output(self, audiodevice, start, end, text):
        audiodevice, start, end = str(audiodevice), int(start), int(end)
        with self.lock:
            if not useTerminalPrint:
                print(start,end,text)
            if audiodevice in self.data:
                if self.data[audiodevice][-1][0] == start:
                    self.data[audiodevice][-1] = (start,end,text)
                elif self.data[audiodevice][-1][0] < start:
                    self.data[audiodevice].append((start,end,text))
                else:
                    raise NotImplementedError
            else:
                self.data[audiodevice] = [(start,end,text)]

            if useTerminalPrint:
                self.print()

    def print(self):
        rows, columns = self.stdscr.getmaxyx()
        columns -= 1

        len_audiodevice = max(len(k) for k in self.data.keys())
        len_timestamps = max(len(str(time)) for v in self.data.values() for x in v for time in x[:2])

        ids = {k:len(v)-1 for k,v in self.data.items()}
        while True:
            key = None
            for k,v in ids.items():
                if key is None and ids[k]>=0:
                    key = k
                elif ids[k]>=0 and self.data[k][ids[k]][0]>self.data[key][ids[key]][0]:
                    key = k

            if key is None:
                break

            audiodevice = key
            start, end, text = self.data[key][ids[key]]

            length = len(text) + len_audiodevice + 2*len_timestamps + 9
            if length > rows * columns:
                for y in range(rows):
                    self.stdscr.addstr(y, 0, " "*columns)
                break
            lines = (length+columns-1) // columns
            print_text = ("%"+str(len_audiodevice)+"s | %0"+str(len_timestamps)+"d | %0"+str(len_timestamps)+"d | %s")%(audiodevice,start,end,text)
            print_text += " "*(lines*columns-len(print_text))
            for y in range(rows-lines,rows):
                self.stdscr.addstr(y, 0, print_text[:columns])
                print_text = print_text[columns:]
            rows -= lines
            ids[key] -= 1

        self.stdscr.refresh()
















