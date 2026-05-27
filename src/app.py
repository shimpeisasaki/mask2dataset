from __future__ import annotations

import tkinter as tk

from src.gui import AppGUI


def main() -> None:
    root = tk.Tk()
    AppGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
