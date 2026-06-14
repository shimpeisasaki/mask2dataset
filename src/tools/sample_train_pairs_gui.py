from __future__ import annotations

import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.tools.sample_train_pairs import sample_dataset_splits


class SamplePairsGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sample Dataset (Train/Val)")
        self.root.geometry("900x620")

        self.var_dataset_root = tk.StringVar(value="")
        self.var_out_root = tk.StringVar(value="")
        self.var_num_train = tk.StringVar(value="100")
        self.var_num_val = tk.StringVar(value="50")
        self.var_seed = tk.StringVar(value="42")
        self.var_dry_run = tk.BooleanVar(value=False)

        self._build_layout()

    def _build_layout(self) -> None:
        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        settings = ttk.LabelFrame(frm, text="Settings")
        settings.pack(fill="x")

        row1 = ttk.Frame(settings)
        row1.pack(fill="x", padx=8, pady=4)
        ttk.Label(row1, text="dataset root", width=14).pack(side="left")
        ttk.Entry(row1, textvariable=self.var_dataset_root).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row1, text="Browse", command=self._browse_dataset_root).pack(side="left")

        row2 = ttk.Frame(settings)
        row2.pack(fill="x", padx=8, pady=4)
        ttk.Label(row2, text="out root", width=14).pack(side="left")
        ttk.Entry(row2, textvariable=self.var_out_root).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row2, text="Browse", command=self._browse_out_root).pack(side="left")

        row3 = ttk.Frame(settings)
        row3.pack(fill="x", padx=8, pady=6)
        ttk.Label(row3, text="num train", width=14).pack(side="left")
        ttk.Entry(row3, textvariable=self.var_num_train, width=10).pack(side="left", padx=(0, 12))
        ttk.Label(row3, text="num val", width=8).pack(side="left")
        ttk.Entry(row3, textvariable=self.var_num_val, width=10).pack(side="left", padx=(0, 12))
        ttk.Label(row3, text="seed", width=8).pack(side="left")
        ttk.Entry(row3, textvariable=self.var_seed, width=10).pack(side="left")
        ttk.Checkbutton(row3, text="dry-run", variable=self.var_dry_run).pack(side="left", padx=(12, 0))

        row4 = ttk.Frame(settings)
        row4.pack(fill="x", padx=8, pady=(2, 8))
        self.btn_run = ttk.Button(row4, text="Run Sampling", command=self._on_run)
        self.btn_run.pack(side="left")

        log_frame = ttk.LabelFrame(frm, text="Log")
        log_frame.pack(fill="both", expand=True, pady=(10, 0))

        self.txt_log = tk.Text(log_frame, wrap="none", height=20)
        self.txt_log.pack(side="left", fill="both", expand=True)
        ybar = ttk.Scrollbar(log_frame, orient="vertical", command=self.txt_log.yview)
        ybar.pack(side="right", fill="y")
        self.txt_log.configure(yscrollcommand=ybar.set)

    def _append_log(self, msg: str) -> None:
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")

    def _browse_dataset_root(self) -> None:
        p = filedialog.askdirectory(title="Select dataset root")
        if p:
            self.var_dataset_root.set(p)

    def _browse_out_root(self) -> None:
        p = filedialog.askdirectory(title="Select output root")
        if p:
            self.var_out_root.set(p)

    def _on_run(self) -> None:
        try:
            dataset_root = Path(self.var_dataset_root.get().strip())
            out_root = Path(self.var_out_root.get().strip())
            num_train = int(self.var_num_train.get().strip())
            num_val = int(self.var_num_val.get().strip())
            seed = int(self.var_seed.get().strip())
            dry_run = bool(self.var_dry_run.get())
        except ValueError:
            messagebox.showerror("Invalid input", "num train / num val / seed は整数で入力してください")
            return

        if num_train < 0 or num_val < 0:
            messagebox.showerror("Invalid input", "num train / num val は 0 以上で入力してください")
            return

        if not str(dataset_root) or not str(out_root):
            messagebox.showerror("Missing input", "すべてのフォルダを指定してください")
            return

        self.btn_run.configure(state="disabled")
        self._append_log("=== start ===")

        def worker() -> None:
            try:
                sample_dataset_splits(
                    dataset_root=dataset_root,
                    out_root=out_root,
                    num_train=num_train,
                    num_val=num_val,
                    seed=seed,
                    dry_run=dry_run,
                    logger=lambda s: self.root.after(0, self._append_log, s),
                )
                self.root.after(0, messagebox.showinfo, "Completed", "サンプリングが完了しました")
            except Exception as e:
                self.root.after(0, messagebox.showerror, "Error", str(e))
            finally:
                self.root.after(0, self.btn_run.configure, {"state": "normal"})

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    SamplePairsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
