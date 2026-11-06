#!/usr/bin/env python3
"""Desktop demonstration for completed utterances and frozen v10 checkpoints."""
from __future__ import annotations
import argparse
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Saved v10 .keras or .joblib checkpoint')
    parser.add_argument('--metadata', help='feature_metadata.json for an acoustic SVM or CNN')
    parser.add_argument('--encoder', help='Verified emotion2vec_base directory; --model then identifies its trained head')
    args = parser.parse_args()
    if not args.encoder and not args.metadata:
        parser.error('--metadata is required unless --encoder is provided')
    if not (Path(__file__).parent/'emotion2vec_predictor_v10.py').exists():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code_review'))
    root = tk.Tk()
    root.title('Speech Emotion Recognition')
    root.geometry('860x570')
    root.minsize(740, 480)
    style = ttk.Style(root)
    style.configure('TLabel', font=('Arial', 12))
    style.configure('Title.TLabel', font=('Arial', 23, 'bold'))
    style.configure('Result.TLabel', font=('Arial', 28, 'bold'))
    frame = ttk.Frame(root, padding=28)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='Speech emotion recognition', style='Title.TLabel').pack(anchor='w')
    ttk.Label(frame, text='Analyze a completed speech recording', padding=(0, 8, 0, 14)).pack(anchor='w')
    backend_name = 'Frozen emotion2vec with a trained classifier' if args.encoder else 'Acoustic SVM or compact CNN'
    ttk.Label(frame, text=backend_name).pack(anchor='w')
    status = tk.StringVar(value='Loading the model…')
    filename = tk.StringVar(value='Choose an audio file after the model loads.')
    result = tk.StringVar(value='')
    detail = tk.StringVar(value='')
    ttk.Label(frame, textvariable=filename, wraplength=760).pack(anchor='w', pady=8)
    button = ttk.Button(frame, text='Open speech recording', state='disabled')
    button.pack(anchor='w', pady=8)
    ttk.Label(frame, textvariable=result, style='Result.TLabel').pack(anchor='w', pady=(24, 12))
    scores = ttk.Treeview(frame, columns=('emotion', 'value'), show='headings', height=4)
    scores.heading('emotion', text='Emotion')
    scores.heading('value', text='Model score')
    scores.column('emotion', width=340)
    scores.column('value', width=210, anchor='center')
    scores.pack(fill='x', pady=6)
    ttk.Label(frame, textvariable=detail, wraplength=760).pack(anchor='w', pady=10)
    ttk.Label(frame, textvariable=status, wraplength=760).pack(anchor='w', pady=6)
    ttk.Label(frame, text='Four acted-speech categories. Scores are uncalibrated and do not measure accuracy.',
              wraplength=760).pack(anchor='w', side='bottom', pady=(8, 0))
    events = queue.Queue()
    state = {'predictor': None}

    def load_model():
        try:
            if args.encoder:
                from emotion2vec_predictor_v10 import Emotion2vecPredictor
                predictor = Emotion2vecPredictor(args.encoder, args.model)
            else:
                from ser_infer import Predictor
                predictor = Predictor(args.model, args.metadata)
            events.put(('ready', predictor))
        except Exception as exc:
            events.put(('error', str(exc)))

    def run_file(path):
        try:
            events.put(('prediction', state['predictor'].predict_file(path)))
        except Exception as exc:
            events.put(('error', str(exc)))

    def choose():
        path = filedialog.askopenfilename(title='Choose a speech recording',
                                         filetypes=[('Audio', '*.wav *.flac *.aiff *.aif'), ('All files', '*')])
        if not path:
            return
        filename.set(Path(path).name)
        result.set('')
        detail.set('')
        for row in scores.get_children(): scores.delete(row)
        button.configure(state='disabled')
        status.set('Analyzing the complete recording…')
        threading.Thread(target=run_file, args=(path,), daemon=True).start()

    def poll():
        try:
            while True:
                kind, payload = events.get_nowait()
                if kind == 'ready':
                    state['predictor'] = payload
                    status.set('Ready')
                    button.configure(state='normal')
                elif kind == 'prediction':
                    pred, timing, info = payload
                    result.set(pred['label'].capitalize())
                    for label, value in pred['scores'].items():
                        scores.insert('', 'end', values=(label.capitalize(), f'{value:.4f}'))
                    detail.set(pred['score_interpretation'])
                    status.set(f"Recording {info['duration_s']:.2f} s · Processing {timing['total_ready_file_ms']:.1f} ms")
                    button.configure(state='normal')
                else:
                    status.set('Could not analyze the recording: ' + payload)
                    if state['predictor'] is not None: button.configure(state='normal')
        except queue.Empty:
            pass
        root.after(100, poll)

    button.configure(command=choose)
    threading.Thread(target=load_model, daemon=True).start()
    root.after(100, poll)
    root.mainloop()


if __name__ == '__main__':
    main()
