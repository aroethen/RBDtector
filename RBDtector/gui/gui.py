# tkinter
import tkinter.messagebox
import tkinter.filedialog
from tkinter import ttk
import tkinter as tk
from ttkthemes import ThemedTk

# external
import logging

# internal
from app_logic.PSG_data import PSGData
from util.error_for_display import ErrorForDisplay

# global variables
_input_placeholder = 'Select input folder'
_output_placeholder = 'Select output folder'


def start_gui():
        # main app layout
        root = ThemedTk(theme='scidblue')       # okay looking ones: breeze, scidblue, scidsand, yaru
        root.title('RBDtector')
        mainframe = ttk.Frame(root)
        mainframe.grid(column=0, row=0, padx=10, pady=15, sticky=(tk.N, tk.W, tk.S, tk.E))

        # input directory section
        input_dir = tk.StringVar()
        input_dir.set(_input_placeholder)

        ttk.Label(
            mainframe,
            text='Input folder:'
        ).grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        input_button = ttk.Button(
            mainframe,
            textvariable=input_dir,
            command=lambda: _select_folder_handler(input_dir),
        )
        input_button.grid(row=0, column=1, padx=10, pady=10, sticky=(tk.W, tk.E))

        # output directory section
        output_dir = tk.StringVar()
        output_dir.set(_output_placeholder)

        ttk.Label(
            mainframe,
            text='Output folder:'
        ).grid(row=1, column=0, padx=10, pady=5)

        output_button = ttk.Button(
            mainframe,
            textvariable=output_dir,
            command=lambda: _select_folder_handler(output_dir),
        )
        output_button.grid(row=1, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        # start button
        ttk.Button(
            mainframe,
            text='Start calculation',
            command=lambda: _trigger_calculation(input_dir.get(), output_dir.get()),
        ).grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        # main loop
        root.columnconfigure(0, weight=1)
        mainframe.columnconfigure(1, weight=1)
        root.mainloop()


def _select_folder_handler(dir_string):
    directory = tk.filedialog.askdirectory()
    if not directory:
        return
    else:
        dir_string.set(str(directory))


def _trigger_calculation(input_dir, output_dir):
    if input_dir != _input_placeholder and output_dir != _output_placeholder:

        logging.info('Start button clicked\n\t\t'
                     'Selected input dir: {}\n\t\t'
                     'Selected output dir: {}'.format(input_dir, output_dir))

        try:
            data = PSGData(input_dir, output_dir)
            data.generate_output()

        except ErrorForDisplay as e:

            tkinter.messagebox.showerror(
                title='Error',
                message=str(e)
            )

    else:
        tkinter.messagebox.showinfo(
            title='Invalid input',
            message='Please select input and output directories'
        )


if __name__ == '__main__':
    pass
