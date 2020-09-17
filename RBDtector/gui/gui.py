# tkinter
import tkinter.messagebox
import tkinter.filedialog
import tkinter as tk

# external
import logging

# internal
import main

# global variables
_input_placeholder = 'Select input folder'
_output_placeholder = 'Select output folder'


def start_gui():
    try:
        # main app layout
        root = tk.Tk()
        root.title('RBDtector')
        mainframe = tk.Frame(root)
        mainframe.grid(column=0, row=0, padx=10, pady=15, sticky=(tk.N, tk.W, tk.S, tk.E))

        # input directory section
        input_dir = tk.StringVar()
        input_dir.set(_input_placeholder)

        tk.Label(
            mainframe,
            text='Input folder:'
        ).grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        input_button = tk.Button(
            mainframe,
            textvariable=input_dir,
            command=lambda: _select_folder_handler(input_dir),
        )
        input_button.grid(row=0, column=1, padx=10, pady=10, sticky=(tk.W, tk.E))

        # output directory section
        output_dir = tk.StringVar()
        output_dir.set(_output_placeholder)

        tk.Label(
            mainframe,
            text='Output folder:'
        ).grid(row=1, column=0, padx=10, pady=5)

        output_button = tk.Button(
            mainframe,
            textvariable=output_dir,
            command=lambda: _select_folder_handler(output_dir),
        )
        output_button.grid(row=1, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        # start button
        tk.Button(
            mainframe,
            text='Start calculation',
            command=lambda: _trigger_calculation(input_dir.get(), output_dir.get()),
            height=1,
            width=35
        ).grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        # main loop
        root.columnconfigure(0, weight=1)
        mainframe.columnconfigure(1, weight=1)
        root.mainloop()

    except Exception:
        # TODO: Die ganze Exception hier entfernen - nur zum Debugging waehrend dem Schreiben gedacht!!!
        logging.exception('Exception in GUI mainloop. Stack trace:')


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

        main.calculate_results(input_dir, output_dir)

    else:
        tkinter.messagebox.showinfo(
            title='Invalid input',
            message='Please select input and output directories'
        )


if __name__ == '__main__':
    pass
