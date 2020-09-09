#!/usr/bin/env python

import tkinter.filedialog
import tkinter as tk
import tkinter.ttk as ttk


def main():

    # main app layout
    root = tk.Tk()
    root.title('RBDtector')
    mainframe = ttk.Frame(root, padding='10 10 10 10')
    mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.S, tk.E))

    # input directory section
    input_dir = tk.StringVar()
    input_dir.set('Select input folder')

    ttk.Label(
        mainframe,
        text='Input folder:'
    ).grid(row=0, column=0, padx=10, pady=10)

    input_button = ttk.Button(
        mainframe,
        textvariable=input_dir,
        command=lambda: select_folder_handler(input_dir)
    )
    input_button.grid(row=0, column=1, padx=10, pady=10)

    # output directory section
    output_dir = tk.StringVar()
    output_dir.set('Select output folder')

    ttk.Label(
        mainframe,
        text='Output folder:'
    ).grid(row=1, column=0, padx=10, pady=5)

    output_button = ttk.Button(
        mainframe,
        textvariable=output_dir,
        command=lambda: select_folder_handler(output_dir)
    )
    output_button.grid(row=1, column=1, padx=10, pady=5)

    # start button
    ttk.Button(
        mainframe,
        text='Start calculation',
        command=lambda: start_calculation(input_dir.get(), output_dir.get())
    ).grid(row=2, column=0, columnspan=2, padx=10, pady=12)

    # main loop
    root.mainloop()


def select_folder_handler(dir_string):
    directory = tk.filedialog.askdirectory()
    print(directory)
    if not directory:
        return
    else:
        dir_string.set(str(directory))


def start_calculation(input_dir, output_dir):
    print(input_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
