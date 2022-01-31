# tkinter
import tkinter.messagebox
import tkinter.filedialog
from tkinter import ttk
import tkinter as tk
import tkinter.scrolledtext as st
from ttkthemes import ThemedTk

# external
import logging
import os
import traceback
from datetime import datetime

import pandas as pd

# internal
from app_logic.PSG_controller import PSGController
from util.error_for_display import ErrorForDisplay

# global variables
_input_placeholder = 'Select input folder'
_output_placeholder = 'Select output folder'


class Gui(ThemedTk):
    def __init__(self, *args, **kwargs):
        ThemedTk.__init__(self, *args, **kwargs)
        container = ttk.Frame(self)
        container.grid(column=0, row=0, padx=10, pady=15, sticky=(tk.N, tk.W, tk.S, tk.E))

        self.dir_option = tk.StringVar()
        self.input_path = tk.StringVar()

        self.rb_frame = RadiobuttonFrame(container, self)
        self.dir_selection_frame = DirSelectionFrame(container, self)

        self.rb_frame.grid(column=0, row=0, padx=10, pady=15, sticky=(tk.N, tk.W, tk.S, tk.E))
        self.rb_frame.tkraise()


    def process_rb_selection(self, selection_var):
        self.dir_option.set(selection_var.get())
        print(f"selected option {self.dir_option.get()}")

        self.dir_selection_frame.grid(column=0, row=0, padx=10, pady=15, sticky=(tk.N, tk.W, tk.S, tk.E))
        self.dir_selection_frame.tkraise()

    def start_calculation(self, input_path, parent_window):
        self.input_path.set(input_path.get())

        if self.dir_option.get() == "single psg":
            _ = PSGController.run_rbd_detection(self.input_path.get(), self.input_path.get())
        elif self.dir_option.get() == "multiple psg":
            superdir_run(self.input_path.get(), parent_window=parent_window)


class RadiobuttonFrame(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller

        selected_option = tk.StringVar()
        selected_option.set("multiple psg")


        r1 = ttk.Radiobutton(self, text="Superdirectory with multiple PSG subdirectories", variable=selected_option,
                             value="multiple psg")
        r1.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        r2 = ttk.Radiobutton(self, text="Single PSG directory", variable=selected_option,
                             value="single psg")
        r2.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)

        select_button = ttk.Button(
            self,
            text='Continue',
            command=lambda: controller.process_rb_selection(selected_option)
        )
        select_button.grid(row=3, column=0, columnspan=1, padx=10, pady=10)


class DirSelectionFrame(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        self.dir_path = tk.StringVar()

        def get_dir_path():
            self.dir_path.set(tk.filedialog.askdirectory())

        entry = ttk.Entry(self, textvariable=self.dir_path)
        entry.grid(row=0, column=0, padx=10, pady=10, sticky=(tk.W, tk.E))
        btnFind = ttk.Button(self, text="Select directory", command=get_dir_path)
        btnFind.grid(row=0, column=1, padx=10, pady=10, sticky=(tk.W, tk.E))

        start_button = ttk.Button(
            self,
            text='Start calculation',
            command=lambda: self.controller.start_calculation(self.dir_path, parent_window=parent)
        )
        start_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)


def _select_folder_handler(dir_text_variable):
    directory = tk.filedialog.askdirectory()
    if not directory:
        return
    else:
        dir_text_variable.set(str(directory))


# def _trigger_calculation(input_dir, output_dir):
#     if input_dir != _input_placeholder and output_dir != _output_placeholder:
#
#         logging.info('Start button clicked\n\t\t'
#                      'Selected input dir: {}\n\t\t'
#                      'Selected output dir: {}'.format(input_dir, output_dir))
#
#         try:
#             start_time = datetime.datetime.now()
#             PSGController.run_rbd_detection(input_dir, output_dir)
#             end_time = datetime.datetime.now()
#             print('Overall calculation time: ' + str(end_time - start_time))
#         except ErrorForDisplay as e:
#
#             tkinter.messagebox.showerror(
#                 title='Error',
#                 message=str(e)
#             )
#
#     else:
#         tkinter.messagebox.showinfo(
#             title='Invalid input',
#             message='Please select input and output directories'
#         )

def superdir_run(path, parent_window=None, dev_run: bool = False):

    dirlist = os.listdir(path)
    reading_problems = []
    df_out_combined = pd.DataFrame()
    df_channel_combinations_combined = pd.DataFrame()

    top = None
    error_scrolled_text = None

    no_error_yet = True
    first = True

    for child in dirlist:
        abs_child = os.path.normpath(os.path.join(path, child))
        if os.path.isdir(abs_child):
            try:
                df_out, df_channel_combinations = PSGController.run_rbd_detection(abs_child, abs_child)

                if first:
                    df_out_combined = df_out.copy()
                    df_channel_combinations_combined = df_channel_combinations.copy()

                    first = False
                else:
                    df_out_combined = pd.concat([df_out_combined, df_out], axis=1)
                    df_channel_combinations_combined = \
                        pd.concat([df_channel_combinations_combined, df_channel_combinations])

                # write intermediate combination results
                try:
                    df_out_combined = df_out_combined \
                        .reindex(['Signal', 'Global', 'EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.'], level=0)
                except:
                    continue

                df_out_combined.transpose().to_csv(
                    os.path.normpath(os.path.join(path, f'Intermediate_combined_results.csv')))
                df_channel_combinations_combined.to_csv(
                    os.path.normpath(os.path.join(path, f'Intermediate_combined_combinations.csv')))

            except (OSError, ErrorForDisplay) as e:
                if dev_run:
                    print(f'Expectable error in file {abs_child}:\n {e}')
                if not dev_run:

                    if no_error_yet:
                        error_scrolled_text = create_error_scrolled_text_toplevel(error_scrolled_text, parent_window)
                        no_error_yet = False

                    error_scrolled_text.configure(state='normal')
                    error_scrolled_text.insert('end', f'Error in file {abs_child}:\n {e}\n')
                    error_scrolled_text.insert('end', 'Full error message can be found in log file.\n\n')
                    error_scrolled_text.configure(state='disabled')

                logging.error(f'Error in file {abs_child}:\n {e}')
                logging.error(traceback.format_exc())
                reading_problems.append(abs_child)
                continue

            except BaseException as e:
                if dev_run:
                    print(f'Unexpected error in file {abs_child}:\n {e}')

                if not dev_run:

                    if no_error_yet:
                        error_scrolled_text = create_error_scrolled_text_toplevel(error_scrolled_text, parent_window)
                        no_error_yet = False

                    error_scrolled_text.configure(state='normal')
                    error_scrolled_text.insert('end', f'Unexpected error in file {abs_child}:\n {e}\n')
                    error_scrolled_text.insert('end', 'Full error message in log file. Please contact developer.\n\n')
                    error_scrolled_text.configure(state='disabled')

                logging.error(f'Unexpected error in file {abs_child}:\n {e}')
                logging.error(traceback.format_exc())
                reading_problems.append(abs_child)
                continue

    if not df_out_combined.empty:
        df_out_combined = df_out_combined \
            .reindex(['Signal', 'Global', 'EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.'], level=0)
    df_out_combined.transpose() \
        .to_excel(os.path.normpath(
        os.path.join(path,
                     f'RBDtector_combined_results_{str(datetime.now()).replace(" ", "_").replace(":", "-")}.xlsx')))
    df_channel_combinations_combined \
        .to_excel(os.path.normpath(
        os.path.join(path,
                     f'Channel_combinations_combined_{str(datetime.now()).replace(" ", "_").replace(":", "-")}.xlsx')))
    if len(reading_problems) != 0:
        logging.error(f'These files could not be processed: {reading_problems}')
        print(f'These files could not be read: {reading_problems}')
    else:
        logging.info(f'All subfolders of {path} were processed without errors.')


def create_error_scrolled_text_toplevel(error_scrolled_text, parent_window):
    top = tk.Toplevel(parent_window)
    tk.Label(top,
             text="Errors in the following PSGs:"
             ).pack(side='top', fill='both', expand=True)
    error_scrolled_text = st.ScrolledText(
        top,
        width=30,
        height=40
    )
    error_scrolled_text.pack(side="top", fill='both', expand=True)
    return error_scrolled_text


if __name__ == '__main__':
    pass
