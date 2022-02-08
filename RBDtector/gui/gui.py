# tkinter
import tkinter.filedialog
from tkinter import ttk
import tkinter as tk
import tkinter.scrolledtext as st

# external
import logging

# internal
from app_logic.PSG_controller import PSGController, superdir_run, single_psg_run

# global variables
_input_placeholder = 'Select input folder'
_output_placeholder = 'Select output folder'


class Gui(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = ttk.Frame(self)
        container.grid(column=0, row=0, padx=10, pady=15, sticky=(tk.N, tk.W, tk.S, tk.E))

        self.dir_option = tk.StringVar()
        self.input_path = tk.StringVar()
        self.processing_variable = tk.StringVar()

        self.rb_frame = SingleOrMultipleSelectionFrame(container, self)
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

        self.processing_variable.set("RBDtection running. Please do not close this window.")
        self.update_idletasks()

        error_messages = 'GUI error - please contact developer.'
        if self.dir_option.get() == "single psg":
            error_messages = single_psg_run(self.input_path.get())
        elif self.dir_option.get() == "multiple psg":
            error_messages = superdir_run(self.input_path.get())

        if not error_messages:
            logging.info(f'All PSGs of {self.input_path.get()} were processed without errors.')
            error_messages = error_messages + f'All PSGs of {self.input_path.get()} were processed without errors.'

        self.processing_variable.set("")

        create_error_scrolled_text_toplevel(error_messages, parent_window)


class SingleOrMultipleSelectionFrame(ttk.Frame):
    """TKinter Frame to select whether a single or multiple PSGs should be evaluated."""
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

        button_select = ttk.Button(self, text="Select directory", command=get_dir_path)
        button_select.grid(row=0, column=1, padx=10, pady=10, sticky=(tk.W, tk.E))

        start_button = ttk.Button(
            self,
            text='Start calculation',
            command=lambda: self.controller.start_calculation(self.dir_path, parent_window=parent)
        )
        start_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        processing_label = ttk.Label(self, textvariable=controller.processing_variable, anchor='center')
        processing_label.grid(column=0, row=4, columnspan=2, rowspan=2, padx=10, pady=10)


def _select_folder_handler(dir_text_variable):
    directory = tk.filedialog.askdirectory()
    if not directory:
        return
    else:
        dir_text_variable.set(str(directory))


def create_error_scrolled_text_toplevel(error_messages, parent_window):
    top = tk.Toplevel(parent_window)
    tk.Label(top,
             text="RBDtector run report:"
             ).pack(side='top', fill='both', expand=True)
    error_scrolled_text = st.ScrolledText(
        top,
        width=30,
        height=40
    )
    error_scrolled_text.pack(side="top", fill='both', expand=True)

    error_scrolled_text.configure(state='normal')
    error_scrolled_text.insert('end', error_messages)
    error_scrolled_text.configure(state='disabled')

    return error_scrolled_text


if __name__ == '__main__':
    pass
