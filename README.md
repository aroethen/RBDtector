# RBDtector

A python software tool to detect REM sleep behavioral disorder (RBD) in .edf files. 
Relies on .txt files of previously classified periods of sleep phases and arousals.

Table of contents:
1. [How to install](#how-to-install)
   1. [Windows](#windows)
   2. [Mac and Linux](#mac-and-linux)
2. [License](#license)
3. [Acknowledgements](#acknowledgements)

## How to install
### Windows
1. Download the zipped version of RBDtector here: [dist/RBDtector.zip](dist/RBDtector.zip)
2. Extract the zip file into a folder of your choice
3. Click the extracted RBDtector.exe to run RBDtector

In some cases, Windows Defender SmartScreen will flag RBDtector as an 'Application of unknown source'.
This warning can simply be clicked away. If you want to, you can also submit the file to [Windows' malware analysis](https://www.microsoft.com/en-us/wdsi/filesubmission) before trusting it.

### Mac and Linux
Installing RBDtector under MacOS or Linux requires the usage of simple terminal commands. 
General commands that work in the terminals of both operating systems can be found [here](https://help.ubuntu.com/community/UsingTheTerminal#File_.26_Directory_Commands).

0. Prerequisites: Ensure you have a version of Git and Python 3 installed
   (including pip and tkinter - usually already included in Python 3)
   - [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
   - [Python](https://www.python.org/downloads/)
   - [pip](https://pip.pypa.io/en/stable/installation/#supported-methods)
   - [tkinter](https://tkdocs.com/tutorial/install.html)
2. Open a terminal 
3. (optional: navigate to a folder into which you want to place RBDtector)
4. Enter the following commands line by line:
   1. Clone git repository `git clone https://github.com/aroethen/RBDtector.git`
   2. Install requirements `python3 -m pip -r RBDtector/requirements.txt`
   3. Change into folder with main.py `cd RBDtector/RBDtector`
   4. Run with `python3 main.py`


## License
This project is licensed under the MIT License (s. [LICENSE](LICENSE) file).

## Acknowledgements

This project gratefully uses the following third-party open source libraries:

| Library   | License   |
| ---       | ---       |
| [pyEDFlib](https://github.com/holgern/pyedflib)      | [BSD 2-Clause "Simplified" License](https://github.com/holgern/pyedflib/blob/master/LICENSE)    |
| [numpy](https://github.com/numpy/numpy)              | [BSD 3-Clause "New" or "Revised" License] (https://github.com/numpy/numpy/blob/main/LICENSE.txt)|
| [pandas](https://github.com/pandas-dev/pandas)       | [BSD 3-Clause "New" or "Revised" License] (https://github.com/pandas-dev/pandas/blob/main/LICENSE)|
| [scipy](https://github.com/scipy/scipy)              | [BSD 3-Clause "New" or "Revised" License] (https://github.com/scipy/scipy/blob/main/LICENSE.txt)|
| [openpyxl](https://foss.heptapod.net/openpyxl/openpyxl)       | [MIT License] (https://foss.heptapod.net/openpyxl/openpyxl/-/blob/branch/3.0/LICENCE.rst)|
