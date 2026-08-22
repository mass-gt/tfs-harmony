import multiprocessing as mp
import os.path
import sys
import traceback

from base64 import b64decode
from sys import argv
from tempfile import mkstemp
from threading import Thread
from typing import Dict
from zlib import decompress

# Tkinter is only needed for the interactive GUI.  Wrapping the imports in
# a try/except allows ``import mass_gt.tfs`` (and therefore library use /
# testing / CI) to succeed on headless machines where ``tkinter`` is not
# installed or no display server is available.
try:
    import tkinter as tk
    from tkinter.ttk import Progressbar
    _TKINTER_AVAILABLE = True
except ImportError:
    tk = None  # type: ignore[assignment]
    Progressbar = None  # type: ignore[assignment]
    _TKINTER_AVAILABLE = False

import mass_gt.calculation.fs.module_fs as module_fs
import mass_gt.calculation.outp.module_outp as module_outp
import mass_gt.calculation.parcel_dmnd.module_parcel_dmnd as module_parcel_dmnd
import mass_gt.calculation.parcel_schd.module_parcel_schd as module_parcel_schd
import mass_gt.calculation.service.module_service as module_service
import mass_gt.calculation.ship.module_ship as module_ship
import mass_gt.calculation.sif.module_sif as module_sif
import mass_gt.calculation.tour.module_tour as module_tour
import mass_gt.calculation.traf.module_traf as module_traf
import mass_gt.calculation.common.arguments as common_arguments

from mass_gt import settings
from mass_gt.calculation.common.dimensions import ModelDimensions
from mass_gt.support import get_logger


class Root:

    def __init__(self):
        '''
        Initialize a GUI object
        '''
        # Get directory of the script
        self.datapath = os.path.dirname(os.path.realpath(argv[0])).replace(os.sep, '/') + '/'

        self.moduleNames = [
            'FS', 'SIF',
            'SHIP', 'TOUR',
            'PARCEL_DMND', 'PARCEL_SCHD',
            'SERVICE',
            'TRAF',
            'OUTP']

        # Set graphics parameters
        self.width = 950
        self.height = 120
        self.bg = 'black'
        self.fg = 'white'
        self.font = 'Verdana'

        # Create a GUI window
        self.root = tk.Tk()
        self.root.title("Tactical Freight Simulator HARMONY")
        self.root.geometry(f'{self.width}x{self.height}+0+0')
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg=self.bg)
        self.canvas.place(x=0, y=0)
        self.statusBar = tk.Label(self.root, text="", anchor='w', borderwidth=0, fg='black')
        self.statusBar.place(x=2, y=self.height - 22, width=self.width, height=22)

        # Remove the default tkinter icon from the window
        icon = decompress(b64decode(
            'eJxjYGAEQgEBBiDJwZDBy' +
            'sAgxsDAoAHEQCEGBQaIOAg4sDIgACMUj4JRMApGwQgF/ykEAFXxQRc='))
        _, self.iconPath = mkstemp()
        with open(self.iconPath, 'wb') as iconFile:
            iconFile.write(icon)
        self.root.iconbitmap(bitmap=self.iconPath)

        # Create control file label and entry
        self.labelControlFile = tk.Label(
            self.root,
            text='Control file:',
            width=20,
            height=1,
            anchor='w',
            bg=self.bg,
            fg=self.fg,
            font=(self.font, 8))
        self.labelControlFile.place(x=5, y=9)
        self.controlFile = tk.StringVar(self.root, self.datapath)
        self.entryControlFile = tk.Entry(
            self.root,
            textvariable=self.controlFile,
            width=120,
            font=(self.font, 8))
        self.entryControlFile.place(x=80, y=10)

        # Create button to search for control file
        self.searchButton = tk.Button(
            self.root,
            text="...",
            command=self.file_dialog,
            width=2,
            font=(self.font, 5))
        self.searchButton.place(x=905, y=12)

        # Create button to run the main function
        self.runButton = tk.Button(
            self.root,
            text="Run",
            width=15,
            height=3,
            command=self.run_main,
            font=(self.font, 8))
        self.runButton.place(x=425, y=35)

        # Create a progress bar
        self.progressBar = Progressbar(
            self.root,
            length=int(self.width / 2))
        self.progressBar.place(x=int(self.width / 2), y=self.height - 22)

        # If the control file is passed as an argument
        # in a batch job or in the command line prompt
        if len(argv) > 1:
            self.controlFile.set(argv[1])
            self.run_main()

        # Keep GUI active until closed
        self.root.mainloop()

    def update_statusbar(self, text):
        self.statusBar.configure(text=text)

    def reset_statusbar(self, event=None):
        self.statusBar.configure(text="")

    def file_dialog(self):
        '''
        Open up a file dialog
        '''
        self.filename = tk.filedialog.askopenfilename(
            initialdir="/",
            title="Select the .ini control file",
            filetype=(("Control files (.ini)", "*.ini"), ("All files", "*.*")))
        self.controlFile.set(self.filename)

    def run_main(self, event=None):
        """
        Starts a thread which runs 'actually_run_main'.
        """
        Thread(target=self.actually_run_main, daemon=True).start()

    def actually_run_main(self):
        """
        Reads the control file and runs 'main'.

        Configuration loading (parsing + validation) is delegated to
        :func:`mass_gt.settings.load_settings`, which returns a varDict and a
        list of error messages rather than raising.  This keeps the GUI's
        "collect errors and show them in a dialog" behaviour while removing
        the duplicated parser that previously lived here.
        """
        # Run the modules or not, is set to False if, for example,
        # an input file could not be found
        run = True

        # Write a log file or not, is set to False if the specified
        # outputfolder does not exist
        writeLog = True

        # In this string we collect the error messages
        errorMessage = ""

        # A fully-initialised varDict (all-empty if loading fails).
        varDict = dict(
            (variableName, "")
            for variableName in common_arguments.variables
        )

        try:
            varDict, errors = settings.load_settings(
                self.controlFile.get(),
                module_names=self.moduleNames,
            )

            if errors:
                run = False
                errorMessage = "\n".join(errors)

            # Can't write a logfile if the outputfolder does not exist
            if (varDict['OUTPUTFOLDER']
                    and not os.path.isdir(varDict['OUTPUTFOLDER'])):
                writeLog = False

        except Exception:
            errorMessage = (
                'Could not find or read the following control file: ' +
                "'" + self.controlFile.get() + "'" +
                '\n\n' + str(sys.exc_info()[0]) +
                '\n\n' + str(traceback.format_exc()))
            run = False
            writeLog = False

        # Check for outputfolder files when not all modules are run
        run, errorMessage = self.check_for_output_files(varDict, run, errorMessage)

        # Check on text files with dimensions / categories
        if os.path.isdir(varDict['DIMFOLDER']):
            dimFilenames = [
                'combustion_type.txt',
                'emission_type.txt',
                'employment_sector.txt',
                'flow_type.txt',
                'logistic_segment.txt',
                'municipality.txt',
                'nstr.txt',
                'shipment_size.txt',
                'vehicle_type.txt']
            for filename in dimFilenames:
                if not os.path.isfile(varDict['DIMFOLDER'] + filename):
                    run = False
                    errorMessage = errorMessage + (
                        f"Expected a text file called '{filename}'" +
                        " in DIMFOLDER, but could not find it.")

        # Open the logfile and write the header, specified arguments and
        # possible error messages
        if writeLog:
            self.logger, log_stream_handler, log_file_handler = get_logger(varDict['OUTPUTFOLDER'])

            self.logger.debug('##################################################')
            self.logger.debug('### Tactical Freight Simulator HARMONY         ###')
            self.logger.debug('### Prototype version, January 2025            ###')
            self.logger.debug('##################################################')
            self.logger.debug('\n')

            self.logger.debug('##################################################')
            self.logger.debug('### Settings                                   ###')
            self.logger.debug('##################################################')

            self.logger.debug('Control file: ' + self.controlFile.get())
            for variableName, value in varDict.items():
                self.logger.debug(f"{variableName} = {value}")

                if variableName in common_arguments.obsolete and value != '':
                    self.logger.debug(
                        f'\t(Note that variable {variableName} has become obsolete.' +
                        ' It is not used in any of the modules anymore.)')

            if not run:
                self.logger.debug('###################################################')
                self.logger.debug('### Errors while reading control file           ###')
                self.logger.debug('###################################################')
                self.logger.debug(errorMessage)

        if run:
            self.statusBar.configure(text="Start calculations...")
            result = self.main(varDict)
            self.statusBar.configure(text="")

            if result[0] == 1:
                self.statusBar.configure(text=(
                    "Error in calculation! " +
                    "(Tool will be closed automatically after 20 seconds.)"))
                self.root.after(20000, lambda: self.root.destroy())

            else:
                self.statusBar.configure(text="Calculations finished.")
                self.progressBar['value'] = 100
                self.root.after(5000, lambda: self.root.destroy())

        else:
            self.statusBar.configure(text="Run was not started. See error message. ")
            errorMessage = (
                f"Could not start the run for the following reasons: \n\n {errorMessage}")
            self.error_screen(text=errorMessage, size=[950, 150])

        if writeLog:
            self.logger.removeHandler(log_stream_handler)
            self.logger.removeHandler(log_file_handler)

    def check_for_output_files(
        self, varDict: Dict[str, str], run: bool, errorMessage: str
    ):
        """
        Checks if necessary output files for modules which are not run are already in
        the output folder.
        """
        if os.path.isdir(varDict['OUTPUTFOLDER']):
            outFileChecks = [
                ["SHIP",
                 "SIF",
                 "CommodityMatrixNUTS3.csv"],
                ["TOUR",
                 "SHIP",
                 "Shipments_" + varDict['LABEL'] + ".csv"],
                ["PARCEL_SCHD",
                 "PARCEL_DMND",
                 "ParcelDemand_" + varDict['LABEL'] + ".csv"],
                ["TRAF",
                 "TOUR",
                 "Tours_" + varDict['LABEL'] + ".csv"],
                ["TRAF",
                 "TOUR",
                 "tripmatrix_" + varDict['LABEL'] + ".txt"],
                ["TRAF",
                 "TOUR",
                 "tripmatrix_" + varDict['LABEL'] + "_TOD0" + ".txt"],
                ["TRAF",
                 "PARCEL_SCHD",
                 "ParcelSchedule_" + varDict['LABEL'] + ".csv"],
                ["TRAF",
                 "PARCEL_SCHD",
                 "tripmatrix_parcels_" + varDict['LABEL'] + ".txt"],
                ["TRAF",
                 "PARCEL_SCHD",
                 "tripmatrix_parcels_" + varDict['LABEL'] + "_TOD0" + ".txt"],
                ["TRAF",
                 "SERVICE",
                 "TripsVanService.mtx"],
                ["TRAF",
                 "SERVICE",
                 "TripsVanConstruction.mtx"]]

            if varDict['FIRMS_REF'] == '':
                outFileChecks.append([
                    "SHIP",
                    "FS",
                    "Firms.csv"])

            for moduleWhichIsRun, moduleWhichIsNotRun, outfileToCheck in outFileChecks:
                if moduleWhichIsRun in varDict['MODULES']:
                    if moduleWhichIsNotRun not in varDict['MODULES']:
                        if not os.path.isfile(varDict['OUTPUTFOLDER'] + outfileToCheck):
                            run = False
                            errorMessage = errorMessage + (
                                'Module ' +
                                moduleWhichIsRun +
                                ' is run but preceding module ' +
                                moduleWhichIsNotRun +
                                ' is not run. ' +
                                'In that case the following file is expected' +
                                ' in the OUTPUTFOLDER, but it is not found: "' +
                                outfileToCheck + '".\n')

        return run, errorMessage

    def error_screen(
        self, text='', event=None, size=[950, 350], title='Error message'
    ):
        '''
        Pop up a window with an error message
        '''
        windowError = tk.Toplevel(self.root)
        windowError.title(title)
        windowError.geometry(f'{size[0]}x{size[1]}+0+{self.height+50}')
        windowError.minsize(width=size[0], height=size[1])
        windowError.iconbitmap(bitmap=self.iconPath)
        labelError = tk.Label(windowError, text=text, anchor='w', justify='left')
        labelError.place(x=10, y=10)

    def main(self, varDict):

        run = True
        result = [0, 0]

        self.logger.debug('###################################################')
        self.logger.debug('### Progress                                    ###')
        self.logger.debug('###################################################')

        dims = ModelDimensions(varDict['DIMFOLDER'])

        for module_abbr, module_name, module in (
            ("FS", "Firm Synthesis", module_fs),
            ("SIF", "Spatial Interaction Freight", module_sif),
            ("SHIP", "Shipment Synthesizer", module_ship),
            ("TOUR", "Tour Formation", module_tour),
            ("PARCEL_DMND", "Parcel Demand", module_parcel_dmnd),
            ("PARCEL_SCHD", "Parcel Scheduling", module_parcel_schd),
            ("SERVICE", "Vans Service/Construction", module_service),
            ("TRAF", "Traffic Assignment", module_traf),
            ("OUTP", "Output Indicators", module_outp),

        ):
            if run and module_abbr in varDict['MODULES']:
                self.logger.debug(module_name)
                self.update_statusbar(f'Running {module_name}...')

                result = module.actually_run_module(self, varDict, dims)

                if result[0] == 1:
                    errorMessage = (
                        f"\nError in {module_name} module!\n\n" +
                        f"{result[1][0]}\n{result[1][1]}\n\n"
                    )
                    self.error_screen(text=errorMessage)
                    self.logger.debug(errorMessage)
                    run = False
                else:
                    self.update_statusbar(f"{module}: Done")

        if run:
            self.logger.debug('Finished all calculations.')
            self.statusBar.configure(text='Finished all calculations.')

        return result


if __name__ == '__main__':
    mp.freeze_support()
    if not _TKINTER_AVAILABLE:
        sys.stderr.write(
            "Error: Tkinter is not available on this system. "
            "The GUI requires tkinter with a display server.\n"
            "Run headlessly instead:  python run_scenario.py --config <file.ini>\n")
        sys.exit(1)
    root = Root()
