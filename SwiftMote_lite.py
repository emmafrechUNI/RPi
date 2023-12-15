import asyncio
import importlib
import json
import math
import struct
import warnings
import time
import threading
import re
import os
import pandas
import subprocess

from Electrode import *
from Data_processing import *
import pickle
import datetime
from Arduino_comm import*
from Tests import Test
from Plots import Plot
from values import Experiment_c
from values import Electrode_c
from values import Titration_c
from values import Test_c


address_default = 'FE:B7:22:CC:BA:8D'
uuids_default = ['340a1b80-cf4b-11e1-ac36-0002a5d5c51b', ]
write_uuid = '330a1b80-cf4b-11e1-ac36-0002a5d5c51b'

sender_battery_voltage = "battery_voltage"
SUPPRESS_WARNINGS = True
if SUPPRESS_WARNINGS:
    warnings.filterwarnings("ignore")


class App():

    def __init__(self, loop: asyncio.AbstractEventLoop):
        """:param loop: parent event loop for asynchronous execution, it is not unique in this app"""
        super().__init__()

        self.Titration_cBox = None
        self.new_data = {}
        self.loop = loop
        self.toggle_cursor = False
        self.width = 10

        self.path_list = []
        self.output_path = "/home/pi/Documents/RPi-laptop/output"
        self.data_path = "/home/pi/Documents/RPi-laptop/data"
        self.titration_path = "/home/pi/Documents/RPi-laptop/data_titration"

        self.electrode_list = {}
        self.titration_list = {}
        self.current_electrode = None
        self.raw_data_df = None
        self.update_raw_data_graph = False
        self.titration_df = None
        self.update_titration_graph = False
        self.to_update_plots = False
        self.datapoint_select_N = 0
        self.thread_result = -1
        self.data_received = False
        self.isHill = False
        self.check_params = False
        self.continuous_running = False
        self.first_gain = 0
        self.first_measure = True

        ################################################# Menu bar #######################################################
        self.tasks = {}  # list of tasks to be continuously executed at the same time (asynchronously, not in parallel)

        ###############################################################################################################
        self.plots = Plot()
        self.init_controls()

        #################################################################
        # Testing purposes
        self.time_type = True
        #################################################################
   

    # def create_directories(self, dir_list):
    #     for path in dir_list:
    #         if not os.path.exists(path):
    #             os.makedirs(path)

    def init_controls(self):

        """Initializes controls param master: reference to parent object """
        self.current_values = {}  # values from variable fields that can be sent to the device
        self.option_cbox_values = {}
        self.graph_values = {}
        self.calib_data_dict = {}


        ############################ Electrode selection with dropdown Experiment updating #############################################
        def update_electrode_list():
            elec_list = list(os.listdir(f"{self.data_path}"))
            self.Electrode_cBox["values"] = elec_list

        self.Electrode_cBox = Electrode_c()

        def load_electrode(name):
            with open(f"{self.data_path}/{name}", "rb") as f:
                self.electrode_list[name] = pickle.load(f)

        def load_titration(name):
            self.update_titration_graph = True
            with open(f"{self.titration_path}/{name}", "rb") as f:
                self.titration_list[name] = pandas.read_pickle(f)

        def set_new_electrode():
            electrode_name = self.Electrode_cBox.get()
            load_electrode(electrode_name)
            self.current_electrode = self.electrode_list[electrode_name]
            self.Experiment_cBox.set("")
            self.titration_df = None
            self.test_cBox.set("")
            self.raw_data_df = None
            self.to_update_plots = True

        def add_electrode():
            name = self.Electrode_cBox.get()
            print("added electrode: ", name)
            if os.path.isfile(f"{self.data_path}/{name}") or name in self.electrode_list.keys():
                messagebox.showerror('Error', f'{name} already exist, please modify name')
            elif name == "":
                messagebox.showerror('Error', f'please add electrode name')
            else:
                self.electrode_list[name] = Electrode(name)
                self.electrode_list[name].save(self.data_path)
                self.Electrode_cBox.set(self.electrode_list[name].name)
                set_new_electrode()

        def start_electrode():
            base_name = "electrode"
            number = 0
            while os.path.isfile(
                    f"{self.data_path}/{base_name}_{number}"):
                number += 1
            name = f"{base_name}_{number}"
            self.Electrode_cBox.set(name)
            add_electrode()
            electrode_name = self.Electrode_cBox.get()
            load_electrode(electrode_name)
            self.current_electrode = self.electrode_list[electrode_name]
            self.Experiment_cBox.set("")
            self.titration_df = None
            self.test_cBox.set("")
            self.raw_data_df = None
            self.to_update_plots = True

        # ############################# Experiment selection from selected Electrode ######################################################
        self.Experiment_cBox = Experiment_c()

        # ############################# Titration selection from file ######################################################
        def update_titration_list():
            titr_list = list(os.listdir(f"{self.titration_path}"))
            self.Titration_cBox['values'] = titr_list

        def force_set_titration():
            self.Titration_cBox.set("take6")
            file_name = self.Titration_cBox.get()
            match = re.match(r"^(.*)\((.*)\)\.csv$", file_name)
            if match:
                electrode_name = match.group(1)
                titration_name = match.group(2)
            else:
                titration_name = file_name
            self.titration_df = None
            if file_name:
                load_titration(titration_name)
                self.current_titration = self.titration_list[file_name]
                self.titration_df = self.current_titration.get_df().sort_values(by=["concentration"])
                if self.plots.prev_min_pt is None:
                    self.plots.min_pt = list(self.titration_df["concentration"])[0]
                    self.plots.max_pt = list(self.titration_df["concentration"])[-1]
                self.to_update_plots = True
            pass

        self.Titration_cBox = Titration_c()

        def add_experiment():
            if self.Experiment_cBox.get() == "":
                messagebox.showerror('Error', f'please add experiment name')
            else:
                name = self.Experiment_cBox.get()
                electrode = self.current_electrode
                if name in electrode.get_experiments():
                    messagebox.showerror('Error', f'{name} already exist, please modify name')
                else:
                    electrode.create_experiment(self.Experiment_cBox.get())
                    print(f"{name} created successfully")
                    electrode.save(self.data_path)

        def start_experiment():
            self.titration_df = None
            base_name = "experiment"
            number = 0
            while os.path.isfile(
                    f"{self.data_path}/{base_name}_{number}"):
                number += 1
            name = f"{base_name}_{number}"
            self.Experiment_cBox.set(name)
            experiment_name = self.Experiment_cBox.get()
            add_experiment()
            if experiment_name not in self.current_electrode.get_experiments():
                print(f"{experiment_name} doesn't exist")
            else:
                if self.current_electrode.get_tests(experiment_name)["Titration"].get_df().shape[0] > 0:
                    self.titration_df = self.current_electrode.get_tests(experiment_name)["Titration"].get_df().sort_values(by=["concentration"])
                    self.plots.min_pt = list(self.titration_df["concentration"])[0]
                    self.plots.max_pt = list(self.titration_df["concentration"])[-1]
                    self.update_titration_graph = True
            self.to_update_plots = True

        # ############################# Experiment type selection from selected Experiment ######################################################
        def update_test_list():
            self.test_cBox["values"] = list(self.current_electrode.get_tests(self.Experiment_cBox.get()).keys())

        self.test_cBox = Test_c()

        def set_test_graph():
            self.raw_data_df = self.current_electrode.get_tests(self.Experiment_cBox.get())[self.test_cBox.get()].get_df()
            self.update_raw_data_graph = True
            self.to_update_plots = True

        ############################################### Results ##################################################
        def update_plot():
            self.to_update_plots = True

        def refresh_slider(event):
            if self.test_cBox.get() != "":
                update_raw_data_graph(event)

        def update_raw_data_graph(event):
            self.update_raw_data_graph = True
            self.to_update_plots = True

        def Force_Create_SWV():
            return self.current_electrode.get_tests(self.Experiment_cBox.get())["SWV"]

        def run_test_thread(test, comport):
            self.thread_result = test.run_test(comport, 115200)

        def handle_test_results_delayed(test):
            while self.thread_result == -1:
                time.sleep(0.1)  # Sleep for 100ms
            handle_test_results(test)  # Call handle_test_results once result is not -1
            self.data_received = True

        def handle_test_results(test):
            if self.thread_result == 1:
                self.current_electrode.save(self.data_path)
                print("Test ran successfully")
                set_test_graph()
                self.calculate_concentration()
                update_plot()
            elif not self.continuous_running:
                messagebox.showerror('Error 2', self.thread_result.__str__())
            else:
                if self.thread_result == "Test stopped by user":
                    messagebox.showerror('Error', self.thread_result.__str__())
                else:
                    print('Error', self.thread_result.__str__())
            self.thread_result = -1

        def run_test(test: Test):
            self.test_runned = True
            self.thread_result == -1
            self.data_received = False
            ## ajout comm8
            comport = '/dev/'+ self.port_Kick
            threading.Thread(target=run_test_thread, args=(test, comport)).start()
            handle_test_results_delayed(test)

        def run_continuous_test(test:Test):
            test.stop_continuous = False
            self.continuous_running = True

            def run_test_and_update_gui():
                try:
                    run_test(test)
                    index = 0
                    if index < test.get_params()["RunTime"] and not test.stop_continuous:
                        while self.data_received == False:
                            pass
                        run_test_and_update_gui()
                    else:
                        self.continuous_running = False
                except Exception as e:
                    messagebox.showerror('Error 2', e.__str__())

            run_test_and_update_gui()  # Start the continuous test

        ###Ajout
        start_electrode()
        ######################################## Info screen params ########################################################################

        ################################################################################################################
        ###AJOUT
                #################################################################
        # Comm ports
        result = subprocess.run(['ls','-l','/dev/serial/by-id/'], stdout=subprocess.PIPE, text=True)
        output = result.stdout
        lines = output.split('\n')
        line_a = [line for line in lines if 'Arduino' in line]
        line_s = [line for line in lines if 'SparkFun' in line]

        self.port_arduino = str([line.split('->')[1].strip().split('/')[-1] for line in line_a]).strip("[]").strip("''")
        self.port_Kick = str([line.split('->')[1].strip().split('/')[-1] for line in line_s]).strip("[]").strip("''")
        self.arduino = Arduino_com(self.port_arduino)

        start_experiment()
        force_set_titration()
        self.isHill = False
        self.prepare_titration()
        test = Force_Create_SWV()
        self.test_cBox.set("SWV")
        time.sleep(1)
        run_continuous_test(test)

    def process_packet(self, data, time_delivered):
        """Processes a packet and returns transaction when finalized
        Worst part of the code, needs to be optimized, but I don't know how"""
        transaction_finished = None

        if "transaction_in_progress" not in dir(self):  # if not defined, create new transaction
            self.transaction_in_progress = Transaction()  # Transaction object, called only when the app starts
        #  -1: transaction is complete, -2: transaction was just completed because new transaction code was detected
        if self.transaction_in_progress.add_packet(data=data, time_delivered=time_delivered) in [-1, -2]:
            # if error, maybe it is beginning of a new transaction? Try to add packet second time
            transaction_finished = self.transaction_in_progress  # save reference of the previous transaction that has completed
            self.transaction_in_progress = Transaction()  # create a new Transaction object, used in the following transactions
            # -2 should never happen
            if self.transaction_in_progress.add_packet(data=data, time_delivered=time_delivered) == -1:
                # self.print("Error of starting new transaction, datahex: ", datahex)
                return -1, None

        if transaction_finished is not None and transaction_finished.finalized:
            return 0, transaction_finished
        if self.transaction_in_progress.finalized:  # Will execute only if 1 packet was expected, in theory
            return 0, self.transaction_in_progress
        else:
            # self.print("Transaction is not complete yet")
            return -2, None

    def prepare_titration(self):
        if self.titration_df is not None:
            if self.plots.prev_min_pt is not None:
                Plot.prev_min_pt = self.plots.min_pt
            if self.plots.prev_max_pt is not None:
                self.plots.max_pt = Plot.prev_max_pt

            if self.update_titration_graph:
                concentration = list(self.titration_df['concentration'])
                max_gain = []
                for i in range(len(self.titration_df['raw_voltages'].iloc[:])):
                    g = self.titration_df['peak_current'].iloc[i]
                    max_gain.append(g)
                # normalized
                first_peak_value = max_gain[0]
                max_gain = [x / first_peak_value for x in max_gain]
                max_gain = [(x - 1) * 100 for x in max_gain]
                if self.isHill:
                    if concentration[concentration.index(self.plots.min_pt)] < concentration[
                        concentration.index(self.plots.max_pt)]:
                        self.hf = HillFit(concentration[
                                          concentration.index(self.plots.min_pt):concentration.index(
                                              self.plots.max_pt) + 1],
                                          max_gain[concentration.index(self.plots.min_pt):concentration.index(
                                              self.plots.max_pt) + 1])
                        self.hf.fitting()
                    else:
                        conc = concentration[concentration.index(self.plots.min_pt):concentration.index(
                            self.plots.max_pt) + 1]
                        gain = max_gain[concentration.index(self.plots.min_pt):concentration.index(
                            self.plots.max_pt) + 1]
                        gain.reverse()
                        self.hf = HillFit(conc, gain)
                        self.hf.fitting()
                        self.hf.y_fit = np.flip(self.hf.y_fit)

                else:
                    self.linear_coefs = np.polyfit(concentration[
                                                   concentration.index(self.plots.min_pt):concentration.index(
                                                       self.plots.max_pt) + 1], max_gain[concentration.index(
                        self.plots.min_pt):concentration.index(self.plots.max_pt) + 1], 1)
                    fit_for_r2 = list(np.polyval(self.linear_coefs, concentration[concentration.index(
                        self.plots.min_pt):concentration.index(self.plots.max_pt) + 1]))
                    r_2 = r2_score(max_gain[concentration.index(self.plots.min_pt):concentration.index(
                        self.plots.max_pt) + 1], fit_for_r2)

                max_x = np.max(max_gain)
                min_x = np.min(max_gain)
                max_concentration = np.max(concentration)
                min_concentration = np.min(concentration)
        else:
            self.plots.reset_titration_graph()

    def calculate_concentration(self):
        try:
            i = len(self.raw_data_df['raw_voltages']) - 1  # Index for the last row
            g = self.raw_data_df['peak_current'].iloc[i]
            maximum_gain = g
            if self.first_measure:
                self.first_gain = maximum_gain
                self.first_measure = False
            # normalized
            maximum_gain = maximum_gain / self.first_gain
            maximum_gain = (maximum_gain - 1) * 100

            concentration = "NA"
            if self.isHill:
                top, bottom, ec50, nH = self.hf.params
                print("top ", top, "bottom ", bottom)
                print("max gain ", maximum_gain)
                if bottom <= maximum_gain <= top:
                    if not np.isnan(ec50 * (((bottom - maximum_gain) / (maximum_gain - top)) ** (1 / nH))):
                        concentration = ec50 * (((bottom - maximum_gain) / (maximum_gain - top)) ** (1 / nH))
                else:
                    concentration = "outRange"
            else:
                concentration = (maximum_gain - self.linear_coefs[1]) / self.linear_coefs[0]

            print("Concentration: ", concentration)
            c_format = "{:.2e}".format(concentration)
            self.arduino.send_concentration(c_format)
        except Exception:
            debug()
            pass


class StableWaiter:
    """Generates intervals between executions of certain parts of code;
    two methods: constant time, and percentage of total execution time """

    def __init__(self, interval=1.0, percentage_of_time=10):
        self.interval = interval
        self.duty_cycle = percentage_of_time / 100
        self.t1 = datetime.datetime.now(datetime.timezone.utc)

    async def wait_async(self):
        """Waits at approximately the same intervals independently of CPU speed
        (if CPU is faster than certain threshold)
        This is not mandatory, but makes UI smoother
        Can be roughly simplified with asyncio.sleep(interval)"""

        t2 = datetime.datetime.now(datetime.timezone.utc)
        previous_frame_time = ((t2 - self.t1).total_seconds())
        self.t1 = t2
        await asyncio.sleep(min((self.interval * 2) - previous_frame_time, self.interval))

    async def wait_async_constant_avg_time(self):
        """Waits constant average time as a percentage of total execution time
        O(1) avg difficulty, used to accelerate O(N^2) or worse algorithms by running them less frequently as N increases
        This is not mandatory, but makes UI smoother
        Can be roughly simplified with asyncio.sleep(interval), for example, it is used by autosaving in this app"""

        t2 = datetime.datetime.now(datetime.timezone.utc)
        previous_frame_time = ((t2 - self.t1).total_seconds())
        self.t1 = t2

        await asyncio.sleep(previous_frame_time / self.duty_cycle - previous_frame_time)


    def print(self, all_datapoints):
        pass


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    app = App(loop)
    loop.run_forever()
