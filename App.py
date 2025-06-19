import numpy as np
import json
import math
import os
import pickle
import requests
import csv
import sys
from datetime import datetime, timedelta, timezone
import emission_measure as emt


class LiveApp:
    def __init__(self, use_test_flux=False, print_output=False, use_secondary_source=False):

        self.use_test_flux = use_test_flux
        self.print_output = print_output
        if not use_secondary_source:
            self.six_hour_flux_url = r"https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
            self.seven_day_flux_url = r"https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
        else:
            self.six_hour_flux_url = r"https://services.swpc.noaa.gov/json/goes/secondary/xrays-6-hour.json"
            self.seven_day_flux_url = r"https://services.swpc.noaa.gov/json/goes/secondary/xrays-7-day.json"
        self.app_start_time = datetime.strftime(datetime.now(timezone.utc), "%Y-%m-%dT%H-%M-%S")
        if not os.path.exists(os.path.join("Artifacts", self.app_start_time)):
            os.mkdir(os.path.join("Artifacts", self.app_start_time))
        self.load_flux()
        self.load_models()
        self.targets = ["<C5", "C, >=C5", "M", "X"]
        self.timestamps = []
        self.params = ["XRSA_flux",
                       "XRSA_flux_1_minute_difference",
                       "XRSA_flux_2_minute_difference",
                       "XRSA_flux_3_minute_difference",
                       "XRSA_flux_4_minute_difference",
                       "XRSA_flux_5_minute_difference",
                       "XRSB_flux",
                       "XRSB_flux_1_minute_difference",
                       "XRSB_flux_2_minute_difference",
                       "XRSB_flux_3_minute_difference",
                       "XRSB_flux_4_minute_difference",
                       "XRSB_flux_5_minute_difference",
                       "Temperature",
                       "Temperature_1_minute_from_xrs_difference",
                       "Temperature_2_minute_from_xrs_difference",
                       "Temperature_3_minute_from_xrs_difference",
                       "Temperature_4_minute_from_xrs_difference",
                       "Temperature_5_minute_from_xrs_difference",
                       "EmissionMeasure",
                       "EmissionMeasure_1_minute_from_xrs_difference",
                       "EmissionMeasure_2_minute_from_xrs_difference",
                       "EmissionMeasure_3_minute_from_xrs_difference",
                       "EmissionMeasure_4_minute_from_xrs_difference",
                       "EmissionMeasure_5_minute_from_xrs_difference",
                       "Temperature_1_minute_difference",
                       "Temperature_2_minute_difference",
                       "Temperature_3_minute_difference",
                       "Temperature_4_minute_difference",
                       "Temperature_5_minute_difference",
                       "EmissionMeasure_1_minute_difference",
                       "EmissionMeasure_2_minute_difference",
                       "EmissionMeasure_3_minute_difference",
                       "EmissionMeasure_4_minute_difference",
                       "EmissionMeasure_5_minute_difference"
                       ]

        self.input_data = {}  # data to feed into models
        self.real_data_timestamps = []  # UTC time data was received
        self.last_good_param = {}  # value of last non-NaN data point for each param
        self.last_value_is_good = {}  # bool, whether or not the last datum for each param is not NaN
        for param in self.params:
            self.input_data[param] = [0] * 5
            self.last_good_param[param] = 0
            self.last_value_is_good[param] = True

        self.prediction_history = {}  # store prediction history for all classes
        for i in range(-5, 16):
            self.prediction_history[i] = {}
            for target in self.targets:
                self.prediction_history[i][target] = None

        self.flare_start_time = None  # True UTC start time of the most recent flare

        self.minutes_since_start = -5
        if self.use_test_flux:
            self.expected_time_tag = datetime.strptime(self.flux[0]['time_tag'], "%Y-%m-%dT%H:%M:00Z")
            self.test_line_idx = 0
            self.offset = -200
        else:
            self.expected_time_tag = datetime.strptime(self.flux[-16]['time_tag'], "%Y-%m-%dT%H:%M:00Z")
            self.offset = -20

        self.satellite = None  # active GOES satellite

        # in theory only need to be 21 (number of models), but still can be left with 21 NaN values in input data
        self.MINUTES_OF_DATA_TO_KEEP = 50

        self.flare_mode_unlocked = False  # are we allowed to go into flare mode?

        self.select_mode("Warmup")

    def create_output_directory(self):
        """Create folder to save outputs"""
        if not os.path.exists("Artifacts"):
            os.mkdir("Artifacts")

        if not os.path.exists(self.app_start_time):
            os.mkdir(self.app_start_time)

    def save_seven_day_flux_history_json(self):
        """Save 7-day flux history from NOAA/GOES"""
        # save off 7-day JSON flux
        out_dir = os.path.join("Artifacts", self.app_start_time)
        resp = requests.get(self.six_hour_flux_url)
        with open(os.path.join(out_dir, "xrays-7-day.json"), 'w') as f:
            json.dump(json.loads(resp.content.decode()), f, indent=4)

    def load_flux(self):
        """Read data, either live or froma  test JSON"""
        if self.use_test_flux:
            flux_subset_filepath = r"test_flux.json"
            with open(flux_subset_filepath, 'rb') as f:
                self.flux = json.load(f)
                
        else:  # live
            resp = requests.get(self.six_hour_flux_url)
            self.flux = json.loads(resp.content.decode())

    def load_models(self):
        """Load prediction models"""
        self.models = {}
        for minutes_since_start in range(-5, 16):
            model_path = os.path.join(f"Models", f"trained_{minutes_since_start}_minutes_since_start")
            with open(model_path, 'rb') as f:
                self.models[minutes_since_start] = pickle.load(f)

    def select_mode(self, mode_name):
        """Toggle operation mode - used primarily as bools for GUI layout"""
        self.in_startup_mode = mode_name == "Warmup"
        self.in_standby_mode = mode_name == "Standby"
        self.in_flare_mode = mode_name == "Flare"

    def unlock_flare_mode(self):
        """Allow app to go into flare mode, assured t=0 will be aligned with the start of a flare"""
        if self.input_data["XRSB_flux"][0] < self.input_data["XRSB_flux"][1] < self.input_data["XRSB_flux"][2] < self.input_data["XRSB_flux"][3]:
            self.flare_mode_unlocked = True

    def delete_old_data(self):
        """Remove the last elements of the input data lists so there's only the last X minutes"""
        for param, data_history in self.input_data.items():
            self.input_data[param] = self.input_data[param][:self.MINUTES_OF_DATA_TO_KEEP]

        self.real_data_timestamps = self.real_data_timestamps[:self.MINUTES_OF_DATA_TO_KEEP]
        self.timestamps = self.timestamps[:self.MINUTES_OF_DATA_TO_KEEP]

    def update_live_data(self, expected_time_tag):
        """Read in teh next data point and parse it for derived datums"""
        xrsa_data_appended = False
        xrsb_data_appended = False

        expected_time_tag = datetime.strftime(expected_time_tag, "%Y-%m-%dT%H:%M:00Z")

        while not xrsa_data_appended and not xrsb_data_appended:

            if not self.use_test_flux:
                resp = requests.get(self.six_hour_flux_url)
                self.flux = json.loads(resp.content.decode())
                datums = [x for x in reversed(self.flux[self.offset:]) if x['time_tag'] == expected_time_tag]
            else:  # test
                datums = [x for x in reversed(self.flux[self.offset:]) if x['time_tag'] == expected_time_tag]
                if not datums:
                    if self.print_output:
                        print("Out of data! Test finished! Quitting! Hope it worked!")
                    sys.exit(0)

            # check for expected newest time tag
            for datum in datums:
                if datum['energy'] == "0.05-0.4nm":  # XRSA
                    self.input_data["XRSA_flux"].insert(0, datum['observed_flux'])
                    for n in range(1, 6):
                        if datum['observed_flux'] is not None and datum['observed_flux'] != 0 and self.input_data['XRSA_flux'][n] is not None:
                            self.input_data[f"XRSA_flux_{n}_minute_difference"].insert(0, datum['observed_flux'] -
                                                                                       self.input_data["XRSA_flux"][n])
                            self.last_value_is_good[f"XRSA_flux_{n}_minute_difference"] = True
                        else:
                            self.input_data[f"XRSA_flux_{n}_minute_difference"].insert(0, None)
                            self.last_good_param[f"XRSA_flux_{n}_minute_difference"] += 1
                            self.last_value_is_good[f"XRSA_flux_{n}_minute_difference"] = False
                    if self.print_output:
                        print(f"Read XRSA at {datum['time_tag']}")
                    xrsa_data_appended = True
                if datum['energy'] == "0.1-0.8nm":  # XRSB
                    self.input_data["XRSB_flux"].insert(0, datum['observed_flux'])
                    for n in range(1, 6):
                        if datum['observed_flux'] is not None and datum['observed_flux'] != 0 and self.input_data['XRSB_flux'][n] is not None:
                            self.input_data[f"XRSB_flux_{n}_minute_difference"].insert(0, datum['observed_flux'] -
                                                                                       self.input_data["XRSB_flux"][n])
                            self.last_value_is_good[f"XRSB_flux_{n}_minute_difference"] = True
                        else:
                            self.input_data[f"XRSB_flux_{n}_minute_difference"].insert(0, None)
                            self.last_good_param[f"XRSB_flux_{n}_minute_difference"] += 1
                            self.last_value_is_good[f"XRSB_flux_{n}_minute_difference"] = False
                    # arbitrarily update satellite here
                    self.satellite = datum['satellite']
                    if self.print_output:
                        print(f"Read XRSB at {datum['time_tag']}")
                    xrsb_data_appended = True

        self.timestamps.insert(0, expected_time_tag)
        self.real_data_timestamps.insert(0, datetime.now(timezone.utc))
        if len(self.real_data_timestamps) > 1 and self.print_output:
            print(f"Time since last datapoint collection: {(self.real_data_timestamps[0] - self.real_data_timestamps[1]).total_seconds()} seconds")

        # Temperature/Emission Measure
        em, temp = emt.compute_goes_emission_measure(xrsa_data=self.input_data[f"XRSA_flux"][0],
                                                     xrsb_data=self.input_data[f"XRSB_flux"][0],
                                                     goes_sat=self.satellite)
        if not math.isnan(temp[0]) and temp[0] is not None:
            self.input_data["Temperature"].insert(0, temp[0])
            self.last_value_is_good["Temperature"] = True
        else:
            self.input_data["Temperature"].insert(0, None)
            self.last_good_param["Temperature"] += 1
            self.last_value_is_good["Temperature"] = False

        if not math.isnan(em[0]) and em[0] is not None:
            self.input_data[f"EmissionMeasure"].insert(0, em[0] / 10 ** 30)
            self.last_value_is_good["EmissionMeasure"] = True
        else:
            self.input_data[f"EmissionMeasure"].insert(0, None)
            self.last_good_param["EmissionMeasure"] += 1
            self.last_value_is_good["EmissionMeasure"] = False

        for n in range(1, 6):
            em_n_minute_diff, temp_n_minute_diff = emt.compute_goes_emission_measure(
                xrsa_data=self.input_data[f"XRSA_flux_{n}_minute_difference"][0],
                xrsb_data=self.input_data[f"XRSB_flux_{n}_minute_difference"][0],
                goes_sat=self.satellite)

            if not math.isnan(em_n_minute_diff[0]) and em_n_minute_diff[0] is not None:
                self.input_data[f"EmissionMeasure_{n}_minute_from_xrs_difference"].insert(0,
                                                                                          em_n_minute_diff[
                                                                                              0] / 10 ** 30)
                self.last_value_is_good[f"EmissionMeasure_{n}_minute_from_xrs_difference"] = True
            else:
                self.input_data[f"EmissionMeasure_{n}_minute_from_xrs_difference"].insert(0, None)
                self.last_good_param[f"EmissionMeasure_{n}_minute_from_xrs_difference"] += 1
                self.last_value_is_good[f"EmissionMeasure_{n}_minute_from_xrs_difference"] = False

            if not math.isnan(temp_n_minute_diff[0]) and temp_n_minute_diff[0] is not None:
                self.input_data[f"Temperature_{n}_minute_from_xrs_difference"].insert(0,
                                                                                      temp_n_minute_diff[0])
                self.last_value_is_good[f"Temperature_{n}_minute_from_xrs_difference"] = True
            else:
                self.input_data[f"Temperature_{n}_minute_from_xrs_difference"].insert(0, None)
                self.last_good_param[f"Temperature_{n}_minute_from_xrs_difference"] += 1
                self.last_value_is_good[f"Temperature_{n}_minute_from_xrs_difference"] = False

        # Non-XRS Differences
        for n in range(1, 6):
            if self.input_data['Temperature'][0] is not None and self.input_data['Temperature'][n] is not None:
                self.input_data[f"Temperature_{n}_minute_difference"].insert(0,
                                                                             self.input_data['Temperature'][0] -
                                                                             self.input_data['Temperature'][n])
                self.last_value_is_good[f"Temperature_{n}_minute_difference"] = True
            else:
                self.input_data[f"Temperature_{n}_minute_difference"].insert(0, None)
                self.last_good_param[f"Temperature_{n}_minute_difference"] += 1
                self.last_value_is_good[f"Temperature_{n}_minute_difference"] = False

            if self.input_data['EmissionMeasure'][0] is not None and self.input_data['EmissionMeasure'][n] is not None:
                self.input_data[f"EmissionMeasure_{n}_minute_difference"].insert(0, self.input_data['EmissionMeasure'][0] - self.input_data['EmissionMeasure'][n])
                self.last_value_is_good[f"EmissionMeasure_{n}_minute_difference"] = True
            else:
                self.input_data[f"EmissionMeasure_{n}_minute_difference"].insert(0, None)
                self.last_good_param[f"EmissionMeasure_{n}_minute_difference"] += 1
                self.last_value_is_good[f"EmissionMeasure_{n}_minute_difference"] = False

        # check if linear interpolation is needed
        for param in self.params:
            if self.last_good_param[param] != 0:
                if not self.last_value_is_good[param]:  # extend
                    for idx, value in enumerate(self.input_data[param]):
                        if value is None:
                            self.input_data[param][idx] = np.interp(idx,  # index to interpolate for
                                                                    [x for x in range(len(self.input_data[param])) if self.input_data[param][x] != None],  # indicies associated with known values
                                                                    [x for x in self.input_data[param] if x != None])  # known values
                else:  # sandwich mode
                    for idx, value in enumerate(self.input_data[param]):
                        if 0 < idx < self.last_good_param[param] + 1:  # if we previously extended for this step
                            self.input_data[param][idx] = np.interp(idx,  # index to interpolate for
                                                                    [0, self.last_good_param[param] + 1],  # indicies associated with known values
                                                                    [self.input_data[param][0], self.input_data[param][self.last_good_param[param] + 1]])  # known values
                    self.last_good_param[param] = 0  # reset

    def get_values_for_prediction(self, offset=0):
        """Isolate the most recent sample from each variable to be used for prediction"""

        x = []
        for param in self.params:
            x.append(self.input_data[param][offset])

        return np.array(x)

    def predict(self, effective_prediction_time, x):
        """Predict flux!"""
        predicted_probabilities = self.models[effective_prediction_time].predict_proba(x.reshape(1, -1))
        for idx, target in enumerate(self.targets):
            self.prediction_history[effective_prediction_time][target] = predicted_probabilities[0, idx]
        if self.prediction_history[effective_prediction_time]["<C5"] >= 0.5:
            pred_string = f"Prediction ({effective_prediction_time}): <C5: "
        else:
            pred_string = f"Prediction ({effective_prediction_time}): >=C5: "
        for target, predicted_probability in zip(self.targets, predicted_probabilities[0]):
            pred_string += f'{target}: %.3f ' % (predicted_probability)
        return f"{pred_string}\n"

    def warmup_mode(self):
        """Get differences from last 5 minutes prior to application start"""
        # find record for start date
        expected_time_tag_string = datetime.strftime(self.expected_time_tag, "%Y-%m-%dT%H:%M:00Z")
        for record in self.flux[self.offset:]:  # skip to end for efficiency
            if record['time_tag'] == expected_time_tag_string:
                for _ in range(8):  # need 8 so the first 3 predictions have reliable 5 minute differences87
                    self.update_live_data(self.expected_time_tag)
                    self.expected_time_tag += timedelta(minutes=1)
                break

        # generate retroactive predictions
        for offset in range(2, -1, -1):  # 2, 1, 0 correspond to 5, 4 and 3 minutes before flare start
            x = self.get_values_for_prediction(offset=offset)
            pred_string = self.predict(self.minutes_since_start, x) + "\n"
            if self.print_output:
                print(pred_string)
            self.minutes_since_start += 1

    def standby_mode(self):
        """Predict while waiting for a flare to start"""
        while True:
            self.update_live_data(self.expected_time_tag)
            # retroactive predictions
            if self.minutes_since_start == -2:
                for offset, retroactive_minute in zip(range(self.minutes_since_start + 5, self.minutes_since_start + 2, -1), [-5, -4, -3]):
                    x = self.get_values_for_prediction(offset=offset)
                    pred_string = self.predict(retroactive_minute, x)
                    if self.print_output:
                        print(pred_string)

            # current prediction
            x = self.get_values_for_prediction()
            pred_string = self.predict(self.minutes_since_start, x)
            if self.print_output:
                print(pred_string)
            self.delete_old_data()
            self.expected_time_tag += timedelta(minutes=1)
            # if 3 consecutive increases, it's a flare!
            if self.flare_mode_unlocked:
                if self.input_data["XRSB_flux"][0] > self.input_data["XRSB_flux"][1] > self.input_data["XRSB_flux"][2] > self.input_data["XRSB_flux"][3]:
                    if self.minutes_since_start == 0:
                        print(self.input_data["XRSB_flux"][0], self.input_data["XRSB_flux"][1], self.input_data["XRSB_flux"][2], self.input_data["XRSB_flux"][3])
                        # self.flare_mode()
                        self.flare_start_time = datetime.now(timezone.utc)
                        self.select_mode("Flare")
                        return pred_string
                # if XRSB increased, progress closer to flare start
                if self.input_data["XRSB_flux"][0] > self.input_data["XRSB_flux"][1]:
                    self.minutes_since_start += 1
                # else, keep going at 3 minutes to flare start
                else:
                    self.minutes_since_start = -2
            else:
                self.unlock_flare_mode()
                self.minutes_since_start = -2
            return pred_string

    def flare_mode(self):
        "Predict in a live flare"
        # we increment at start here to account for the 0-minute prediction being made in standby mode,
        # so <15 means the 15 minute prediction will still be made here
        self.expected_time_tag += timedelta(minutes=1)
        self.minutes_since_start += 1
        self.update_live_data(self.expected_time_tag)
        x = self.get_values_for_prediction()
        pred_string = self.predict(self.minutes_since_start, x)
        time_since_flare_start_total_seconds = (self.real_data_timestamps[0] - self.flare_start_time).total_seconds()
        time_since_flare_start_minutes = int(time_since_flare_start_total_seconds / 60)
        time_since_flare_start_seconds = int(time_since_flare_start_total_seconds % 60)
        if self.print_output:
            print(f"Time since flare start: {time_since_flare_start_minutes}:{time_since_flare_start_seconds}\n"
                  f"Prediction at {self.minutes_since_start} minutes\n"
                  f"{pred_string}")
        self.delete_old_data()
        return

    def make_flare_output_folder(self):

        formatted_flare_start_timestamp = datetime.strftime(self.flare_start_time, "%Y-%m-%dT%H%M00")
        out_dir = os.path.join("Artifacts", self.app_start_time, formatted_flare_start_timestamp)
        if not os.path.exists(out_dir):  # should only exist already is in test mode?
            os.mkdir(out_dir)

        return out_dir

    def exit_flare_mode(self):

        # create output folder for flare
        out_dir = self.make_flare_output_folder()

        self.save_seven_day_flux_history_json()

        # Dump log
        with open(os.path.join(out_dir, f"prediction_history.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            header = [""] + self.targets + ["Reported Data Time", "True Data Arrival Time"]
            w.writerow(header)
            number_of_minutes_to_report = self.minutes_since_start + 6  # if cancelled, report to now plus first 5 minutes
            for minutes_since_start, reported_data_time, real_data_time in zip(range(-5, 16), reversed(self.timestamps[:number_of_minutes_to_report]), reversed(self.real_data_timestamps[:number_of_minutes_to_report])):
                row = [minutes_since_start]
                for target in self.targets:
                    row.append(self.prediction_history[minutes_since_start][target])
                row.append(reported_data_time)
                row.append(real_data_time.strftime("%Y-%m-%dT%H:%M:%S"))
                w.writerow(row)

        self.minutes_since_start = -2  # reset
        self.flare_mode_unlocked = False  # wait for gradual phase to end before starting another flare
        self.in_flare_mode = False



if __name__ == '__main__':

    l = LiveApp(use_test_flux=False, print_output=True, use_secondary_source=False)
    l.warmup_mode()
    l.select_mode("Standby")
    while True:
        while l.in_standby_mode:
            l.standby_mode()
        l.select_mode("Flare")
        if l.print_output:
            print("\n\nIN FLARE MODE!\n\n")
        while l.minutes_since_start < 15:
            l.flare_mode()
        l.exit_flare_mode()
        l.select_mode("Standby")