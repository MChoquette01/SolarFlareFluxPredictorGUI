
def update_test_data(self):
    """Read or wait for the next data sample"""
    checker = 0.0

    while checker != 1.0:
        datum = self.flux[self.test_line_idx]
        if datum['energy'] == "0.05-0.4nm":  # XRSA
            self.input_data["XRSA_flux"].insert(0, datum['observed_flux'])
            for n in range(1, 6):
                if datum['observed_flux'] is not None and self.input_data['XRSA_flux'][n] is not None:
                    self.input_data[f"XRSA_flux_{n}_minute_difference"].insert(0, datum['observed_flux'] -
                                                                               self.input_data["XRSA_flux"][n])
                else:
                    self.input_data[f"XRSA_flux_{n}_minute_difference"].insert(0, None)
                    self.last_good_param[f"XRSA_flux_{n}_minute_difference"] += 1
            print(f"Read XRSA at {datum['time_tag']}")
            checker += 0.5
        elif datum['energy'] == "0.1-0.8nm":  # XRSB
            self.input_data["XRSB_flux"].insert(0, datum['observed_flux'])
            for n in range(1, 6):
                if datum['observed_flux'] is not None and self.input_data['XRSB_flux'][n] is not None:
                    self.input_data[f"XRSB_flux_{n}_minute_difference"].insert(0, datum['observed_flux'] -
                                                                               self.input_data["XRSB_flux"][n])
                else:
                    self.input_data[f"XRSB_flux_{n}_minute_difference"].insert(0, None)
                    self.last_good_param[f"XRSB_flux_{n}_minute_difference"] += 1
            print(f"Read XRSB at {datum['time_tag']}")
            checker += 0.5

        self.test_line_idx += 1

    if checker == 1.0:
        # Temperature/Emission Measure
        em, temp = emt.compute_goes_emission_measure(xrsa_data=self.input_data[f"XRSA_flux"][0],
                                                     xrsb_data=self.input_data[f"XRSB_flux"][0],
                                                     goes_sat=datum['satellite'])
        if not math.isnan(temp[0]) and temp[0] is not None:
            self.input_data["Temperature"].insert(0, temp[0])
        else:
            self.input_data["Temperature"].insert(0, None)
            self.last_good_param["Temperature"] += 1

        if not math.isnan(em[0]) and em[0] is not None:
            self.input_data[f"EmissionMeasure"].insert(0, em[0] / 10 ** 30)
        else:
            self.input_data[f"EmissionMeasure"].insert(0, None)
            self.last_good_param["EmissionMeasure"] += 1

        for n in range(1, 6):
            em_n_minute_diff, temp_n_minute_diff = emt.compute_goes_emission_measure(
                xrsa_data=self.input_data[f"XRSA_flux_{n}_minute_difference"][0],
                xrsb_data=self.input_data[f"XRSB_flux_{n}_minute_difference"][0],
                goes_sat=16)

            if not math.isnan(em_n_minute_diff[0]) and em_n_minute_diff[0] is not None:
                self.input_data[f"EmissionMeasure_{n}_minute_from_xrs_difference"].insert(0,
                                                                                          em_n_minute_diff
                                                                                              [0] / 10 ** 30)
            else:
                self.input_data[f"EmissionMeasure_{n}_minute_from_xrs_difference"].insert(0, None)
                self.last_good_param[f"EmissionMeasure_{n}_minute_from_xrs_difference"] += 1

            if not math.isnan(temp_n_minute_diff[0]) and temp_n_minute_diff[0] is not None:
                self.input_data[f"Temperature_{n}_minute_from_xrs_difference"].insert(0,
                                                                                      temp_n_minute_diff[0])
            else:
                self.input_data[f"Temperature_{n}_minute_from_xrs_difference"].insert(0, None)
                self.last_good_param[f"Temperature_{n}_minute_from_xrs_difference"] += 1

        # Non-XRS Differences
        for n in range(1, 6):
            if self.input_data['Temperature'][0] is not None and self.input_data['Temperature'][n] is not None:
                self.input_data[f"Temperature_{n}_minute_difference"].insert(0, self.input_data['Temperature'][0] -
                                                                             self.input_data['Temperature'][n])
            else:
                self.input_data[f"Temperature_{n}_minute_difference"].insert(0, None)
                self.last_good_param[f"Temperature_{n}_minute_difference"] += 1

            if self.input_data['EmissionMeasure'][0] is not None and self.input_data['EmissionMeasure'][n] is not None:
                self.input_data[f"EmissionMeasure_{n}_minute_difference"].insert(0, self.input_data['EmissionMeasure'][0] - self.input_data['EmissionMeasure'][n])
            else:
                self.input_data[f"EmissionMeasure_{n}_minute_difference"].insert(0, None)
                self.last_good_param[f"EmissionMeasure_{n}_minute_difference"] += 1

    checker = 0.0  # reset

    # check if linear interpolation is needed
    for param in self.params:
        if self.input_data[param][0] == None:
            for idx, value in enumerate(self.input_data[param]):
                if value is None:
                    self.input_data[param][idx] = np.interp(idx,  # index to interpolate for
                                                            [x for x in range(len(self.input_data[param])) if
                                                             self.input_data[param][x] != None],
                                                            # indicies associated with known values
                                                            [x for x in self.input_data[param] if x != None])  # known values



def warmup_mode(self):
    """Get differences from last 5 minutes prior to application start"""
    # find record for start date
    for record in self.flux:
        if record['time_tag'] == self.start_time:
            for _ in range(5):
                self.update_test_data()
            self.in_startup_mode = False
            break

    while self.minutes_since_start < -2:
        self.update_test_data()
        x = self.get_current_values_for_prediction(self.input_data)
        pred_string = self.predict(x)
        print(pred_string)
        self.minutes_since_start += 1

    self.standby_mode()


def standby_mode(self):
    """Predict while waiting for a flare to start"""
    self.in_standby_mode = True

    while True:
        self.update_test_data()
        x = self.get_current_values_for_prediction(self.input_data)
        pred_string = self.predict(x)
        print(pred_string)
        self.delete_old_data()
        # if 3 consecutive increases, it's a flare!
        if self.input_data["XRSB_flux"][0] > self.input_data["XRSB_flux"][1] > self.input_data["XRSB_flux"][2] > \
                self.input_data["XRSB_flux"][3]:
            if self.minutes_since_start == 0:
                self.in_standby_mode = False
                self.minutes_since_start += 1
                self.flare_mode()
        # if XRSB increased, progress clsoer to flare start
        if self.input_data["XRSB_flux"][0] > self.input_data["XRSB_flux"][1]:
            self.minutes_since_start += 1
        # else, keep going at 3 minutes to flare start
        else:
            self.minutes_since_start = -2


def flare_mode(self):
    "Predict in a live flare"
    self.in_flare_mode = True
    while self.minutes_since_start < 16:
        self.update_test_data()
        x = self.get_current_values_for_prediction(self.input_data)
        pred_string = self.predict(x)
        print(pred_string)
        self.delete_old_data()
        self.minutes_since_start += 1

    in_flare_mode = False
    self.minutes_since_start = -2
    self.standby_mode()