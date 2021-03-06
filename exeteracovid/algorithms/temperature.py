# Copyright 2020 KCL-BMEIS - King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import numpy as np

from exetera.core import validation as val


class ValidateTemperature1:
    def __init__(self, min_temp_incl, max_temp_incl, f_missing_temp, f_bad_temp):
        """
        Check the temperature field and convert fahrenheit to celsius.

        :param min_temp_incl: The minimal numeric value for temperature.
        :param max_temp_incl: The maximum numeric value for temperature.
        :param f_missing_temp: An array marking missing temperature fields.
        :param f_bad_temp: An array marking invalid temperature fields.
        """
        self.min_temp_incl = min_temp_incl
        self.max_temp_incl = max_temp_incl
        self.f_missing_temp = f_missing_temp
        self.f_bad_temp = f_bad_temp

    def __call__(self, temps, filter_list):
        """
        Perform the temperature check.

        :param temps: The temperature column from assessments dataframe.
        :param filter_list: The filter marking unusable temperature fields, e.g. missing value or invalid.
        """
        temperature_c = np.zeros_like(temps, dtype=np.float)
        for ir, t in enumerate(temps):
            if t == '':
                dest_temp = 0.0
                filter_list[ir] |= self.f_missing_temp
            else:
                t = float(t)
                dest_temp = (t - 32) / 1.8 if t > self.max_temp_incl else t
                if dest_temp == 0.0:
                    filter_list[ir] |= self.f_missing_temp
                    temperature_c[ir] = dest_temp
                elif dest_temp <= self.min_temp_incl or dest_temp >= self.max_temp_incl:
                    temperature_c[ir] = 0.0
                    filter_list[ir] |= self.f_bad_temp
                else:
                    temperature_c[ir] = dest_temp

        return temperature_c


def validate_temperature_1(min_temp, max_temp,
                           temps, temp_units, temp_set,
                           dest_temps, dest_temps_valid, dest_temps_modified):
    """
    Deprecated, please use temperature.validate_temperature_v1()
    """
    warnings.warn("deprecated", DeprecationWarning)
    raw_temps = temps[:]
    raw_dest_temps = np.where(raw_temps > max_temp, (raw_temps - 32) / 1.8, raw_temps)
    raw_dest_temps_valid = temp_set[:] & (min_temp <= raw_dest_temps) & (raw_dest_temps <= max_temp)
    raw_dest_temps_modified = raw_temps != raw_dest_temps
    dest_temps.write(raw_dest_temps)
    dest_temps_valid.write(raw_dest_temps_valid)
    dest_temps_modified.write(raw_dest_temps_modified)


def validate_temperature_v1(session, min_temp, max_temp,
                           temps, temp_units, temp_set,
                           dest_temps, dest_temps_valid, dest_temps_modified):
    """
    Check the temperature field and convert fahrenheit to celsius.

    :param session: The Exetera session instance.
    :param min_temp: The minimal numeric value for temperature.
    :param max_temp: The maximum numeric value for temperature.
    :param temps: The 'temperature' column from assessments dataframe.
    :param temp_units: The 'temperature_unit' column from assessments dataframe.
    :param temp_set: A field marking if the temperature field is set.
    :param dest_temps: A destination field to write the temperature values to.
    :param dest_temps_valid: A destination field to indicates if the temperature is valid, e.g. in between minimum and
        maximum numeric values.
    :param dest_temps_modified: A destination field to indicates if the temperature has been modified here.
    """
    raw_temps = val.raw_array_from_parameter(session, "temps", temps)
    raw_temp_set = val.raw_array_from_parameter(session, "temp_set", temp_set)
    raw_dest_temps = np.where(raw_temps > max_temp, (raw_temps - 32) / 1.8, raw_temps)
    raw_dest_temps_valid = raw_temp_set & (min_temp <= raw_dest_temps) & (raw_dest_temps <= max_temp)
    raw_dest_temps_modified = raw_temps != raw_dest_temps
    dest_temps.data.write(raw_dest_temps)
    dest_temps_valid.data.write(raw_dest_temps_valid)
    dest_temps_modified.data.write(raw_dest_temps_modified)