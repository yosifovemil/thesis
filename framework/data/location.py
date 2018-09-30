# An enum for the different locations supported by the PhenoReader classes
import framework.util.config_parser as config_parser
import os


class Location():
    ABR15 = "ABR15"
    BSBEC = "BSBEC"
    JKI15 = "JKI15"
    JKI20 = "JKI20"
    _KNOWN_FIELDS = [ABR15, BSBEC, JKI15, JKI20]

    def __init__(self, location_name, year=""):
        self._config = config_parser.SpaceAwareConfigParser()
        self._config.read(os.path.expandvars("$ALF_CONFIG/field_config.ini"))
        self._year = int(year)

        if location_name in self._KNOWN_FIELDS:
            self._section_name = location_name
        else:
            raise Exception("Unknown location: %s" % location_name)

    def year_check(self):
        if self._year == "":
            raise Exception("Error! Location has unspecified year.")

    def get_sunscan_dir(self):
        self.year_check()
        return os.path.expandvars(self._config.get(self._section_name,
                                  "sunscan_dir") % self._year)

    def get_pheno_dataf(self):
        self.year_check()
        return os.path.expandvars(self._config.get(self._section_name,
                                  "pheno_data") % self._year)

    def get_mscan_name(self):
        return self._config.get(self._section_name, "mscan_name")

    def get_year(self):
        self.year_check()
        return self._year

    def get_name(self):
        return self._section_name

    def get_met_data(self):
        self.year_check()
        return os.path.expandvars(self._config.get(self._section_name,
                                  "met_data") % self._year)

    def get_plant_area(self):
        return float(self._config.get(self._section_name, "plant_area"))

    def get_fll_location(self):
        self.year_check()
        fll_data_loc = self._config.get(self._section_name, "fll_data")
        return os.path.expandvars(fll_data_loc % self._year)

    def get_leaf_conversion(self, genotype):
        try:
            # try to find genotype specific leaf conversion
            val = float(self._config.get(self._section_name,
                        "%s_leaf_conversion" % genotype))
            return val

        except config_parser.ConfigParser.NoOptionError:
            # return the generic leaf_conversion value
            val = float(self._config.get(self._section_name,
                                         "leaf_conversion"))
            return val

    def get_transmission_location(self):
        self.year_check()
        return os.path.expandvars(self._config.get(self._section_name,
                                  "transmission_dataf") % self._year)

    def get_harvest_dataf(self):
        self.year_check()
        return os.path.expandvars(self._config.get(self._section_name,
                                  "harvest_data") % self._year)
