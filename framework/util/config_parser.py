import ConfigParser

class SpaceAwareConfigParser(ConfigParser.ConfigParser):

	_quotes = ['\"']

	def read(self, filename):
		self.optionxform = str
		ConfigParser.ConfigParser.read(self, filename)

		#remove the included quotes
		sections = self._sections
		for section_name in sections:
			for var_name in sections[section_name]:
				for quote in self._quotes:
					temp = sections[section_name][var_name]
					sections[section_name][var_name] = temp.replace(quote, "")

