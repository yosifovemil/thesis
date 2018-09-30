#module that handles user details: username and password

import getpass
import os 

_username = None
_password = None

def init(interactive = True):
	global _username 
	global _password 
	if _username is None:
		if interactive:
			_username = raw_input("Username:")
			_password = getpass.getpass("Password:")
		else:
			f = open(os.path.join(os.environ['HOME'], '.user'), 'r')
			_username = f.readline().strip()
			_password = f.readline().strip()
	else:
		#username and password already entered
		pass

def get_username():
	global _username
	if _username is not None:
		return _username
	else:
		raise Exception("Username and password not available. "
						"Please run %s.init() first" % __name__)

def get_password():
	global _password
	if _password is not None:
		return _password
	else:
		raise Exception("Username and password not available. "
						"Please run %s.init() first" % __name__)
