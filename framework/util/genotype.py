from framework.data.location import Location

def assign_genotype(uid, location):
	location_name = location.get_name()
	if location_name == Location.BSBEC:
		return assign_genotype_BSBEC(uid)
	else:
		raise Exception("Unhandled location %s" % location_name)

def assign_genotype_BSBEC(uid):
	uid = int(uid)
	if uid in [35,48,50,61]:
		return "EMI-11"
	elif uid in [36,42,56,62]:
		return "Giganteus"
	elif uid in [39,46,57,53]:
		return "Goliath"
	elif uid in [40,51,58,44]:
		return "Sac-5"
	else:
		raise Exception("Unknown uid %d" % uid)

