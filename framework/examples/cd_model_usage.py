#!/usr/bin/env python

#example on how to parameterise and use the CD model. 
#We will train the model on BSBEC 2011 and 2012 data, 
#the way it has been done in the model

from framework.models.cd_model import ChrisDaveyModel
from framework.data.location import Location

locations = [Location(Location.BSBEC, 2011),
            Location(Location.BSBEC, 2012)]
