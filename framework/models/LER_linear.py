class LERLinear:

    def __init__(self, slopes, breakpoint = None):
        self._slopes = slopes
        self._breakpoint = breakpoint

    def get_slope(self, x):
        if self._breakpoint == None:
            return self._slopes #slopes contains only 1 slope
        else: 
            #slopes contain two slopes
            if x < self._breakpoint:
                return self._slopes[0]
            else:
                return self._slopes[1]
