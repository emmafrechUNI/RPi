from Utils import debug
from dateutil import tz
import tkinter as tk
import numpy as np

class Plot():
    prev_min_pt = None
    prev_max_pt = None

    def __init__(self):
        """Initializes plots
                param self: reference to parent object
                """
        self.min_pt = 10000
        self.max_pt = -10000

       