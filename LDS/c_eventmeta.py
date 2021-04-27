
class EventMeta:
    SUM_CHANNELS = [8, 15, 24, 31, 40, 47, 56, 63]
    SN_ADC = {
        1: 175780172,
        2: 175854781
    }
    
    def __init__(self, evid):
        self.evid = evid # Event number
        self.ntp_dt = None # Reference ntp time in datetime
        self.ntp_us = None # Reference ntp time in us (epoch)
        self.dec_offset_us = None # Declared offset with respect to last PPS
        self.jitter_ns = { # Jitter around declared offset
            1: None,
            2: None
        } 
        self.offset_us = { # Actual offset found after window search
            1: None,
            2: None
        } 
        self.successfullyMerged = None # Whether the event is fully merged (0: Error, 1: Ok, 2: Ok, but either multiple entries or other issues)
        self.triggered_ADCs = [] # Which ADCs have a busy signal
        self.nTriggered_ADCs = None # How many ADCs actually produced data
        self.busyFront_ns = { # ns location of busy front
            1: None,
            2: None
        }
        self.entryPerChannel = { # Each channel's waveform is located in TTree['th1s_ptr'][entryNumber]
            1: {
                self.SUM_CHANNELS[0]: None,
                self.SUM_CHANNELS[1]: None,
                self.SUM_CHANNELS[2]: None,
                self.SUM_CHANNELS[3]: None,
                self.SUM_CHANNELS[4]: None,
                self.SUM_CHANNELS[5]: None,
                self.SUM_CHANNELS[6]: None,
                self.SUM_CHANNELS[7]: None,
            },
            2: {
                self.SUM_CHANNELS[0]: None,
                self.SUM_CHANNELS[1]: None,
                self.SUM_CHANNELS[2]: None,
                self.SUM_CHANNELS[3]: None,
                self.SUM_CHANNELS[4]: None,
                self.SUM_CHANNELS[5]: None,
                self.SUM_CHANNELS[6]: None,
                self.SUM_CHANNELS[7]: None,
            }
        }

        self.hist_b = { # Histogram bins
            1: {
                0: None,
                self.SUM_CHANNELS[0]: None,
                self.SUM_CHANNELS[1]: None,
                self.SUM_CHANNELS[2]: None,
                self.SUM_CHANNELS[3]: None,
                self.SUM_CHANNELS[4]: None,
                self.SUM_CHANNELS[5]: None,
                self.SUM_CHANNELS[6]: None,
                self.SUM_CHANNELS[7]: None,
            },
            2: {
                0: None,
                self.SUM_CHANNELS[0]: None,
                self.SUM_CHANNELS[1]: None,
                self.SUM_CHANNELS[2]: None,
                self.SUM_CHANNELS[3]: None,
                self.SUM_CHANNELS[4]: None,
                self.SUM_CHANNELS[5]: None,
                self.SUM_CHANNELS[6]: None,
                self.SUM_CHANNELS[7]: None,
            }
        }

        self.hist_h = { # Histogram contents
            1: {
                0: None,
                self.SUM_CHANNELS[0]: None,
                self.SUM_CHANNELS[1]: None,
                self.SUM_CHANNELS[2]: None,
                self.SUM_CHANNELS[3]: None,
                self.SUM_CHANNELS[4]: None,
                self.SUM_CHANNELS[5]: None,
                self.SUM_CHANNELS[6]: None,
                self.SUM_CHANNELS[7]: None,
            },
            2: {
                0: None,
                self.SUM_CHANNELS[0]: None,
                self.SUM_CHANNELS[1]: None,
                self.SUM_CHANNELS[2]: None,
                self.SUM_CHANNELS[3]: None,
                self.SUM_CHANNELS[4]: None,
                self.SUM_CHANNELS[5]: None,
                self.SUM_CHANNELS[6]: None,
                self.SUM_CHANNELS[7]: None,
            }
        }

        self.t_rel = { # Time relative to busy
            1: {
                self.SUM_CHANNELS[0]: None,
                self.SUM_CHANNELS[1]: None,
                self.SUM_CHANNELS[2]: None,
                self.SUM_CHANNELS[3]: None,
                self.SUM_CHANNELS[4]: None,
                self.SUM_CHANNELS[5]: None,
                self.SUM_CHANNELS[6]: None,
                self.SUM_CHANNELS[7]: None,
            },
            2: {
                self.SUM_CHANNELS[0]: None,
                self.SUM_CHANNELS[1]: None,
                self.SUM_CHANNELS[2]: None,
                self.SUM_CHANNELS[3]: None,
                self.SUM_CHANNELS[4]: None,
                self.SUM_CHANNELS[5]: None,
                self.SUM_CHANNELS[6]: None,
                self.SUM_CHANNELS[7]: None,
            }
        }
        