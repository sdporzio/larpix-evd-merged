import os, uproot, ROOT
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
import LDS.c_eventmeta as cem

# Locate the file archive and load it
pathFromHere = Path(__file__).parent.absolute()
fileArchivePath = str(pathFromHere)+'/fileArchive.csv'
fArc = pd.read_csv(fileArchivePath)

### GET ALL WAVEFORM INFORMATION FROM WAVEFORM FILES
def GetEventMetadata(df,lpath,event_n,datime,offset_us,window_us=1500):
    # Open ROOT file
    rfile = ROOT.TFile.Open(lpath, 'read')
    rwf = rfile.Get('rwf')
    # Create event metadata object
    eventmeta = cem.EventMeta(event_n)
    # Assign datetime object and convert to epoch linux time
    eventmeta.ntp_dt = datime
    eventmeta.ntp_us = (dt.datetime.strptime(datime,"%Y-%m-%d %H:%M:%S") - dt.datetime(1970,1,1)).total_seconds()
    eventmeta.dec_offset_us = offset_us
    # Save the serial numbers for the two ADCs
    sn1,sn2 = eventmeta.SN_ADC[1], eventmeta.SN_ADC[2]

    # Looking for busy signal in declared timing range
    print(f"Searching Busy at {eventmeta.ntp_dt}, with {offset_us} us offset, in a {window_us} us-wide window.\n")
    # Find all data with a second (1000 ms) of our datetime timestamp
    query_df = df.query(f'abs(utime_ms-{eventmeta.ntp_us*1000.:.0f})<1000')
    # Find all data within a 'window_us'-wide window of our timing offset from last PPS (tai_ns)
    query_df = query_df.query(f'abs(tai_ns/1e3-{eventmeta.dec_offset_us:.0f})<{window_us}')
    # Find all the SUM signals plus the Busy signal (ch=00) contained in our search window
    query_df = query_df.query(f'ch==8 | ch==15 | ch==24 | ch==31 | ch==40 | ch==47 | ch==56 | ch==63 | ch==0')
    # Find the busy signal for each ADC
    adc1_df = query_df.query(f"sn=={sn1}")
    adc2_df = query_df.query(f'sn=={sn2}')

    # Check how many recording channels each ADC has
    if len(adc1_df)>0:
        print(f"- Found following channels for ADC 1 (Mult. {len(adc1_df)})")
        channel_list = [int(r['ch']) for i,r in adc1_df.iterrows()]
        print('|_', channel_list)
    else:
        print("- Found no channels with entries for ADC1 (Mult. 0)")
        print('|_ []')

    if len(adc2_df)>0:
        print(f"- Found following channels for ADC 2 (Mult. {len(adc2_df)})")
        channel_list = [int(r['ch']) for i,r in adc2_df.iterrows()]
        print('|_', channel_list)
    else:
        print("- Found no channels with entries for ADC2 (Mult. 0)")
        print('|_ []')

    # Take the timing 
    t0_adc1 = adc1_df.query(f"ch==0")['tai_ns'].values
    t0_adc2 = adc2_df.query(f"ch==0")['tai_ns'].values

    # Check how many busy signals we have found in the timing window
    # If we don't have enough it's time to give up and close the function
    if (len(t0_adc1)>1 or len(t0_adc2)>1):
        print("ATTENTION: Multiple t0 found in same window. Will choose the earliest.")
        eventmeta.successfullyMerged = 2
        return eventmeta
    elif (len(t0_adc1)==0 and len(t0_adc2)==0):
        print("ERROR: No t0 found in timing window. Maybe try with a larger one.")
        eventmeta.successfullyMerged = 0
        return eventmeta
    else:
        eventmeta.successfullyMerged = 1

    # At least a busy has been found. Create a register of ADCs that do have a busy
    # Then assign the timing to the variables.
    # Assign ADC1 value
    if (len(t0_adc1)>0):
        eventmeta.offset_us[1] = t0_adc1[0]/1000.
        eventmeta.triggered_ADCs.append(1)
    else:
        print("No t0 for ADC1 found. Will assign to it t0 from ADC2.")
        eventmeta.offset_us[1] = t0_adc2[0]/1000.
    # Assign ADC2 value 
    if (len(t0_adc2)>0):
        eventmeta.offset_us[2] = t0_adc2[0]/1000.
        eventmeta.triggered_ADCs.append(2)
    else:
        print("No t0 for ADC2 found. Will assign to it t0 from ADC1.")
        eventmeta.offset_us[2] = t0_adc1[0]/1000.
    # Calculate how off the actual tai are
    eventmeta.jitter_ns[1] = eventmeta.offset_us[1] - eventmeta.dec_offset_us
    eventmeta.jitter_ns[2] = eventmeta.offset_us[2] - eventmeta.dec_offset_us
    # Update nTriggered_ADCs
    eventmeta.nTriggered_ADCs = len(eventmeta.triggered_ADCs)

    print(f'\nOffset: {eventmeta.dec_offset_us} us')
    print(f't0_ADCs: [{eventmeta.offset_us[1]} | {eventmeta.offset_us[2]}] us')
    print(f'Jitter w.: [{eventmeta.jitter_ns[1]} | {eventmeta.jitter_ns[2]}] ns\n')

    # Let's now find the front of the busy signal
    for trigadc in eventmeta.triggered_ADCs:

        adc_sn = eventmeta.SN_ADC[trigadc]
        # Get the entry number of the root tree corresponding to the channel/adc pair we're looking for
        entry_number = query_df.query(f'ch==0 & sn=={adc_sn}').iloc[0].name
        # Let's grab the corresponding waveform histogram, by switching to the proper entry
        eventmeta.entryPerChannel[trigadc][0] = entry_number
        rwf.GetEntry(entry_number)
        wf_hist = getattr(rwf,'th1s_ptr')
        # Light ADC clock at 100 MHz, bin corresponding to 10 ns, so mult. by 10 to get ns
        eventmeta.hist_b[trigadc][0] = [wf_hist.GetBinLowEdge(i) for i in range(1,wf_hist.GetNbinsX()+1)]
        eventmeta.hist_h[trigadc][0] = [wf_hist.GetBinContent(i) for i in range(1,wf_hist.GetNbinsX()+1)]
        for i,h in enumerate(eventmeta.hist_h[trigadc][0]):
            if h<-1000:
                eventmeta.busyFront_ns[trigadc] = i*10
                print(f'Found busy front on ADC {trigadc} ({adc_sn}), at {eventmeta.busyFront_ns[trigadc]} ns.')
                break 
        # Let's get a timing relative to the busy
        trel = query_df.query(f'ch==0 & sn=={adc_sn}')['tai_ns'].values[0]
        trel = int(trel - eventmeta.offset_us[trigadc]*1000.) # Let's have the time relative to ADC t0
        trel = trel - eventmeta.busyFront_ns[trigadc] # Let's subtract the busy front
        trel = trel + 325 # Busy delay
        eventmeta.t_rel[trigadc][0] = trel
        nbins = len(eventmeta.hist_b[trigadc][0])
        eventmeta.hist_b[trigadc][0] = np.linspace(trel,trel+(nbins*10),nbins)

        # And let's do the same for all the other channels
        for ch in eventmeta.SUM_CHANNELS:
            # Get the entry number of the root tree corresponding to the channel/adc pair we're looking for
            entry_number = query_df.query(f'ch=={ch} & sn=={adc_sn}').iloc[0].name
            # Let's grab the corresponding waveform histogram, by switching to the proper entry
            eventmeta.entryPerChannel[trigadc][ch] = entry_number
            rwf.GetEntry(eventmeta.entryPerChannel[trigadc][ch])
            wf_hist = getattr(rwf,'th1s_ptr')
            # Let's get a timing relative to the busy
            trel = query_df.query(f'ch=={ch} & sn=={adc_sn}')['tai_ns']
            trel = int(trel - eventmeta.offset_us[trigadc]*1000.) # Let's have the time relative to ADC t0
            trel = trel - eventmeta.busyFront_ns[trigadc] # Let's subtract the busy front
            trel = trel + 325 # Busy delay
            eventmeta.t_rel[trigadc][ch] = trel
            # Light ADC clock at 100 MHz, bin corresponding to 10 ns, so mult. by 10 to get ns
            eventmeta.hist_b[trigadc][ch] = [wf_hist.GetBinLowEdge(i) for i in range(1,wf_hist.GetNbinsX()+1)]
            eventmeta.hist_h[trigadc][ch] = [wf_hist.GetBinContent(i) for i in range(1,wf_hist.GetNbinsX()+1)]
            nbins = len(eventmeta.hist_b[trigadc][ch])
            eventmeta.hist_b[trigadc][ch] = np.linspace(trel,trel+(nbins*10),nbins)


    return eventmeta



def FindPartnerLightFile(cfile,ldirectory):
    # Extract the name substring from the charge file path
    split = np.array(cfile.split('/')[-1].split('_'))
    cfileCore = '_'.join(split[ np.where(split=='2021')[0][0] : np.where(split=='CEST')[0][0] ])
    # Query the database for the corresponding light file
    lfileCore = fArc[fArc['chargeFile'].str.contains(cfileCore)]['lightFile'].values
    # Check if we at least one match
    if len(lfileCore)==0:
        raise Exception(f'ERROR. No matching light file for {cfileCore}. Exiting.')
    # If not let's notify that's the case
    print(f'Charge file {cfileCore} has been matched to light file {lfileCore}.')
    # Form the light file name
    lfile = 'rwf_'+lfileCore[0]+'.root'
    lpath = ldirectory+'/'+lfile
    # Now make sure the file actually exists
    if not os.path.isfile(lpath):
        raise Exception(f'ERROR. Matching file {lpath} does not exist. Exiting.')
    # Load the necessary files
    df = (uproot.open(lpath)['rwf']).arrays(['event','sn','ch','utime_ms','tai_ns'],library='pd')
    return df, lpath

