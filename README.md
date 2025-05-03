Workflow
1. run log_piezo_voltage_multi.py to collect raw piezo signals
2. run filtering on raw piezo signals
3. run triang.py on filtered signals - record estmated FHR and percent error in excel sheet
4. run triang_wDataCollection.py on filtered signals - record automatically estimated location in csv format
5. run FHR Error.py with avg FHR percent avg FHR percent error.csv (saved from excel sheet)
6. run Loc Error.py with All_Final_estimated_Locations.csv (from running triang_wDataCollection.py)

FHR numbering

- FHR 1: 147 bpm
- FHR 2: 172 bpm
- FHR 3: 142 bpm
- FHR 4: 110 bpm
