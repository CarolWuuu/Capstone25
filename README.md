Workflow
1. run DFPlayer.ino in Arduino to make speakers in gel phantom play sounds 
2. while the speaker is playing, run log_piezo_voltage_multi.py to collect raw piezo signals
3. run filtering on raw piezo signals
4. run triang.py on filtered signals - record estmated FHR and percent error in excel sheet
5. run triang_wDataCollection.py on filtered signals - record automatically estimated location in csv format
6. run FHR Error.py with avg FHR percent avg FHR percent error.csv (saved from excel sheet)
7. run Loc Error.py with All_Final_estimated_Locations.csv (from running triang_wDataCollection.py)

FHR numbering

- FHR 1: 147 bpm
- FHR 2: 172 bpm
- FHR 3: 142 bpm
- FHR 4: 110 bpm

Note:
- Name mp3 files in TF card as 0001.mp3, 0002.mp3, etc. No folder needed.
- Due to the file size or recording quality, the 110 BPM fetal heart rate audio must be stored on the TF card as a standalone file, with no other recordings present.