The following Python scripts are presented alongside the OutFin dataset:
** All scripts are written in Python 3.6.4
===============================================================================
NAME: Reliability.py
PURPOSE: Demonstrates that the smartphones/apps provide consistent measurements
         at different points in time.	
REQUIRED IMPORTS: - os
                  - pandas
                  - scipy
                  - random	
PATH OF REQUIRED DATA: OutFin/Code/temporal_data
===============================================================================
NAME: Validity1.py
PURPOSE: Demonstrates that the smartphones/apps accurately measure what they
         are supposed to measure.	
REQUIRED IMPORTS: - os
                  - pandas
                  - scipy
                  - random	
PATH OF REQUIRED DATA: OutFin/Code/temporal_data
===============================================================================
NAME: Validity2.py
PURPOSE: Visualizes the data collected by the smartphones over randomly 
         selected RPs.
REQUIRED IMPORTS: - os
                  - pandas
                  - sklearn
                  - matplotlib
                  - numpy
                  - random	
PATH OF REQUIRED DATA: OutFin/Measurements
===============================================================================
NAME: Descriptive_Statistics.py
PURPOSE: Provides descriptive statistics of the most important variables of the 
         dataset. 
REQUIRED IMPORTS: - os
                  - pandas
                  - statistics
                  - numpy	
PATH OF REQUIRED DATA: OutFin/Measurements
===============================================================================
NAME: Calibration.py
PURPOSE: Provides Hard/Soft-iron calibration values for a given phone and day. 
REQUIRED IMPORTS: - os
                  - pandas	
PATH OF REQUIRED DATA: OutFin/Calibration
===============================================================================
NAME: Fingerprint_Interpolation.py
PURPOSE: Provides a demonstration of how OutFin can be used for fingerprint 
         interpolation. 
REQUIRED IMPORTS: - os
                  - pandas
                  - numpy
                  - scipy
                  - matplotlib	
PATH OF REQUIRED DATA: OutFin/Measurements and OutFin/Coordinates
===============================================================================
NAME: Feature_Extraction.py
PURPOSE: Provides a demonstration of how OutFin can be used for feature 
         extraction. 
REQUIRED IMPORTS: - os
                  - pandas
                  - numpy
                  - keras
                  - matplotlib
                  - sklearn
                  - random	
PATH OF REQUIRED DATA: OutFin/Measurements
===============================================================================
NAME: Performance_Evaluation.py
PURPOSE: Provides a demonstration of how OutFin can be used for performance 
         evaluation. 
REQUIRED IMPORTS: - os
                  - pandas
                  - numpy
                  - math
                  - sklearn
PATH OF REQUIRED DATA: OutFin/Measurements and OutFin/Coordinates
===============================================================================
NAME: Signal_Denoising.py
PURPOSE: Provides a demonstration of how OutFin can be used for signal 
         denoising. 
REQUIRED IMPORTS: - os
                  - pandas
                  - numpy
                  - math
                  - sklearn
                  - matplotlib
                  - statistics
                  - keras
PATH OF REQUIRED DATA: OutFin/Measurements and OutFin/Coordinates
===============================================================================