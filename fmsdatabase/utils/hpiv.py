from __future__ import annotations
from typing import TYPE_CHECKING, Any
# Standard library imports
import os
import re
import sys
import time
import traceback

# Third-party imports
import fitz
import openpyxl
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Adjust sys.path for relative imports
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local application imports
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from ..fms_data_structure import FMSDataStructure

from ..db import HPIVCharacteristics, HPIVCertification, HPIVRevisions
from .general_utils import LimitStatus, HPIVParameters, HPIVParts
from .ocr_reader import OCRReader
from .hpiv_query import HPIVQuery


class HPIVDataListener(FileSystemEventHandler):
    """
    File system event handler for monitoring HPIV data packages.
    
    This class extends FileSystemEventHandler to monitor a specified directory
    for new PDF files containing HPIV test data. When a new PDF is detected,
    it automatically processes the file to extract test results.
    
    Attributes
    ----------
        path (str): Directory path to monitor for new PDF files
        observer (Observer): Watchdog observer instance for file monitoring
        _processing (bool): Flag to prevent concurrent processing of multiple files
    """
    
    def __init__(self, path="HPIV_data_packages"):
        self.path = path
        self.observer = Observer()
        self.observer.schedule(self, path, recursive=False)
        self.observer.start()
        self.processed = False

    def on_created(self, event):
        """
        Handle file creation events in the monitored directory.
        
        This method is called when a new file is created in the monitored directory.
        It specifically looks for PDF files and processes them to extract HPIV test data.
        
        Args:
            event: File system event object containing information about the created file
        """
        if event.is_directory:
            return
        if event.src_path.endswith('.pdf'):
            pdf_file = event.src_path
            print(f"New PDF detected: {event.src_path}")
            # Ensure only one PDF is processed at a time
            if hasattr(self, '_processing') and self._processing:
                print("A PDF is already being processed. Skipping this file.")
                return
            self._processing = True
            try:
                print(self.path)
                if self.path.endswith("HPIV_data_packages"):
                    self.tr = HPIVData(pdf_file=pdf_file)
                    self.tr.extract_hpiv_data()
                elif self.path.endswith("certifications"):
                    if 'space solutions' in pdf_file.lower():
                        self.tr = HPIVData(pdf_file=pdf_file)                    
                        self.tr.get_ocr_certification()
                self.processed = True
            finally:
                # Ensure processing flag is reset after processing
                self._processing = False

class HPIVData:
    """
    Class for extracting and processing HPIV test results from PDF documents.
    
    This class handles the extraction of HPIV test data from PDF files, processes
    the data according to defined parameters, and manages the results for further
    analysis or reporting.
    
    Attributes
    ----------
        test_results (list): 
            List of extracted test results for multiple HPIVs.
        pdf_file (str): 
            Path to the EIDP PDF file containing HPIV test data.
        parameter_names (list): 
            List of parameter names from HPIVParameters enum.
        excel_file (str): 
            Path to Excel template file for limits.
        hpiv_ids (list):
            List of HPIV IDs extracted from the PDF.
        hpivs (set):
            Set of unique HPIV IDs processed.
        total_lines (str):
            Raw text content extracted from the PDF.
        certification (str):
            Certification batch associated with the HPIVs.
        revision_data (dict):
            Dictionary to store revision data for HPIVs.
        part_number_map (dict):
            Mapping of HPIV parts to their corresponding part numbers.

    Methods
    -------
    get_hpiv_parameters():
        Initialize the HPIV parameters dictionary with standard test limits.
    extract_hpiv_data():
        Extract HPIV test data from the current PDF document.
    check_within_limits():
        Check if for the parameters in the test results attribute, 
        the corresponding value is within defined limits.
    get_ocr_certification():
        Use the OCR reader to extract certification information from the PDF (might become obsolete). ***REMARK: USES OCRReader CLASS, NOT TEXTRACT!***
    get_certification():
        Use Textract to extract certification information from the PDF (might become obsolete).
    get_hpiv_limits_from_excel():
        Retrieve HPIV test limits from the specified Excel template file.
    """ 
    
    def __init__(self, pdf_file=None):
        self.test_results = []
        self.pdf_file = pdf_file
        self.parameter_names = [param.value for param in HPIVParameters]
        self.hpiv_ids = []
        self.hpivs = set()
        self.total_lines = ''
        self.certification = None
        self.revision_data = {}
        self.part_number_map = {
            HPIVParts.HPIV: "VS197-00-00",
            HPIVParts.SEAT_BODY: "VS197-00-01",
            HPIVParts.COIL_ASSY: "VS197-30-00",
            HPIVParts.SPOOL_ASSY: "VS197-10-00",
            HPIVParts.SPOOL_ASSY_R: "VS197-10-04",
            HPIVParts.LOWER_SPOOL: "VS197-10-01",
            HPIVParts.NON_MAGNETIC_TUBE: "VS197-10-02",
            HPIVParts.UPPER_SPOOL: "VS197-10-03",
            HPIVParts.HOUSING: "VS197-40-02",
            HPIVParts.PLUNGER_ASSY: "VS197-40-00",
            HPIVParts.PLUNGER_R: "VS197-40-01",
            HPIVParts.SEAL: "VS197-40-02",
            HPIVParts.DISK_SPRING: "VS197-40-03",
            HPIVParts.COPPER_WIRE: None,
            HPIVParts.KAPTON_TAPE: None,
            HPIVParts.LEAD_WIRE: None,
            HPIVParts.SHRINK_TUBE: None,
            HPIVParts.SOLDER_FILLER: None,
            HPIVParts.SPRING: "VS197-00-03",
            HPIVParts.SHIM1: "VS197-00-04",
            HPIVParts.SHIM2: "VS197-00-05",
            HPIVParts.FILTER_ASSY: "VS197-20-00",
            HPIVParts.FRAME: "VS197-20-01",
            HPIVParts.SUPPORTER: "VS197-20-02",
            HPIVParts.MESH: "VS197-20-03",
        }

    def get_hpiv_parameters(self) -> None:
        """
        Initialize the HPIV parameters dictionary with standard test limits.
        
        Creates a comprehensive dictionary containing all HPIV test parameters
        with their standard min/max values and units. This method sets up the
        structure for storing test results and limits based on HPIV acceptance
        test specifications.
        
        Returns:
            None: Updates self.hpiv_parameters in place
        """
        self.hpiv_parameters = {
            'serial_nr': '',
            'weight': {'min': 0, 'max': 200, 'unit': 'g'},
            'proof_closed': {'min': 320, 'max': 1000, 'unit': 'bar'},
            'proof_open': {'min': 465, 'max': 1000, 'unit': 'bar'},
            'leak_4_hp': {'min': 0, 'max': 1e-06, 'unit': 'scc/s'},
            'leak_4_lp': {'min': 0, 'max': 1e-06, 'unit': 'scc/s'},
            'leak_6_hp': {'min': 0, 'max': 1e-06, 'unit': 'scc/s'},
            'leak_6_lp': {'min': 0, 'max': 1e-06, 'unit': 'scc/s'},
            'leak_15_hp': {'min': 0, 'max': 1e-06, 'unit': 'scc/s'},
            'leak_15_lp': {'min': 0, 'max': 1e-06, 'unit': 'scc/s'},
            'leak_4_hp_press': {'min': 310, 'max': 1000, 'unit': 'bar'},
            'leak_4_lp_press': {'min': 0, 'max': 10, 'unit': 'bar'},
            'leak_6_hp_press': {'min': 310, 'max': 1000, 'unit': 'bar'},
            'leak_6_lp_press': {'min': 0, 'max': 10, 'unit': 'bar'},
            'leak_15_hp_press': {'min': 310, 'max': 1000, 'unit': 'bar'},
            'leak_15_lp_press': {'min': 0, 'max': 10, 'unit': 'bar'},
            'dielectric_str': {'min': 0, 'max': 2, 'unit': 'mA'},
            'insulation_res': {'min': 100, 'max': 1e+20, 'unit': 'ohm'},
            'power_temp': {'min': 18, 'max': 22, 'unit': '°C'},
            'power_res': {'min': 42.8, 'max': 43.8, 'unit': 'ohm'},
            'power_power': {'min': 0, 'max': 10, 'unit': 'W'},
            'ext_leak': {'min': 0, 'max': 1e-10, 'unit': 'scc/s'},
            'pullin_pres': {'min': 310, 'max': 1000, 'unit': 'bar'},
            'pullin_volt': {'min': 0, 'max': 18, 'unit': 'Vdc'},
            'dropout_volt': {'min': 2, 'max': 1000, 'unit': 'Vdc'},
            'resp_pres': {'min': 310, 'max': 1000, 'unit': 'bar'},
            'respo_volt': {'min': 17.98, 'max': 18.02, 'unit': 'Vdc'},
            'respo_time': {'min': 0, 'max': 20, 'unit': 'ms'},
            'respc_volt': {'min': 31.8, 'max': 32.2, 'unit': 'Vdc'},
            'respc_time': {'min': 0, 'max': 20, 'unit': 'ms'},
            'flowrate': {'min': 10, 'max': 1000, 'unit': 'cc/s'},
            'pressd': {'min': 0, 'max': 1, 'unit': 'bar'},
            'cleanliness_6_10': {'min': 0, 'max': 140, 'unit': '-'},
            'cleanliness_11_25': {'min': 0, 'max': 20, 'unit': '-'},
            'cleanliness_26_50': {'min': 0, 'max': 5, 'unit': '-'},
            'cleanliness_51_100': {'min': 0, 'max': 1, 'unit': '-'},
            'cleanliness_100': {'min': 0, 'max': 0, 'unit': '-'},
            'before_vib_peak_x': {'min': 0, 'max': 100, 'unit': 'g'},
            'before_vib_freq_x': {'min': 0, 'max': 2000, 'unit': 'Hz'},
            'before_vib_peak_y': {'min': 0, 'max': 100, 'unit': 'g'},
            'before_vib_freq_y': {'min': 0, 'max': 2000, 'unit': 'Hz'},
            'after_vib_peak_x': {'min': 0, 'max': 100, 'unit': 'g'},
            'after_vib_freq_x': {'min': 0, 'max': 2000, 'unit': 'Hz'},
            'after_vib_peak_y': {'min': 0, 'max': 100, 'unit': 'g'},
            'after_vib_freq_y': {'min': 0, 'max': 2000, 'unit': 'Hz'},
            'vib_grms_x': {'min': 9, 'max': 11, 'unit': 'g RMS'},
            'vib_grms_y': {'min': 9, 'max': 11, 'unit': 'g RMS'},
        }
        

    def extract_hpiv_data(self, output_folder: str = "") -> None:
        """
        Extract HPIV test data from PDF documents.
        
        This method processes PDF files containing HPIV acceptance test reports,
        extracts test results from multiple valves, and organizes the data into
        structured format. It handles PDF parsing, image extraction, and data
        extraction from specific pages of the test reports.
        
        The method:
        1. Opens and processes the PDF document
        2. Identifies individual valve test reports within the PDF
        3. Extracts test data from specific pages
        4. Saves individual valve reports and associated images
        5. Populates the test results with extracted values
        
        Returns:
            None: Updates self.test_results list with extracted data
        """
        #pdf_file = 'SSC-VS197-21-13_End Item Data Package_Rev.D_241219'
        split_report_folder = os.path.join(output_folder, 'HPIV_reports') if output_folder else 'HPIV_reports'
        pdf_document = fitz.open(self.pdf_file)
        if not os.path.exists(split_report_folder):
            os.makedirs(split_report_folder)
        new_acceptance_report = False
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            page_text = page.get_text()

            if 'ACCEPTANCE TEST REPORT\nValve Serial Number :' in page_text:
                #start of new report identified
                valve_serial = page_text.replace('\n','').split('Serial Number : ')[-1]
                self.get_hpiv_parameters()  # Initialize parameters for the new valve

                self.hpiv_parameters[HPIVParameters.HPIV_ID.value] = valve_serial
                self.hpivs.add(valve_serial)

                valve_folder = os.path.join(split_report_folder, valve_serial)
                os.makedirs(valve_folder, exist_ok=True)
                images_folder = os.path.join(valve_folder, "extracted_images") if output_folder else "extracted_images"
                os.makedirs(images_folder, exist_ok=True)

                new_pdf = fitz.open()
                new_pdf.insert_pdf(pdf_document, from_page=page_number, to_page=page_number+24)
                new_pdf.save(os.path.join(valve_folder, f'Acceptance_report_valve_{valve_serial}.pdf'))

                image_count = 0
                processed_images = set()  # Store hashes of already saved images
                for newpdf_page_number in range(len(new_pdf)):
                    newpdf_page = new_pdf[newpdf_page_number]

                    # Extract images from the page
                    for img_index, img in enumerate(newpdf_page.get_images(full=True)):
                        xref = img[0]
                        # Extract the image bytes
                        base_image = new_pdf.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                    
                        # Check if the image is similar to any processed image
                        is_duplicate = False
                        for saved_image in processed_images:
                            if image_bytes==saved_image:
                                # print(f"Skipped similar image on page {page_number + 1}, index {img_index + 1}")
                                is_duplicate = True
                                break
                        if is_duplicate:
                            continue
                        
                        # Save the image if it's unique
                        image_filename = f"page{page_number+1}_img{img_index+1}.{image_ext}"
                        image_path = os.path.join(images_folder, image_filename)
                    
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                    
                        # print(f"Saved: {image_path}")
                        processed_images.add(image_bytes)
                        image_count += 1
                    
                #extracting results from the report
                #proof pressure test results
                newpdf_page = new_pdf[10]
                local_text = newpdf_page.get_text().split('\n')
                self.hpiv_parameters[HPIVParameters.WEIGHT.value]['value'] = float(local_text[local_text.index('200 gr Maximum')+1].replace('g','').replace(' ',''))
                self.hpiv_parameters[HPIVParameters.PROOF_OPEN.value]['value'] = float(local_text[local_text.index('Valve Open')+1].replace('',''))
                self.hpiv_parameters[HPIVParameters.PROOF_CLOSED.value]['value'] = float(local_text[local_text.index('Valve Closed')+1].replace('',''))
                
                # leak test results
                newpdf_page = new_pdf[11]
                local_text = newpdf_page.get_text().split('\n')
                self.hpiv_parameters[HPIVParameters.LEAK_4_HP_PRESS.value]['value'] = float(local_text[local_text.index('4')+2].replace('',''))
                self.hpiv_parameters[HPIVParameters.LEAK_4_HP.value]['value'] = float(local_text[local_text.index('4')+6].split('×10')[0]) * 10**float(local_text[local_text.index('4')+6].split('×10')[1])
                self.hpiv_parameters[HPIVParameters.LEAK_4_LP_PRESS.value]['value'] = float(local_text[local_text.index('4')+8].replace('',''))  
                self.hpiv_parameters[HPIVParameters.LEAK_4_LP.value]['value'] = float(local_text[local_text.index('4')+10].split('×10')[0]) * 10**float(local_text[local_text.index('4')+10].split('×10')[1])
                self.hpiv_parameters[HPIVParameters.LEAK_6_HP_PRESS.value]['value'] = float(local_text[local_text.index('6')+2].replace('',''))
                self.hpiv_parameters[HPIVParameters.LEAK_6_HP.value]['value'] = float(local_text[local_text.index('6')+6].split('×10')[0]) * 10**float(local_text[local_text.index('6')+6].split('×10')[1])
                self.hpiv_parameters[HPIVParameters.LEAK_6_LP_PRESS.value]['value'] = float(local_text[local_text.index('6')+8].replace('',''))  
                self.hpiv_parameters[HPIVParameters.LEAK_6_LP.value]['value'] = float(local_text[local_text.index('6')+10].split('×10')[0]) * 10**float(local_text[local_text.index('6')+10].split('×10')[1])
                self.hpiv_parameters[HPIVParameters.LEAK_15_HP_PRESS.value]['value'] = float(local_text[local_text.index('15')+2].replace('',''))
                self.hpiv_parameters[HPIVParameters.LEAK_15_HP.value]['value'] = float(local_text[local_text.index('15')+6].split('×10')[0]) * 10**float(local_text[local_text.index('15')+6].split('×10')[1])
                self.hpiv_parameters[HPIVParameters.LEAK_15_LP_PRESS.value]['value'] = float(local_text[local_text.index('15')+8].replace('',''))  
                self.hpiv_parameters[HPIVParameters.LEAK_15_LP.value]['value'] = float(local_text[local_text.index('15')+10].split('×10')[0]) * 10**float(local_text[local_text.index('15')+10].split('×10')[1])
            
                # Vibration test results
                newpdf_page = new_pdf[13]
                local_text = newpdf_page.get_text().split('\n')
                vib_results_index = local_text.index('5 – 2000 Hz')
                if 'refer to' in str(local_text[vib_results_index+1]).lower():
                    self.hpiv_parameters[HPIVParameters.BEFORE_VIB_FREQ_X.value]['value'] = 0
                    self.hpiv_parameters[HPIVParameters.BEFORE_VIB_FREQ_Y.value]['value'] = 0
                    self.hpiv_parameters[HPIVParameters.BEFORE_VIB_PEAK_X.value]['value'] = 0
                    self.hpiv_parameters[HPIVParameters.BEFORE_VIB_PEAK_Y.value]['value'] = 0
                    self.hpiv_parameters[HPIVParameters.AFTER_VIB_PEAK_X.value]['value'] = 0
                    self.hpiv_parameters[HPIVParameters.AFTER_VIB_FREQ_X.value]['value'] = 0
                    self.hpiv_parameters[HPIVParameters.AFTER_VIB_PEAK_Y.value]['value'] = 0
                    self.hpiv_parameters[HPIVParameters.AFTER_VIB_FREQ_Y.value]['value'] = 0
                else:
                    self.hpiv_parameters[HPIVParameters.BEFORE_VIB_FREQ_X.value]['value'] = float(local_text[vib_results_index+1].replace('Hz','').replace(' ',''))
                    self.hpiv_parameters[HPIVParameters.BEFORE_VIB_PEAK_X.value]['value'] = float(local_text[vib_results_index+2].replace('g','').replace(' ',''))
                    self.hpiv_parameters[HPIVParameters.BEFORE_VIB_FREQ_Y.value]['value'] = float(local_text[vib_results_index+3].replace('Hz','').replace(' ',''))
                    self.hpiv_parameters[HPIVParameters.BEFORE_VIB_PEAK_Y.value]['value'] = float(local_text[vib_results_index+4].replace('g','').replace(' ',''))
                    self.hpiv_parameters[HPIVParameters.AFTER_VIB_FREQ_X.value]['value'] = float(local_text[vib_results_index+7].replace('Hz','').replace(' ',''))
                    self.hpiv_parameters[HPIVParameters.AFTER_VIB_PEAK_X.value]['value'] = float(local_text[vib_results_index+8].replace('g','').replace(' ',''))
                    self.hpiv_parameters[HPIVParameters.AFTER_VIB_FREQ_Y.value]['value'] = float(local_text[vib_results_index+9].replace('Hz','').replace(' ',''))
                    self.hpiv_parameters[HPIVParameters.AFTER_VIB_PEAK_Y.value]['value'] = float(local_text[vib_results_index+10].replace('g','').replace(' ',''))
                
                self.hpiv_parameters[HPIVParameters.VIB_GRMS_X.value]['value'] = float(local_text[local_text.index('Vibration (grms)')+1])
                self.hpiv_parameters[HPIVParameters.VIB_GRMS_Y.value]['value'] = float(local_text[local_text.index('Vibration (grms)')+2])
            
                # electric test results
                newpdf_page = new_pdf[20]
                local_text = newpdf_page.get_text().split('\n')
                self.hpiv_parameters[HPIVParameters.DIELECTRIC_STR.value]['value'] = float(local_text[local_text.index('< 2')+1].replace('',''))
                raw_value = local_text[local_text.index('> 100')+1].replace(' ', '')
                if '∞' in raw_value:
                    raw_value = raw_value.replace('∞', '1e+19')
                self.hpiv_parameters[HPIVParameters.INSULATION_RES.value]['value'] = float(raw_value)
                newpdf_page = new_pdf[21]
                local_text = newpdf_page.get_text().split('\n')
                self.hpiv_parameters[HPIVParameters.POWER_TEMP.value]['value'] = float(local_text[local_text.index('20±2')+1].replace('',''))
                self.hpiv_parameters[HPIVParameters.POWER_RES.value]['value'] = float(local_text[local_text.index('43.3±0.5')+1].replace('',''))
                self.hpiv_parameters[HPIVParameters.POWER_POWER.value]['value'] = float(local_text[local_text.index('< 10')+1].replace('',''))
                self.hpiv_parameters[HPIVParameters.EXT_LEAK.value]['value'] = float(local_text[local_text.index('≤ 1.0×10-10')+1].split('×10')[0]) * 10**float(local_text[local_text.index('≤ 1.0×10-10')+1].split('×10')[1])
                
                newpdf_page = new_pdf[22]
                local_text = newpdf_page.get_text().split('\n')
                self.hpiv_parameters[HPIVParameters.PULLIN_PRES.value]['value'] = float(local_text[local_text.index('> 310')+1].replace('',''))
                self.hpiv_parameters[HPIVParameters.PULLIN_VOLT.value]['value'] = float(local_text[local_text.index('< 18')+1].replace('',''))
                try:
                    self.hpiv_parameters[HPIVParameters.DROPOUT_VOLT.value]['value'] = float(local_text[local_text.index('> 2 ')+1].replace('',''))
                except Exception as e:
                    self.hpiv_parameters[HPIVParameters.DROPOUT_VOLT.value]['value'] = float(local_text[local_text.index('2<x<3.1 Vdc')+1].replace('',''))
                
                newpdf_page = new_pdf[23]
                local_text = newpdf_page.get_text().split('\n')
                self.hpiv_parameters[HPIVParameters.RESPO_PRES.value]['value'] = float(local_text[local_text.index('> 310')+1].replace('',''))
                self.hpiv_parameters[HPIVParameters.RESPO_VOLT.value]['value'] = float(local_text[local_text.index('18±0.2')+1].replace('',''))
                self.hpiv_parameters[HPIVParameters.RESPO_TIME.value]['value'] = float(local_text[local_text.index('< 20')+1].replace('',''))
                local_text = local_text[local_text.index('< 20')+1:]
                self.hpiv_parameters[HPIVParameters.RESPC_VOLT.value]['value'] = float(local_text[local_text.index('32±0.2')+1].replace('',''))
                self.hpiv_parameters[HPIVParameters.RESPC_TIME.value]['value'] = float(local_text[local_text.index('< 20')+1].replace('',''))
                self.hpiv_parameters[HPIVParameters.PRESSD.value]['value'] = float(local_text[local_text.index('< 1')+1].replace('',''))
                self.hpiv_parameters[HPIVParameters.FLOWRATE.value]['value'] = float(local_text[local_text.index('> 10')+1].replace('',''))           
                
                # cleanliness results
                newpdf_page = new_pdf[24]
                local_text = newpdf_page.get_text().split('\n')
                self.hpiv_parameters[HPIVParameters.CLEANLINESS_6_10.value]['value'] = float(local_text[local_text.index('6-10')+2].replace('-','0'))
                self.hpiv_parameters[HPIVParameters.CLEANLINESS_11_25.value]['value'] = float(local_text[local_text.index('11-25')+2].replace('-','0'))
                self.hpiv_parameters[HPIVParameters.CLEANLINESS_26_50.value]['value'] = float(local_text[local_text.index('26-50')+2].replace('-','0'))
                self.hpiv_parameters[HPIVParameters.CLEANLINESS_51_100.value]['value'] = float(local_text[local_text.index('51-100')+2].replace('-','0'))
                self.hpiv_parameters[HPIVParameters.CLEANLINESS_100.value]['value'] = float(local_text[local_text.index('Over 100')+2].replace('-','0'))

                self.test_results.append(self.hpiv_parameters)
            
            if "HARDWARE" in page_text and 'Part' in page_text:
                local_text: list[str] = page_text.split('\n')
                next_page = pdf_document[page_number+1]
                local_text.extend(next_page.get_text().split('\n'))
                next_page2 = pdf_document[page_number+2]
                local_text.extend(next_page2.get_text().split('\n'))

                for part_name, part_number in self.part_number_map.items():
                    try:
                        rev = local_text[local_text.index(part_number)+1]
                    except Exception as e:
                        continue
                    self.revision_data[part_number] = {
                        'revision': rev,
                        'part_name': part_name.value
                    }

        pdf_document.close()
        self.check_within_limits() 
        
    def check_within_limits(self) -> None:
        """
        Check if test result values are within defined limits.
        This method iterates through the test results stored in the test_results
        attribute and checks each parameter's value against its defined minimum
        and maximum limits. It updates the test results to indicate whether each
        parameter is within limits, on the limit, or out of limits.
        Returns:
            None: Updates self.test_results in place
        """
        for result in self.test_results:
            for param, values in result.items():

                if isinstance(values, dict):
                    try:
                        min_value = float(values.get('min', 0))
                        max_value = float(values.get('max', 0))
                        value = float(values.get('value', None))
                    except (ValueError, TypeError):
                        min_value = 0
                        max_value = 0
                        value = None
                    if value is not None:
                        if value < min_value or value > max_value:
                            result[param]['within_limits'] = LimitStatus.FALSE
                        elif min_value < value < max_value:
                            result[param]['within_limits'] = LimitStatus.TRUE
                        else:
                            result[param]['within_limits'] = LimitStatus.ON_LIMIT
                
    def get_ocr_certification(self) -> None:
        """
        Use OCR to extract certification information from the PDF.
        This method utilizes the OCRReader class to extract text from the provided PDF
        file and searches for HPIV IDs within the extracted text. It populates the
        hpiv_ids attribute with the found IDs and sets the certification attribute
        based on the extracted data.
        """
        ocr_reader = OCRReader(pdf_file=self.pdf_file)
        ocr_reader.main_delivery_slip_reader(part_type='hpiv')
        self.total_lines = ocr_reader.total_lines
        self.certification = ocr_reader.certification
        if self.total_lines:
            matches = re.findall(r'\bvs197-\d{3}\b', self.total_lines.lower(), flags=re.IGNORECASE)
            self.hpiv_ids = [m.upper() for m in matches]    
            self.hpiv_ids = sorted(set(self.hpiv_ids)) 

    def get_certification(self, total_lines: list[str]) -> None:
        """
        Extract certification information from the provided text lines.
        This method processes a list of text lines to identify HPIV IDs and
        extract the certification number from the filename. It populates the
        hpiv_ids attribute with the found IDs and sets the certification attribute
        based on the extracted data.
        Args:
            total_lines (list[str]): List of text lines to process
        """
        # Extract certification number from filename
        match = re.search(r'C\d{2}-\d{4}', os.path.basename(self.pdf_file))
        self.certification = match.group(0) if match else None

        self.hpiv_ids = []

        # Loop through all lines and extract VS197 IDs
        for line in total_lines:
            # First, look for explicit VS197 IDs
            found_ids = re.findall(r'vs197-(\d{3})', line, flags=re.IGNORECASE)
            self.hpiv_ids.extend([f'VS197-{num.zfill(3)}' for num in found_ids])

            # Then check for ranges like vs197-064 ... vs197-070
            range_match = re.search(r'vs197-(\d{3})\s*[-–]\s*vs197-(\d{3})', line, flags=re.IGNORECASE)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                self.hpiv_ids.extend([f'VS197-{i:03d}' for i in range(start, end + 1)])

        # Remove duplicates and sort
        self.hpiv_ids = sorted(set(self.hpiv_ids))
        print(self.hpiv_ids)

        # print(self.hpiv_ids)

    def get_hpiv_limits_from_excel(self, excel_file: str = "Excel_templates/hpiv_results.xlsx") -> None:
        """
        Extract HPIV limits from an Excel file.
        
        This method reads an Excel file containing HPIV parameter limits and
        populates the hpiv_parameters dictionary with the minimum and maximum
        values for each test parameter.
        
        The method:
        1. Opens the Excel workbook specified in self.excel_file
        2. Iterates through the columns to extract min/max values
        3. Updates the hpiv_parameters dictionary with the limits
        
        Returns:
            None: Updates self.hpiv_parameters dictionary in place
        """
        self.get_hpiv_parameters()  # Initialize parameters before extracting limits

        workbook = openpyxl.load_workbook(excel_file, data_only=True)
        sheet = workbook.active

        for idx,col in enumerate(sheet.iter_cols(min_row=2, max_row=3, min_col=2)):
            parameter_name = self.parameter_names[idx]
            if parameter_name == HPIVParameters.HPIV_ID.value:
                continue
            min_value = col[0].value
            max_value = col[1].value
            self.hpiv_parameters[parameter_name]['min'] = min_value if min_value is not None else 0
            self.hpiv_parameters[parameter_name]['max'] = max_value if max_value is not None else 0

class HPIVLogicSQL:
    """
    HPIVLogicSQL is the base class that handles the SQL operations for HPIV data.
    It listens for new HPIV data packages and certifications,
    processes the data, and updates the database accordingly.

    Attributes
    ----------
    Session : sessionmaker
        SQLAlchemy session factory for database interactions.
    fms : FMS_logic
        Instance of FMS_logic for additional functionality.
    hpiv_data_packages : str
        Directory path for HPIV data packages.
    hpiv_certifications : str
        Directory path for HPIV certifications.
    hpiv_certification : dict
        Stores the latest HPIV certification data.

    Methods
    -------
    listen_to_hpiv_data():
        listen for new HPIV data packages and process them.
    listen_to_hpiv_certification():
        Listen for new HPIV certification files in the specified folder.
    update_hpiv_characteristics():
        Update HPIV characteristics in the database with extracted test results.
    update_hpiv_revisions():
        Update HPIV revisions in the database with extracted revision data.
    update_hpiv_certifications():
        Update HPIV certifications in the database with extracted data.
    """

    def __init__(self, session: "Session", fms: "FMSDataStructure"):
        self.Session = session
        self.fms = fms
        self.hpiv_certification = None

    def listen_to_hpiv_data(self, hpiv_data_packages: str = "HPIV_data_packages") -> None:
        """
        Listen for new HPIV data packages and process them.
        
        This method runs in a separate thread to continuously monitor for new PDF files
        containing HPIV test data. It handles errors gracefully to ensure the listener
        continues running even when processing fails.
        """
        data_folder = os.path.join(os.getcwd(), hpiv_data_packages)
        
        try:
            self.hpiv_listener = HPIVDataListener(data_folder)
            print(f"Started monitoring HPIV data packages in: {data_folder}")
            while True:
                try:
                    time.sleep(1)  # Keep the script running to monitor for new files
                    
                    # Check if listener has processed new data
                    if hasattr(self.hpiv_listener, 'processed') and self.hpiv_listener.processed:

                        if hasattr(self.hpiv_listener, 'tr') and self.hpiv_listener.tr:
                            self.hpiv_test_results = self.hpiv_listener.tr.test_results
                            self.hpivs = self.hpiv_listener.tr.hpivs
                            self.hpiv_certification = self.hpiv_listener.tr.certification
                            self.hpiv_revisions = self.hpiv_listener.tr.revision_data
                            self.update_hpiv_characteristics()
                            self.update_hpiv_revisions()
                            # Reset the processed flag
                            self.hpiv_listener.processed = False
                            
                except Exception as e:
                    print(f"Error in listener loop: {str(e)}")
                    print("Listener will continue monitoring...")
                    traceback.print_exc()
                    
        except KeyboardInterrupt:
            print("Stopping HPIV data listener...")
            if hasattr(self, 'hpiv_listener') and self.hpiv_listener:
                self.hpiv_listener.observer.stop()
                self.hpiv_listener.observer.join()
        except Exception as e:
            print(f"Fatal error in HPIV data listener: {str(e)}")
            traceback.print_exc()
            # Try to restart the listener after a brief delay
            time.sleep(5)
            print("Attempting to restart HPIV data listener...")
            self.listen_to_hpiv_data(hpiv_data_packages=hpiv_data_packages)
    
    def listen_to_hpiv_certification(self, hpiv_certifications: str = "certifications") -> None:
        """
        Listen for HPIV certification updates.
        This method continuously monitors a specified folder for new HPIV certification files.
        When a new certification file is detected, it extracts the relevant data and updates the database.
        """
        data_folder = os.path.join(os.getcwd(), hpiv_certifications)

        try:
            self.hpiv_certification_listener = HPIVDataListener(data_folder)
            print(f"Started monitoring HPIV certifications in: {data_folder}")
            # Placeholder for future implementation
            while True:
                try:
                    time.sleep(1)
                    # Check if listener has processed new certification data
                    if hasattr(self.hpiv_certification_listener, 'processed') and self.hpiv_certification_listener.processed:
                        
                        if hasattr(self.hpiv_certification_listener, 'tr') and self.hpiv_certification_listener.tr:
                            self.hpiv_certification = self.hpiv_certification_listener.tr.certification
                            self.hpiv_ids = self.hpiv_certification_listener.tr.hpiv_ids
                        
                            self.update_hpiv_certifications()

                            # Reset the processed flag
                            self.hpiv_certification_listener.processed = False
                except Exception as e:
                    print(f"Error in certification listener loop: {str(e)}")
                    print("Listener will continue monitoring...")
                    traceback.print_exc()
        except KeyboardInterrupt:
            print("Stopping HPIV certification listener...")
            if hasattr(self, 'hpiv_certification_listener') and self.hpiv_certification_listener:
                self.hpiv_certification_listener.observer.stop()
                self.hpiv_certification_listener.observer.join()
        except Exception as e:
            print(f"Fatal error in HPIV certification listener: {str(e)}")
            traceback.print_exc()
            # Try to restart the listener after a brief delay
            time.sleep(5)
            print("Attempting to restart HPIV certification listener...")
            self.listen_to_hpiv_certification(hpiv_certifications=hpiv_certifications)

    def update_hpiv_characteristics(self, hpiv_data: HPIVData = None) -> None:
        """
        Update HPIV characteristics in the database with extracted test results.
        
        This method processes the test results and updates the database with
        the extracted HPIV characteristics. It includes error handling to ensure
        database issues don't crash the listener.
        Args:
            hpiv_data (HPIVData, optional): The HPIV test results data to process.
                If None, uses self.hpiv_test_results attribute obtained from the listening event. Defaults to None.
        """
        self.hpiv_test_results: list[dict[str, Any]] = hpiv_data.test_results
        session: "Session" = None
        try:
            session = self.Session()
            if not hasattr(self, 'hpiv_test_results') or not self.hpiv_test_results:
                print("No HPIV test results to process")
                return
                
            for hpiv in self.hpiv_test_results:
                try:
                    serialnr = hpiv.get(HPIVParameters.HPIV_ID.value)
                    if not serialnr:
                        print("Warning: HPIV data missing serial number, skipping...")
                        continue
                    characteristics = session.query(HPIVCharacteristics).filter_by(
                        hpiv_id=serialnr).all()
                    certification_entry = session.query(HPIVCertification).filter_by(
                        hpiv_id=serialnr).first()
                    if not certification_entry:
                        new_entry = HPIVCertification(
                            hpiv_id=serialnr,
                            certification=self.hpiv_certification if self.hpiv_certification else None,
                            allocated=None
                        )
                        session.add(new_entry)
                    for param, values in hpiv.items():
                        if characteristics:
                            print(f"Found existing characteristics in DB for {serialnr}")
                            break
                        if param == HPIVParameters.HPIV_ID.value:
                            continue
                        # Handle cases where values might not be in expected format
                        if not isinstance(values, dict):
                            print(f"Warning: Invalid data format for parameter {param}, skipping...")
                            continue
                            
                        value = values.get('value')

                        if (isinstance(value, float) and np.isnan(value)) or str(value).lower() == "nan":
                            continue
                        min_value = values.get('min', 0)  # Default to 0 if not provided
                        max_value = values.get('max', 100)  # Default to 100 if not provided
                        unit = values.get('unit', 'units')  # Default to 'units' if not provided
                        within_limits_str = values.get('within_limits', None)
                        within_limits = None
                        if within_limits_str is not None:
                            try:
                                within_limits = LimitStatus(within_limits_str)
                            except ValueError:
                                print(f"Invalid within_limits value: {within_limits_str}")

                        new_characteristic = HPIVCharacteristics(
                            hpiv_id=serialnr,
                            parameter_name=param,
                            parameter_value=value,
                            min_value=min_value,  
                            max_value=max_value,  
                            unit=unit, 
                            within_limits=within_limits
                        )
                        session.add(new_characteristic)
                            
                except Exception as e:
                    print(f"Error processing HPIV {serialnr}: {str(e)}")
                    continue

            if self.hpiv_certification:
                # Add or update HPIV certification information
                certifications = session.query(HPIVCertification).filter_by(certification = self.hpiv_certification).all()
                for hpiv_id in self.hpivs:
                    if certifications:
                        print(f"Found existing certifications in DB for {self.hpiv_certification}")
                        return
                    else:
                        new_certification = HPIVCertification(
                            hpiv_id=hpiv_id,
                            certification=self.hpiv_certification,
                            allocated=None
                        )
                        session.add(new_certification)
                        
            session.commit()
            # print(f"Successfully updated database with {len(self.hpiv_test_results)} HPIV records")
            self.fms.print_table(HPIVCharacteristics)
            
        except Exception as e:
            print(f"Error updating HPIV characteristics: {str(e)}")
            if session:
                session.rollback()
            traceback.print_exc()
        finally:
            if session:
                session.close()

    def update_hpiv_revisions(self, hpiv_data: HPIVData = None) -> None:
        """
        Update HPIV part revisions in the database.
        
        This method processes the part revision data and updates the database
        with the latest revision information. It includes error handling to ensure
        database issues don't crash the listener.
        Args:
            hpiv_data (HPIVData, optional): The HPIV test results data to process.
                If None, uses self.hpiv_revisions attribute obtained from the listening event. Defaults to None.
        """
        session: "Session" = None
        self.hpiv_revisions: dict[str, dict] = hpiv_data.revision_data
        self.hpivs = hpiv_data.hpivs
        try:
            session = self.Session()
            if not hasattr(self, 'hpiv_revisions') or not self.hpiv_revisions:
                print("No HPIV revision data to process")
                return
            revision_check = session.query(HPIVRevisions).filter(HPIVRevisions.hpiv_id.in_(self.hpivs)).all()
            if revision_check:
                print(f"Found existing revisions in DB for HPIVs: {self.hpivs}")
                return
            for part_number, rev_data in self.hpiv_revisions.items():
                try:
                    for serial in self.hpivs:
                        new_revision = HPIVRevisions(
                            part_number=part_number,
                            revision=rev_data.get('revision'),
                            part_name=rev_data.get('part_name'),
                            hpiv_id=serial
                        )
                        session.add(new_revision)

                except Exception as e:
                    print(f"Error processing HPIV revision for {part_number}: {str(e)}")
                    continue

            session.commit()
            print(f"Successfully updated database with HPIV part revisions")
            self.fms.print_table(HPIVRevisions)
            
        except Exception as e:
            print(f"Error updating HPIV revisions: {str(e)}")
            if session:
                session.rollback()
            traceback.print_exc()
        finally:
            if session:
                session.close()

    def update_hpiv_certifications(self, hpiv_certification: HPIVData = None):
        """
        Update HPIV certifications in the database.
        
        This method processes the HPIV certification data and updates the database
        with the latest certification information. It includes error handling to ensure
        database issues don't crash the listener.
        """
        session: "Session" = None
        self.hpiv_certification: str = hpiv_certification.certification
        self.hpiv_ids: list[str] = hpiv_certification.hpiv_ids
        try:
            session = self.Session()
            if not hasattr(self, 'hpiv_certification') or not self.hpiv_certification:
                print("No HPIV certification data to process")
                return
            certifications = session.query(HPIVCertification).filter_by(certification = self.hpiv_certification).all()
            for hpiv_id in self.hpiv_ids:
                try:
                    if certifications:
                        print(f"Found existing certifications in DB for {self.hpiv_certification}")
                        return
                    hpiv_id_check = session.query(HPIVCertification).filter_by(hpiv_id=hpiv_id).first()
                    if hpiv_id_check:
                        hpiv_id_check.certification = self.hpiv_certification
                        session.merge(hpiv_id_check)
                    else:
                        new_certification = HPIVCertification(
                            hpiv_id=hpiv_id,
                            certification=self.hpiv_certification,
                            allocated=None
                        )
                        session.add(new_certification)
                        
                except Exception as e:
                    print(f"Error processing HPIV certification for {hpiv_id}: {str(e)}")
                    continue

            session.commit()
            print(f"Successfully updated database with HPIV certifications")
            self.fms.print_table(HPIVCertification)
            
        except Exception as e:
            print(f"Error updating HPIV certifications: {str(e)}")
            if session:
                session.rollback()
            traceback.print_exc()
        finally:
            if session:
                session.close()

    def hpiv_query(self):
        query = HPIVQuery(session=self.Session(), fms_entry=None, hpiv_certification=HPIVCertification, hpiv_characteristics=HPIVCharacteristics,
                        hpiv_revisions=HPIVRevisions, limit_status=LimitStatus)
        
        query.hpiv_query_field()


if __name__ == "__main__":
    # Example usage
    file = r"certifications\C24-0192 Space Solutions Co 512146.pdf"
    # Example usage
    hpiv_data = HPIVData(pdf_file = file)
    company = "Space Solutions"
    # reader = TextractReader(pdf_file=file, bucket_folder="Certifications", company=company)
    # total_lines = reader.get_text()
    # hpiv_data.get_certification(total_lines)
    hpiv_data.get_ocr_certification()
    print(hpiv_data.hpiv_ids)
