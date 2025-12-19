# Standard library
import os
import re
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path

# Third-party
import fitz
from tqdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Local packages – database models
from .db import (
    Base,
    FMSMain,
    TVCertification,
    TVStatus,
    TVTestRuns,
    FMSProcedures,
)

# Local packages – utilities
from .utils.tv import TVData, TVLogicSQL
from .utils.hpiv import HPIVData, HPIVLogicSQL
from .utils.lpt_manifold import ManifoldData, ManifoldLogicSQL
from .utils.fr import FRData, FRLogicSQL
from .utils.fms import FMSData, FMSLogicSQL
from .utils.general_utils import TVParts, load_from_json, save_to_json

# Local packages – queries and processing
from .utils.fms_query import FMSQuery
from .utils.certification_listener import CertificationListener
from .utils.tv_assembly import TVAssembly
from .utils.hpiv_query import HPIVQuery
from .utils.tv_query import TVQuery
from .utils.manifold_query import ManifoldQuery
from .utils.textract import TextractReader
from .utils.fms_testing import FMSTesting
from .utils.tv_assembly import TVAssembly


#pip install sqlalchemy tqdm networkx openpyxl chardet pymupdf docxtpl docx2pdf soupsieve pytesseract pdf2image opencv-python ipywidgets ipyvuetify tzlocal streamlit watchdog flask requests
#TODO discuss which new measurements are going to be done by SK, and include those, or which measurements in general
#TODO safran p/n
#TODO capture point in MAIT for fms flow testing, discussion
#TODO think about piece parts batch traceability
#TODO tools and test operator for TV

#TODO thermal valve measurement report, extract measurements
#TODO add 'dashboard' especially showing what data has come in and any errors regarding the processing
#TODO maybe add table for tolerances, from the drawings
#TODO session start logic, make robust against connectivity issues
#TODO allocation queries
#TODO tv test input field tv id selection not disabled

class FMSDataStructure:
    """
    Main class for managing FMS (Fluidic Management System) data structure.
    
    This class handles the database operations for HPIV, TV, LPT manifold, and FRs
    test data, including listening for new data packages, processing PDF files,
    and updating the database with extracted characteristics.
    
    Attributes
    ----------
        engine: 
            SQLAlchemy database engine.
        Session: 
            SQLAlchemy session factory.
        hpiv_data_packages (str): 
            Path to directory containing HPIV data packages.
        tv_test_runs (str): 
            Path to directory containing TV test runs.
        certifications (str): 
            Path to directory containing certification documents.
        lpt_files (str): 
            Path to directory containing LPT manifold data.
        fms_main_files (str): 
            Path to directory containing FMS main test results.
        excel_extraction (bool): 
            Flag to enable/disable Excel data extraction.
        fms_main_files (str):
            Path to FMS main test results.
        companies (list):
            List of company names for certification filtering.
        base_path (str):
            Base path for certification documents.
        test_path (str):
            Path for FMS testing documents/files.
        anode_fr_path (str):
            Path to anode FR test results Excel file.
        cathode_fr_path (str):
            Path to cathode FR test results Excel file.
        fms_status_path (str):
            Path to FMS status overview Excel file.
        tv_assembly_path (str):
            Path to TV assembly procedure Excel file.
        tv_summary_path (str):
            Path to TV summary database Excel file.
        tv_test_path (str):
            Path to TV test results directory.
        certification_folder (str):
            Path to certification documents folder.
        tv_data (TVData): 
            Instance for handling TV data.
        fr_data (FRData): 
            Instance for handling FR data.
        hpiv_data (HPIVData): 
            Instance for handling HPIV data.
        lpt_data (ManifoldData): 
            Instance for handling LPT manifold data.
        hpiv_sql (HPIVLogicSQL): 
            SQL logic handler for HPIV data.
        tv_sql (TVLogicSQL): 
            SQL logic handler for TV data.
        lpt_sql (ManifoldLogicSQL): 
            SQL logic handler for LPT manifold data.
        fr_sql (FRLogicSQL): 
            SQL logic handler for FR data.
        fms_sql (FMSLogicSQL): 
            SQL logic handler for FMS main data.
        obj_list (list): 
            List of data handler instances.
        lpt_found (bool): 
            Flag indicating if LPT data was found.
        fr_found (bool): 
            Flag indicating if FR data was found.
        hpiv_found (bool): 
            Flag indicating if HPIV data was found.
        certification_listener (CertificationListener): 
            Listener for certification documents.
        all_current_certifications (list): 
            List of all current certification file paths.
        hpiv_test_results (list): 
            List of processed HPIV test results.

    Methods
    -------
    print_table_structures():
        Print the structure of all database tables in the schema.
    listen_to_certifications():
        Start listening for new certification documents,
        using the CertificationListener.
    listen_to_hpiv_data():
        Start listening for new HPIV data packages.
    listen_to_tv_test_results():
        Start listening for new TV test results.
    add_tv_electrical_data():
        Add TV electrical data to the database.
    listen_to_lpt_calibration():
        Start listening for new LPT manifold calibration data.
    add_tv_assembly_data():
        Add TV assembly data from Excel files to the database.
    add_manifold_assembly_data():
        Add manifold assembly data from Excel files to the database.
    add_fr_test_data():
        Add FR test data from Excel files to the database.
    electric_assembly():
        Add TV electric assembly input field data to the database.
    flow_restrictor_testing():
        Add FR flow test input data to the database.
    listen_to_fms_main_results():
        Start listening for new FMS main test results.
    estimate_amazon_invoice_total_cost(files, s3_rate_per_gb=0.023, textract_rate_per_1000=1.50):
        Estimate the total cost of processing PDF files using Amazon S3 and Textract.
    get_all_certifications():
        Process all certification documents and update the database accordingly.
    add_tv_test_results():
        Add TV test results from Excel files to the database.
    add_hpiv_data():
        Add HPIV data from PDF files to the database.
    add_lpt_calibration_data():
        Add LPT manifold calibration data from JSON files to the database.
    add_fms_main_test_data():
        Add FMS main test data from the test reports to the database.
    save_procedure(self, procedure_name, procedure_json):
        Save a procedure JSON to the database.
    load_procedure(self, procedure_name, version):
        Load a procedure JSON from the database.
    get_all_current_data():
        Transfer all current local, de-centralized data into the right structure 
        and update to the database.
    update_procedure(procedure_name):
        Gets a local JSON procedure and uploads it to the database.
    print_table(table_class, limit):
        Print rows of a given table class from the database, to a specified limit.
    """
    
    def __init__(self, excel_extraction: bool = True, test_path: str = r"\\be.local\Doc\DocWork\20025 - CHEOPS2 Low Power\70 - Testing",
                 absolute_data_dir: str = r"C:\\Users\\TANTENS\\Documents\\fms_data_collection", local = True) -> None:

        if local:
            local_appdata = Path(os.environ.get("LOCALAPPDATA", os.getcwd()))
            app_data_dir = local_appdata / "FMSDatabase"
            app_data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = app_data_dir / "FMS_DataStructure.db"

            # Initialize the engine
            self.engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.default_manifold_drawing = "20025.10.08-R4"    
        self.excel_extraction = excel_extraction
        self.companies = [
            'sk technology', 'sk', 'veldlaser', 'ceratec', 'pretec', 'veld laser',
            'branl', 'ejay filtration', 'space solutions', 'spacesolutions',
            'keller', 'coremans'
        ]

        self.absolute_data_dir = absolute_data_dir
        
        year = datetime.now().year
        base_path = r"\\be.local\Doc\DocWork\Certificaten Bradford"
        self.test_path = test_path
        self.fms_status_path = r"\\be.local\Doc\DocWork\20025 - CHEOPS2 Low Power\LP FMS Status Overview_V0.xlsx"
        self.anode_fr_path = r"\\be.local\Doc\DocWork\20025 - CHEOPS2 Low Power\70 - Testing\LP FMS FR Testing - 6\FMS-LP-BE-TRS-0021-i1-0 - FR Testing - Anode.xlsx"
        self.cathode_fr_path = r"\\be.local\Doc\DocWork\20025 - CHEOPS2 Low Power\70 - Testing\LP FMS FR Testing - 6\FMS-LP-BE-TRS-0021-i1-0 - FR Testing - Cathode.xlsx"
        self.tv_assembly_path = r"\\be.local\Doc\DocWork\20025 - CHEOPS2 Low Power\70 - Testing\Thermal Valve Assembly Testing\TV Assembly Procedure_V1.xlsx"
        self.tv_summary_path = r"\\be.local\Doc\DocWork\20025 - CHEOPS2 Low Power\70 - Testing\Thermal Valve Assembly Testing\Thermal valve database_V2.xlsx"
        self.tv_test_path = r"\\be.local\Doc\DocWork\20025 - CHEOPS2 Low Power\70 - Testing\Thermal Valve Assembly Testing"
        self.default_certifications_path = os.path.join(absolute_data_dir, "certifications")
        self.certification_folder = os.path.join(base_path, str(year))

        self.hpiv_sql = HPIVLogicSQL(session=self.Session, fms=self)
        
        self.tv_sql = TVLogicSQL(session=self.Session, fms=self)
        
        self.lpt_sql = ManifoldLogicSQL(session=self.Session, fms=self)
        
        self.fr_sql = FRLogicSQL(session=self.Session, fms=self)
        
        self.fms_sql = FMSLogicSQL(session=self.Session, fms=self)
        
        self.tv_data = TVData()
        self.fr_data = FRData()
        self.hpiv_data = HPIVData()
        self.lpt_data = ManifoldData()

        self.obj_list = [self.tv_data, self.fr_data, self.hpiv_data, self.lpt_data]
        self.lpt_found = False
        self.fr_found = False
        self.hpiv_found = False

        self.certification_listener = None

    def print_table_structures(self) -> None:
        """
        Print the structure of all database tables.
        
        This method displays the table names and their column definitions
        for all tables in the database schema.
        """
        session = self.Session()
        for table_name, table in Base.metadata.tables.items():
            print(f"\nTable: {table_name}")
            for column in table.columns:
                print(f"  {column.name} ({column.type})")
        session.close()

    def fms_query(self, specification_dict: dict = {
            "lpt_pressures": [0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.4], 
            "max_opening_response": 300,
            "max_response": 60,
            "lpt_voltages": [10, 15, 17, 20, 24, 25, 30, 35], 
            "min_flow_rates": [0.61, 1.23, 1.51, 1.85, 2.40, 2.43, 3.13, 3.72], 
            "max_flow_rates": [0.96, 1.61, 1.9, 2.34, 2.93, 3.07, 3.81, 4.54], 
            "range12_low": [13, 41], "range24_low": [19, 54],
            "range12_high": [25, 95], "range24_high": [35, 140], 
            "initial_flow_rate": 0.1, "lpt_set_points": [1, 1.625, 2.25, 1.625, 1, 0.2]
        }):
        """
        Create and start an FMSQuery instance.
        Args:
            specification_dict (dict): Dictionary containing query specifications.
        """
        query = FMSQuery(session=self.Session(), **specification_dict)
        query.fms_query_field()

    def manifold_query(self, specification_dict: dict = {
            "min_radius": 0.22, "max_radius": 0.25, "ref_thickness": 0.25, 
            "thickness_tol": 0.01, "anode_reference_orifice": 0.07095, "cathode_reference_orifice": 0.01968,
            "anode_target_15": 3.006, "anode_target_24": 4.809, "cathode_target_15": 0.231, "cathode_target_24": 0.370, 
            "pressure_threshold": 0.2, "signal_tolerance": 0.05, "signal_threshold": 7.5, "anode_fs": 20, "cathode_fs": 2,
            "fs_error": 0.001, "reading_error": 0.005, "xenon_density": 5.894, "krypton_density": 3.749
        }):
        """
        Create and start a ManifoldQuery instance.
        Args:
            specification_dict (dict): Dictionary containing query specifications.
        """
        query = ManifoldQuery(session=self.Session(), **specification_dict)
        query.manifold_query_field()

    def tv_query(self, specification_dict: dict = {}):
        """
        Create and start a TVQuery instance.
        Args:
            specification_dict (dict): Dictionary containing query specifications.
        """
        query = TVQuery(session=self.Session(), **specification_dict)
        query.tv_query_field()

    def hpiv_query(self, specification_dict: dict = {}):
        """
        Create and start an HPIVQuery instance.
        Args:
            specification_dict (dict): Dictionary containing query specifications.
        """
        query = HPIVQuery(session=self.Session(), **specification_dict)
        query.hpiv_query_field()

    def tv_assembly(self, word_path: str = "TVAssemblyProcedure\\tv_assembly_procedure.docx", main_path: str = "TVAssemblyProcedure"):
        """
        Create and start a TVAssembly instance.
        Args:
            word_path (str): Path to the Word template for TV assembly.
            main_path (str): Main path for TV assembly files.    
        """
        assembly = TVAssembly(session=self.Session(), fms=self, word_template_path=word_path, main_path=main_path)
        assembly.start_assembly()

    def fms_acceptance_testing(self, word_template_path: str = r"FMSAcceptanceTests\fms_test_draft.docx",  
                 save_path: str = r"\\be.local\Doc\DocWork\99999 - FMS industrialisation\40 - Engineering\03 - Data flow\FMS Acceptance Test Reports"):
        """
        Create and start an FMSTesting instance.    
        Args:
            word_template_path (str): Path to the Word template for FMS acceptance testing.
            save_path (str): Path to save the generated test reports.
        """
        testing = FMSTesting(session=self.Session(), fms=self, word_template_path=word_template_path, save_path=save_path)
        testing.start_testing()

    def listen_to_certifications(self, certifications: str = "") -> None:
        """
        Start listening for new certification documents.
        This method initializes the CertificationListener to monitor the
        certifications directory for new or modified files.
        """
        if not certifications:
            certifications = self.default_certifications_path
        self.certification_listener = CertificationListener(fms=self, path=certifications, load_json=load_from_json, save_json=save_to_json)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.certification_listener.observer.stop()
            self.certification_listener.observer.join()

    def listen_to_hpiv_data(self, hpiv_data_packages: str = "") -> None:
        """
        Start listening for new HPIV data packages.
        This method initializes a thread to monitor the HPIV data packages
        directory for new or modified files.
        """
        if not hpiv_data_packages:
            hpiv_data_packages = os.path.join(self.absolute_data_dir, "HPIV_data_packages")
        self.hpiv_data_thread = threading.Thread(target=self.hpiv_sql.listen_to_hpiv_data, args=(hpiv_data_packages,), daemon=False)
        self.hpiv_data_thread.start()

    def listen_to_tv_test_results(self, tv_test_runs: str = r"") -> None:
        """
        Start listening for new TV test results.
        This method initializes a thread to monitor the TV test results
        directory for new or modified files.
        Args:
            tv_test_runs (str): Path to the directory containing TV test runs.
        """
        if not tv_test_runs:
            tv_test_runs = os.path.join(self.absolute_data_dir, r"TV_test_runs")
        self.tv_sql.listen_to_tv_test_results(tv_test_runs = tv_test_runs)
        # self.tv_test_thread = threading.Thread(target=self.tv_sql.listen_to_tv_test_results, args=(tv_test_runs,), daemon=False)
        # self.tv_test_thread.start()

    def add_tv_electrical_data(self, electrical_data: str = "") -> None:
        """
        Add TV electrical data to the database.
        Args:
            electrical_data (str): Path to the directory containing TV electrical data.
        """
        if not electrical_data:
            electrical_data = os.path.join(self.absolute_data_dir, "Electrical_data")
            print(electrical_data)
        print(electrical_data)
        self.tv_sql.add_electrical_data(electrical_data = electrical_data)

    def listen_to_lpt_calibration(self, lpt_calibration: str = "") -> None:
        """
        Start listening for new LPT manifold calibration data.
        This method initializes a thread to monitor the LPT manifold
        calibration directory for new or modified files.
        Args:
            lpt_calibration (str): Path to the directory containing LPT calibration data.
        """
        if not lpt_calibration:
            lpt_calibration = os.path.join(self.absolute_data_dir, "LPT_data/LPT_coefficient_data")
        self.lpt_calibration_thread = threading.Thread(target=self.lpt_sql.listen_to_lpt_calibration, args=(lpt_calibration,), daemon=False)
        self.lpt_calibration_thread.start()

    def add_tv_assembly_data(self, tv_assembly: str = "", tv_summary: str = "", status_file: str = "") -> None:
        """
        Add TV assembly data from Excel files to the database.
        Args:
            tv_assembly (str): Path to the TV assembly procedure Excel file.
            tv_summary (str): Path to the TV summary database Excel file.
            status_file (str): Path to the FMS status overview Excel file.
        """
        if not tv_assembly:
            tv_assembly = self.tv_assembly_path
        if not tv_summary:
            tv_summary = self.tv_summary_path
        if not status_file:
            status_file = self.fms_status_path
        self.tv_sql.add_tv_assembly_data(self.excel_extraction, tv_assembly = tv_assembly, tv_summary = tv_summary, status_file = status_file)

    def add_manifold_assembly_data(self, status_file: str = "") -> None:
        """
        Add manifold assembly data from Excel files to the database.
        Args:
            status_file (str): Path to the FMS status overview Excel file.
        """
        if not status_file:
            status_file = self.fms_status_path
        self.lpt_sql.add_manifold_assembly_data(assembly_file=status_file)

    def add_fr_test_data(self, anode_fr_path: str = "", cathode_fr_path: str = "") -> None:
        """
        Add flow restrictor test data from Excel files to the database.
        Args:
            anode_fr_path (str): Path to the anode FR test results Excel file.
            cathode_fr_path (str): Path to the cathode FR test results Excel file.
        """
        if not anode_fr_path and not cathode_fr_path:
            anode_paths = [self.anode_fr_path, r"\\be.local\Doc\DocWork\20025 - CHEOPS2 Low Power\70 - Testing\LP FMS FR Testing - 8\XXX - FR Testing - Anode.xlsx"]
            cathode_paths = [self.cathode_fr_path, r"\\be.local\Doc\DocWork\20025 - CHEOPS2 Low Power\70 - Testing\LP FMS FR Testing - 8\XXX - FR Testing - Cathode.xlsx"]
            
            for anode_path, cathode_path in zip(anode_paths, cathode_paths):
                self.fr_sql.update_fr_test_results(self.excel_extraction, anode_path=anode_path, cathode_path=cathode_path)
        else:
            self.fr_sql.update_fr_test_results(self.excel_extraction, anode_path=anode_fr_path, cathode_path=cathode_fr_path)

    def flow_restrictor_testing(self) -> None:
        """
        Show flow restrictor testing input field.
        """
        self.fr_sql.flow_test_inputs()

    def listen_to_fms_main_results(self, data_folder: str = "") -> None:
        """
        Start listening for new FMS main results.
        This method initializes a thread to monitor the FMS main results
        directory for new or modified files.
        Args:
            data_folder (str): Path to the directory containing FMS main test results.
        """
        if not data_folder:
            data_folder = os.path.join(self.absolute_data_dir, "FMS_data")
        self.fms_main_results_thread = threading.Thread(target=self.fms_sql.listen_to_fms_main_results, args=(data_folder,), daemon=False)
        self.fms_main_results_thread.start()

    def estimate_amazon_invoice_total_cost(self, files: list, s3_rate_per_gb: float = 0.023, textract_rate_per_1000: float = 1.50) -> float:
        """
        Estimate the total cost of processing PDF files using Amazon S3 and Textract.
        Args:
            files (list): List of PDF file paths to be processed.
            s3_rate_per_gb (float): Cost rate for S3 storage per GB. Default is $0.023.
            textract_rate_per_1000 (float): Cost rate for Textract per 1000 pages. Default is $1.50.
        Returns:
            float: Estimated total cost for processing the files.
        """
        total_s3_cost = 0
        total_textract_cost = 0

        for idx, pdf_file in enumerate(files):
            # --- File size in GB for S3 cost ---
            size_bytes = os.path.getsize(pdf_file)
            size_gb = size_bytes / (1024**3)
            total_s3_cost += size_gb * s3_rate_per_gb

            # --- Page count for Textract cost ---
            doc = fitz.open(pdf_file)
            pages = doc.page_count
            total_textract_cost += (pages / 1000) * textract_rate_per_1000

        return total_s3_cost + total_textract_cost

    def get_all_certifications(self, local_certifications: str = "") -> None:
        """
        Process all certification documents and update the database accordingly.
        This method scans the certifications directory for PDF files,
        extracts relevant information using Textract, and updates the
        database with the extracted data.
        Args:
            local_certifications (str): Path to the local directory containing certifications
        """
        if not local_certifications:
            local_certifications = self.default_certifications_path

        relevant_amazon_certifications = [
            "C24-0766","C24-0767","C25-0033","C25-0040","C25-0044",
            "C25-0088","C25-0110","C25-0115","C25-0135","C25-0134",
            "C25-0138","C25-0172","C25-0554","C25-0624","C25-0262",
            "C25-0226","C25-0260","C25-0227","C25-0352","C25-0352",
            "C25-0353","C25-0380","C25-0164", "C25-0165", "C25-0081",
            "C25-0082", "C25-0638", "C25-0637", "C25-0555", "C25-0486",
            "C25-0448", "C25-0380", "C25-0353", "C25-0352", "C25-0260","C25-0226",
            "C25-0178", "C25-0343","C25-0670", "C25-0672", "C25-0686", "C25-0763", "C25-0799",
            "C25-0798", "C25-0939", "C25-0978", "C25-1041"
        ]

        restrictor_certifications = [
            "C25-0053","C25-0054","C25-0055","C25-0332","C25-0333","C24-0089","C24-0090",
            "C24-0370","C24-0371","C24-0204","C25-0412","C25-0413", "C25-1033", "C25-1034"
        ]

        outlet_certifications = [
            "C25-0066", "C25-0046"
        ]

        hpiv_certifications = [
            "C25-0016", "C25-0143", "C25-0347", "C24-0724", "C24-0192"
        ]

        lpt_assembly_cert = ["C25-0261", "C25-0259", "C25-0487", "C25-0470", "C25-0699", "C25-1046"]

        lpt_cert = ["C24-0111", "C24-0112", "C25-0154"]

        restrictor_ocr = load_from_json("restrictor_ocr_certifications")
        lpt_ocr = load_from_json("lpt_ocr_certifications")
        manifold_ocr = load_from_json("manifold_ocr_certifications")
        outlet_ocr = load_from_json("outlet_ocr_certifications")
        filter_ocr = load_from_json("filter_ocr_certifications")
        amazon_certs = load_from_json("certifications_text")

        amazon_keys = [k.split("/")[-1] for k in amazon_certs.keys()]
        total_processed_certs = list(set(amazon_keys + list(restrictor_ocr.keys()) + list(lpt_ocr.keys()) + \
                                         list(manifold_ocr.keys()) + list(outlet_ocr.keys()) + list(filter_ocr.keys())))
        #Test C25-0033
        cert_files = []

        for folder in [self.certification_folder, local_certifications]:
            cert_files.extend([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith('.pdf')
            ])

        cert_files = [f for f in cert_files if any(company in f.lower() for company in self.companies)]

        unique_files = {}
        for f in cert_files:
            base = os.path.basename(f)
            if base not in unique_files or f.startswith(local_certifications):
                unique_files[base] = f

        self.all_current_certifications = list(unique_files.values())
        # self.all_current_certifications = [f for f in self.all_current_certifications if "C25-0763" in f]
        # print(self.all_current_certifications)
        # total_cost = self.estimate_amazon_invoice_total_cost(self.all_current_certifications)
        # print(total_cost): $6.91
        # Check manifold!

        # Initialize progress bar for all files
        with tqdm(total=len(self.all_current_certifications), desc="Processing PDFs") as pbar:
            for file in self.all_current_certifications:
                match = re.search(r'C\d{2}-\d{4}', os.path.basename(file))
                for obj in self.obj_list:
                    obj.pdf_file = file

                certification = match.group(0) if match else ""

                tqdm.write(f"Processing: {certification}")
                company = next((c for c in self.companies if c in file.lower()), None)

                if not any(certification in cert for cert in total_processed_certs) and not certification in \
                    ["C25-0939", "C25-0799", "C25-0798", "C25-0699", "C25-0670", "C25-0672", "C25-0686", "C24-0187", \
                     "C25-0146", "C25-0156", "C25-0087", "C25-0036", "C25-0763", "C25-0978"]:
                    tqdm.write(f"Skipping: {certification}")
                    continue

                if certification == "C24-0187":
                    self.fr_data.extracted_fr_parts = {
                        'restrictor outlet':
                        {
                            "amount": 19,
                            "drawing": "20025.10.21-R5",
                            "certification": certification
                        }
                    }
                    self.fr_data.certification = certification
                    self.fr_sql.update_fr_certification(self.fr_data)
                    self.fr_data.extracted_fr_parts = {}
                    self.fr_data.certification = None
                elif certification == "C25-0146":
                    self.lpt_data.extracted_manifold_parts = {
                        "lpt assembly": {"8": {
                            "certification": certification,
                            "ratio": 13,
                            "drawing": "20025.10.AB-R4"
                        }}
                    }
                    self.lpt_data.certification = certification
                    self.lpt_sql.update_manifold_certification(self.lpt_data)
                    self.lpt_data.extracted_manifold_parts = {}
                    self.lpt_data.certification = None
                elif certification == "C25-0156":
                    self.lpt_data.extracted_manifold_parts = {
                        "lpt assembly": {"9": {
                            "certification": certification,
                            "ratio": 13,
                            "drawing": "20025.10.AB-R4"
                        },
                        "10": {
                            "certification": certification,
                            "ratio": 13,
                            "drawing": "20025.10.AB-R4"
                        }
                    }
                    }
                    self.lpt_data.certification = certification
                    self.lpt_sql.update_manifold_certification(self.lpt_data)
                    self.lpt_data.extracted_manifold_parts = {}
                    self.lpt_data.certification = None

                # elif certification == 'C25-0066':
                #     self.fr_data.extracted_fr_parts = {'restrictor outlet': {'amount': 30, 'certification': 'C25-0066', 'drawing': '20025.10.21-R5'}}
                #     self.fr_sql.update_fr_certification(self.fr_data)
                #     self.fr_data.extracted_fr_parts = {}

                elif certification == "C25-0087":
                    self.lpt_data.extracted_manifold_parts = {
                        "manifold": {
                            "amount": 45,
                            "drawing": self.default_manifold_drawing,
                            'certification': certification
                        }
                    }
                    self.lpt_sql.update_manifold_certification(self.lpt_data)
                    self.lpt_data.extracted_manifold_parts = {}

                elif certification == "C25-0036":
                    self.lpt_data.extracted_manifold_parts = {
                        "manifold": {
                            "amount": 2,
                            "drawing": self.default_manifold_drawing.replace("R4", "R3"),
                            'certification': certification
                        }
                    }
                    self.lpt_sql.update_manifold_certification(self.lpt_data)
                    self.lpt_data.extracted_manifold_parts = {}

                elif certification in outlet_certifications:
                    reader = TextractReader(pdf_file=file, bucket_folder="Certifications", company=company)
                    total_lines = reader.get_text()
                    self.fr_data.get_certification(total_lines)
                    self.fr_sql.update_fr_certification(self.fr_data)
                    self.fr_data.extracted_fr_parts = {}

                elif certification in lpt_assembly_cert:
                    reader = TextractReader(pdf_file=file, bucket_folder="Certifications", company=company)
                    total_lines = reader.get_text()
                    self.lpt_data.get_assembly_certification(total_lines)
                    self.lpt_sql.update_manifold_certification(self.lpt_data)
                    self.lpt_data.extracted_manifold_parts = {}

                elif certification in lpt_cert:
                    reader = TextractReader(pdf_file=file, bucket_folder="Certifications", company=company)
                    total_lines = reader.get_text()
                    self.lpt_data.get_lpt_certification(total_lines)
                    self.lpt_sql.update_lpt_certification(self.lpt_data)
                    self.lpt_data.extracted_lpt_serials = []

                elif certification in hpiv_certifications:
                    reader = TextractReader(pdf_file=file, bucket_folder="Certifications", company=company)
                    total_lines = reader.get_text()
                    self.hpiv_data.get_certification(total_lines)
                    self.hpiv_sql.update_hpiv_certifications(self.hpiv_data)
                    self.hpiv_data.hpiv_ids = []

                elif certification in restrictor_certifications:
                    reader = TextractReader(pdf_file=file, bucket_folder="Certifications", company=company)
                    total_lines = reader.get_text()
                    self.fr_data.get_certification(total_lines)       
                    self.fr_sql.update_fr_certification(self.fr_data)
                    self.fr_data.extracted_fr_parts = {}             
                    tqdm.write(f"Amazon certification processed: {certification}")

                elif certification in relevant_amazon_certifications:
                    reader = TextractReader(pdf_file=file, bucket_folder="Certifications", company=company)
                    total_lines = reader.get_text()
                    self.tv_data.get_certification(total_lines)
                    self.tv_sql.update_tv_certification(self.tv_data)
                    self.tv_data.extracted_tv_parts = {}
                    tqdm.write(f"Amazon certification processed: {certification}")
                else:
                    # Worker definitions
                    def lpt_worker():
                        if not certification in lpt_assembly_cert:
                            self.lpt_data.get_ocr_certification()
                            self.lpt_data.get_manifold_ocr_certification()
                            if self.lpt_data.extracted_lpt_serials:
                                self.lpt_data.certification = certification
                                self.lpt_sql.update_lpt_certification(self.lpt_data)
                                self.lpt_data.extracted_lpt_serials = []
                                self.lpt_data.certification = None
                                tqdm.write("LPT data found and updated.")
                            elif self.lpt_data.extracted_manifold_parts:
                                self.lpt_data.certification = certification
                                self.lpt_sql.update_manifold_certification(self.lpt_data)
                                self.lpt_data.extracted_manifold_parts = {}
                                self.lpt_data.certification = None
                                tqdm.write("Manifold data found and updated.")

                    # def fr_worker():
                    #     if company in ["sk technology", "sk", "ejay filtration"]:
                    #         self.fr_data.get_ocr_certification()
                    #         if self.fr_data.extracted_fr_parts:
                    #             self.fr_sql.update_fr_certification(self.fr_data)
                    #             self.fr_data.extracted_fr_parts = {}
                    #             tqdm.write("FR data found and updated.")

                    # def fr_outlet_worker():
                    #     if company in ["sk technology", "sk"]:
                    #         self.fr_data.get_outlet_certification()
                    #         if self.fr_data.extracted_fr_parts:
                    #             self.fr_sql.update_fr_certification(self.fr_data)
                    #             self.fr_data.extracted_fr_parts = {}
                    #             tqdm.write("FR outlet data found and updated.")

                    def fr_filter_worker():
                        if company in "ejay filtration":
                            self.fr_data.get_filter_certification()
                            if self.fr_data.extracted_fr_parts:
                                self.fr_data.certification = certification
                                self.fr_sql.update_fr_certification(self.fr_data)
                                self.fr_data.extracted_fr_parts = {}
                                self.fr_data.certification = None
                                tqdm.write("FR filter data found and updated.")

                    threads = [
                        threading.Thread(target=lpt_worker, daemon=True),
                        # threading.Thread(target=fr_worker, daemon=True),
                        # threading.Thread(target=fr_outlet_worker, daemon=True),
                        threading.Thread(target=fr_filter_worker, daemon=True)
                    ]

                    # Start threads
                    for idx,t in enumerate(threads):
                        tqdm.write(f"Starting thread {idx}")
                        t.start()

                    # Wait for all threads to finish
                    for idx,t in enumerate(threads):
                        t.join()

                # Update the progress bar for this file
                pbar.update(1)

    def add_tv_test_results(self, tv_test_path: str = "") -> None:
        """
        Add TV test results from Excel files to the database.
        This method scans the TV test results directory for Excel files,
        extracts relevant test data, and updates the database with the
        extracted data.
        Args:
            tv_test_path (str): Path to the directory containing TV test runs.
        """
        if not tv_test_path:
            tv_test_path = self.tv_test_path
        session = self.Session()
        for folder in os.listdir(tv_test_path):

            full_folder = os.path.join(tv_test_path, folder)
            if os.path.isdir(full_folder) and "test valve #" in folder.lower():
                # Extract TV number
                tv_id = int(folder.split("#")[-1].strip().split(" ")[0])

                tv_sql = TVLogicSQL(session=session, fms=self)

                # Collect all .xls files in this folder
                test_files = [os.path.join(full_folder, f)
                              for f in os.listdir(full_folder)
                              if f.lower().endswith('.xls') and 'quench' not in f.lower()]

                # Check for subfolders and collect their .xls files
                for subfolder in os.listdir(full_folder):
                    full_subfolder = os.path.join(full_folder, subfolder)
                    if os.path.isdir(full_subfolder):
                        test_files.extend([os.path.join(full_subfolder, f)
                                           for f in os.listdir(full_subfolder)
                                           if f.lower().endswith('.xls') and 'quench' not in f.lower()])

                # Get welded date from certification if available
                cert_check = session.query(TVCertification).filter_by(tv_id=tv_id, part_name=TVParts.WELD.value).first()
                welded_date = cert_check.date if cert_check else None

                # Sort test files by date extracted from filename
                parsed_tests = []
                for test_file in test_files:
                    test_reference = os.path.basename(test_file).split('_LP_')[0]
                    try:
                        test_date = datetime.strptime(test_reference, "%Y_%m_%d_%H-%M-%S").date()
                        parsed_tests.append((test_date, test_file, test_reference))
                    except Exception as e:
                        print(f"Error parsing date from {test_reference}: {str(e)}")

                parsed_tests.sort(key=lambda x: x[0])

                last_pre_weld_opening_temp = None
                last_opening_temp = None

                # If welded_date is None, check status entry to split tests in half if welded=True
                status_entry = session.query(TVStatus).filter_by(tv_id=tv_id).first()
                status_welded = status_entry.welded if status_entry else False
                half_index = None
                if not welded_date and len(parsed_tests) > 1:
                    half_index = len(parsed_tests) // 2

                for idx, (test_date, test_file, test_reference) in enumerate(parsed_tests):
                    test_check = session.query(TVTestRuns).filter_by(tv_id=tv_id, test_reference=test_reference).first()
                    if test_check:
                        last_opening_temp = test_check.opening_temp
                        continue

                    tv_data = TVData(test_results_file=test_file)
                    update = tv_data.extract_tv_test_results_from_excel()
                    if not update:
                        continue

                    if welded_date:
                        welded = test_date >= welded_date.date()
                    else:
                        if half_index is not None and status_welded:
                            welded = idx >= half_index
                        elif status_welded == False:
                            welded = False

                    tv_sql.tv_id = tv_id
                    tv_sql.tv_test_reference = test_reference
                    tv_sql.tv_welded = welded
                    tv_sql.update_tv_test_results(tv_data)

                    if not welded:
                        last_pre_weld_opening_temp = tv_data.opening_temperature
                    last_opening_temp = tv_data.opening_temperature

                if status_entry:
                    if last_opening_temp is not None:
                        status_entry.opening_temp = last_opening_temp
                    if last_pre_weld_opening_temp is not None:
                        status_entry.pre_weld_opening_temp = last_pre_weld_opening_temp

        session.commit()

    def add_hpiv_data(self, hpiv_data_packages: str = "", output_folder: str = "") -> None:
        """
        Add HPIV data from PDF files to the database.
        This method scans the HPIV data packages directory for PDF files,
        extracts relevant HPIV data, and updates the database with the
        extracted data.
        Args:
            hpiv_data_packages (str): Path to the directory containing HPIV data packages.
            output_folder (str): Path to the output folder for the extracted HPIV reports.
        """
        if not hpiv_data_packages:
            hpiv_data_packages = os.path.join(self.absolute_data_dir, "HPIV_data_packages")
        data_packages = [os.path.join(hpiv_data_packages, f) for f in os.listdir(hpiv_data_packages) if f.lower().endswith('.pdf')]
        if not output_folder:
            output_folder = os.path.join(self.absolute_data_dir, "extracted_HPIV_reports")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        for package in data_packages:
            hpiv_data = HPIVData(pdf_file=package) 
            hpiv_data.extract_hpiv_data(output_folder=output_folder)
            self.hpiv_sql.update_hpiv_characteristics(hpiv_data)
            self.hpiv_sql.update_hpiv_revisions(hpiv_data)

    def add_lpt_calibration_data(self, lpt_path: str = "") -> None:
        """
        Add LPT calibration data from JSON files to the database.
        This method scans the LPT calibration directory for JSON files,
        extracts relevant calibration data, and updates the database with the
        extracted data.
        
        Args:
            lpt_path (str): Path to the directory containing LPT calibration data.
        """
        if not lpt_path:
            lpt_path = os.path.join(self.absolute_data_dir, "LPT_data/LPT_coefficient_data")
        json_files = []
        for root, dirs, files in os.walk(lpt_path):
            for f in files:
                if f.lower().endswith('.json'):
                    json_files.append(os.path.join(root, f))
        self.lpt_data.json_files = json_files
        self.lpt_data.extract_coefficients_from_json()
        self.lpt_sql.update_lpt_calibration(self.lpt_data)
        self.lpt_data.lpt_coefficients = {}
        self.lpt_data.lpt_calibration = {}
        self.lpt_data.json_files = []

    def add_fms_main_test_data(self, fms_main_files: str =  "",
        fms_status_path: str = "") -> None:
        """
        Add FMS main test data from the test reports to the database.
        This method scans the FMS main test results directory for PDF files,
        extracts relevant test data, and updates the database with the
        extracted data.
        Args:
            fms_main_files (str): Path to the directory containing FMS main test results.
            fms_status_path (str): Path to the FMS status overview Excel file.
        """
        if not fms_main_files:
            fms_main_files = os.path.join(self.absolute_data_dir, "FMS_data/test_results")
        if not fms_status_path:
            fms_status_path = self.fms_status_path
    
        main_files = [os.path.join(fms_main_files, f) for f in os.listdir(fms_main_files) if f.lower().endswith('.pdf')]
        for main in main_files:
            fms_data = FMSData(pdf_file=main, status_file=fms_status_path)
            fms_data.extract_FMS_test_results()
            self.fms_sql.add_fms_assembly_data(fms_data)
            self.fms_sql.update_fms_main_test_results(fms_data)

    def add_fms_functional_test_data(self, test_path: str = "", fms_ids: list[str] = []) -> None:
        """
        Add FMS main functional test data from the test reports to the database.
        This method scans the FMS test results directory for Excel files,
        extracts relevant functional test data, and updates the database with the
        extracted data.
        Args:
            test_path (str): Path to the directory containing FMS functional test results.
            fms_id (str): Specific FMS ID to process. If empty, process all.
        """
        slope_files = {}
        closed_loop_files = {}
        fr_files = {}
        tvac_files = {}
        open_loop_files = {}

        if not test_path:
            test_path = self.test_path
        
        serials = fms_ids if fms_ids else ["24-100", "24-101", "24-102", "24-188", "24-189", "24-190"] + [f"25-{i:03d}" for i in range(45, 65)]
        # serials = ["25-062"]

        # Helper function to find first match in folder
        def first_match(path: str, keyword: str, extension: str = None, folder_only: bool = False) -> str | None:
            if path is None or not os.path.exists(path):
                return None
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                if folder_only and not os.path.isdir(full_path):
                    continue
                if keyword.lower() in entry.lower():
                    if extension and not entry.lower().endswith(extension.lower()):
                        continue
                    return full_path
            return None

        for f in os.listdir(test_path):
            full_f_path = os.path.join(test_path, f)
            if not os.path.isdir(full_f_path):
                continue
            serial = next((s for s in serials if s in f), None)
            if not serial:
                continue

            slope_files[serial] = []
            closed_loop_files[serial] = []
            fr_files[serial] = []
            tvac_files[serial] = []
            open_loop_files[serial] = []
            # Functional folders
            functional_folder = first_match(full_f_path, "function", folder_only=True)
            low_folder = first_match(functional_folder, "10 bar", folder_only=True)
            high_folder = first_match(functional_folder, "190 bar", folder_only=True)

            # Helper to add files
            def add_files(target_list, folder, keywords, ext=".xls"):
                for kw in keywords:
                    file_path = first_match(folder, kw, extension=ext)
                    if file_path:
                        target_list.append(file_path)

            # Add low/high functional files
            add_files(slope_files[serial], low_folder, ["slope"])
            add_files(closed_loop_files[serial], low_folder, ["closed loop"])
            add_files(fr_files[serial], low_folder, ["fr", "characteristics", "fr_test"])
            add_files(slope_files[serial], high_folder, ["slope"])
            add_files(closed_loop_files[serial], high_folder, ["closed loop"])
            add_files(fr_files[serial], high_folder, ["fr", "characteristics", "fr_test"])

            # TVAC folders
            tvac_folder = first_match(full_f_path, "tvac", folder_only=True)
            if not tvac_folder:
                continue

            tvac_cycle_folder = first_match(tvac_folder, "cycl", folder_only=True)
            if tvac_cycle_folder:
                tvac_files[serial] = [os.path.join(tvac_cycle_folder, i) 
                                    for i in os.listdir(tvac_cycle_folder) if i.lower().endswith(".csv")]

            # Temperature folders
            temp_conditions = ["-15 degC", "22 degC", "70 degC"]
            temp_map = {}
            for temp in temp_conditions:
                folder = first_match(tvac_folder, temp, folder_only=True)
                if folder:
                    function_folder = first_match(folder, "function", folder_only=True)
                    temp_map[temp] = function_folder

            pressures = ["10 bar", "190 bar"]

            # Add TVAC slope & closed loop files for each temp & pressure
            for temp, func_folder in temp_map.items():
                if not func_folder:
                    continue
                for pressure in pressures:
                    pressure_folder = first_match(func_folder, pressure, folder_only=True)
                    if not pressure_folder:
                        continue
                    add_files(slope_files[serial], pressure_folder, ["slope"])
                    add_files(closed_loop_files[serial], pressure_folder, ["closed loop"])
                    add_files(open_loop_files[serial], pressure_folder, ["open loop"])

        session = self.Session()
        for serial in serials:
            fms_check = session.query(FMSMain).filter_by(fms_id=serial).first()
            if fms_check:
                gas_type = fms_check.gas_type
            else:
                gas_type = "Xe"
            slope = slope_files.get(serial, [])
            closed_loop = closed_loop_files.get(serial, [])
            fr = fr_files.get(serial, [])
            tvac = tvac_files.get(serial, [])
            fms_sql = FMSLogicSQL(session=session, fms=self)
            fms_sql.selected_fms_id = serial
            for slope_test in slope:
                fms_data = FMSData(test_type="slope", flow_test_file=slope_test)
                fms_data.extract_slope_data()
                fms_data.gas_type = gas_type
                fms_sql.update_flow_test_results(fms_data)

            for open_loop_test in open_loop_files.get(serial, []):
                fms_data = FMSData(test_type="open_loop", flow_test_file=open_loop_test)
                fms_data.extract_slope_data()
                fms_data.gas_type = gas_type
                fms_sql.update_flow_test_results(fms_data)

            for closed_loop_test in closed_loop:
                fms_data = FMSData(test_type="closed_loop", flow_test_file=closed_loop_test)
                fms_data.extract_slope_data()
                fms_data.gas_type = gas_type
                fms_sql.update_flow_test_results(fms_data)

            for fr_test in fr:
                fms_data = FMSData(test_type="fr_characteristics", flow_test_file=fr_test)
                fms_data.extract_slope_data()
                fms_data.gas_type = gas_type
                fms_sql.update_fr_characteristics_results(fms_data)

            if tvac:
                print(serial)
                fms_data = FMSData(test_type="tvac_cycle", flow_test_file=None)
                fms_data.csv_files = tvac
                fms_data.extract_tvac_from_csv()
                fms_data.gas_type = gas_type
                fms_sql.update_tvac_cycle_results(fms_data)   

    def save_procedure(self, procedure_name: str = None, procedure_json: dict = None, script_version: str = None, report_version: str = None) -> None:
        """
        Save or update a procedure JSON script in the database.
        Automatically increments the version in #.# format (e.g., 1.9 -> 2.0).

        Args:
            procedure_name (str): Name of the procedure script.
            procedure_json (dict): JSON content of the procedure script.
            script_version (str, optional): Initial version of the procedure script. Defaults to None.
            report_version (str, optional): Report version associated with the procedure. Defaults to None.
        """
        session = self.Session()
        try:
            if procedure_name and procedure_json:
                existing_script = session.query(FMSProcedures).filter_by(script_name=procedure_name).first()
                if existing_script:
                    # Increment version automatically
                    current_version = existing_script.version
                    major, minor = map(int, current_version.split('.'))
                    minor += 1
                    if minor >= 10:
                        major += 1
                        minor = 0
                    new_version = f"{major}.{minor}"
                    
                    existing_script.script = procedure_json
                    existing_script.version = new_version
                    if report_version:
                        existing_script.report_version = report_version
                else:
                    new_script = FMSProcedures(
                        script_name=procedure_name,
                        version=script_version,
                        script=procedure_json,
                        report_version=report_version
                    )   
                    session.add(new_script)
                session.commit()
        except Exception as e:
            print(f"Error updating JSON script: {str(e)}")
            traceback.print_exc()
        finally:
            session.close()

    def load_procedure(self, procedure_name: str, version: str = None) -> dict:
        """
        Load a procedure JSON script from the database.
        Args:
            procedure_name (str): Name of the procedure script.
            version (str, optional): Specific version of the procedure script to load. Defaults to None.
        Returns:
            dict: JSON content of the procedure script.
        """
        session = self.Session()
        try:
            script_entry = session.query(FMSProcedures).filter_by(script_name=procedure_name).first() \
                if version is None else session.query(FMSProcedures).filter_by(script_name=procedure_name, version=version).first()
            if script_entry:
                return script_entry.script
            else:
                print(f"No procedure found with name: {procedure_name} and version: {version}")
                return None
        except Exception as e:
            print(f"Error loading procedure {procedure_name}: {str(e)}")
            traceback.print_exc()
        finally:
            session.close()
            fall_back = load_from_json(procedure_name)
            return fall_back

    def get_all_current_data(self,
            local_certifications: str = "",
            tv_assembly: str = "",
            tv_summary: str = "",
            status_file: str = "",
            electrical_data: str = "",
            tv_test_path: str = "",
            anode_fr_path: str = "",
            cathode_fr_path: str = "",
            hpiv_data_packages: str = "",
            lpt_path: str = "",
            fms_main_files: str = "",
            test_path: str = ""
        ) -> None:
        """
        Retrieve and process all current local data from the relevant sources,
        and add the processed data to the database.
        Args:
            local_certifications (str): Path to local certifications folder.
            tv_assembly (str): Path to TV assembly data folder.
            tv_summary (str): Path to TV summary Excel file.
            status_file (str): Path to FMS status overview Excel file.
            electrical_data (str): Path to TV electrical data folder.
            tv_test_path (str): Path to TV test results folder.
            anode_fr_path (str): Path to anode FR data folder.
            cathode_fr_path (str): Path to cathode FR data folder.
            hpiv_data_packages (str): Path to HPIV data packages folder.
            lpt_path (str): Path to LPT calibration data folder.
            fms_main_files (str): Path to FMS main test results folder.
            test_path (str): Path to FMS functional test results folder.
        """
        print("Getting Certifications")
        self.get_all_certifications(local_certifications=local_certifications)
        print("Getting TV Assembly Data")
        self.add_tv_assembly_data(tv_assembly=tv_assembly, tv_summary=tv_summary, status_file=status_file)
        print("Getting TV Electrical Data")
        self.add_tv_electrical_data(electrical_data=electrical_data)
        print("Getting TV Test Results")
        self.add_tv_test_results(tv_test_path=tv_test_path)
        print("Getting FR Test Data")
        self.add_fr_test_data(anode_fr_path=anode_fr_path, cathode_fr_path=cathode_fr_path)
        print("Getting HPIV Data")
        self.add_hpiv_data(hpiv_data_packages=hpiv_data_packages)
        print("Getting LPT Calibration Data")
        self.add_lpt_calibration_data(lpt_path=lpt_path)
        print("Getting Manifold Assembly Data")
        self.add_manifold_assembly_data(status_file=status_file)
        print("Getting FMS Main Test Data")
        self.add_fms_main_test_data(fms_main_files=fms_main_files, fms_status_path=status_file)
        print("Getting FMS Functional Test Data")
        self.add_fms_functional_test_data(test_path=test_path)

    def update_procedure(self, procedure_name: str = None, version: str = None, report_version: str = None) -> None:
        """
        Update a procedure JSON script in the database by reloading it from the local JSON file.
        Args:
            procedure_name (str): Name of the procedure script.
            version (str, optional): Specific version of the procedure script to update. Defaults to None.
            report_version (str, optional): Report version associated with the procedure. Defaults to None.
        """
        procedure = load_from_json(procedure_name)
        self.save_procedure(procedure_name=procedure_name, procedure_json=procedure, script_version=version, report_version=report_version)

    def print_table(self, table_class: object, limit: int = None) -> None:
        """
        Print the contents of a database table to the console.
        Args:
            table_class (object): SQLAlchemy ORM class representing the table.
            limit (int, optional): Maximum number of records to print. Defaults to None (print all).
        """
        session = None
        return
        try:
            session = self.Session()
            query = session.query(table_class)

            # Try to order by primary key column if available
            pk_columns = [col.name for col in table_class.__table__.primary_key.columns]
            if limit is not None and pk_columns:
                query = query.order_by(getattr(table_class, pk_columns[0]).desc()).limit(limit)
            elif limit is not None:
                query = query.limit(limit)

            records = query.all()
            if not records:
                print(f"No records found in table {table_class.__tablename__}")
                return

            print(f"\n--- {table_class.__tablename__} ({len(records)} records) ---")
            for record in records:
                attrs = vars(record)
                attrs = {k: v for k, v in attrs.items() if not k.startswith('_')}

                # Exclude specific fields for LPTCalibration
                if table_class.__tablename__ == "lpt_calibration":
                    excluded_fields = {"p_calculated", "temp_calculated", "signal", "resistance"}
                    attrs = {k: v for k, v in attrs.items() if k not in excluded_fields}

                print(", ".join(f"{k}: {v}" for k, v in attrs.items()))
            print(f"--- End of {table_class.__tablename__} ---\n")

        except Exception as e:
            print(f"Error printing table {table_class.__tablename__}: {str(e)}")
            traceback.print_exc()
        finally:
            if session:
                session.close()

if __name__ == "__main__":
    fms = FMSDataStructure(excel_extraction=True)
    # wb = openpyxl.load_workbook(fms.cathode_fr_path)
    # print(wb.sheetnames)
    fms.get_all_current_data()
    # fms.fms_sql.update_limit_database()
    # fms.print_table(AnodeFR, limit=5)
    # fms.print_table(CathodeFR, limit=5)
    # fms.print_table(FMSTestResults)

    # fms.update_procedure(procedure_name="tv_assembly_steps")
    # session = fms.Session()
    # jsons = session.query(FMSProcedures).filter_by(script_name="tv_assembly_steps").all()
    # for js in jsons:
    #     script = js.script
    #     print(script["72"])

    # fms.listen_to_fms_main_results()
    """
    conda activate base
    python -u "C:/Users/TANTENS/Documents/FMS_data_structure/fms_data_structure.py"
    """