from sqlalchemy import Column, Integer, String, JSON, ForeignKey, Float, Enum, DateTime
from ..utils.general_utils import FunctionalTestType
from sqlalchemy.orm import relationship
from .base import Base

class FMSFunctionalTests(Base):
    """
    ------------------------------
    FMS Functional Tests Table 1.3
    ------------------------------
    Columns
    -------
    id : Integer
        Primary Key, unique Test ID.
    fms_id : String
        Foreign Key, FMS ID (references the main FMS table).
    test_id : String
        Test Identifier.
    test_type : Enum(FunctionalTestType)
        Type of functional test conducted.
    date : DateTime
        Date of the test.
    remark : String
        Additional remarks about the test.
    temp_type : Enum(FunctionalTestType)
        Type of temperature measurement used.
    trp_temp : Float
        TRP temperature of the FMS during the test, in °C.
    inlet_pressure : Float
        Inlet pressure of the FMS during the test, in bar.
    outlet_pressure : Float
        Outlet pressure of the FMS during the test, in bar.
    gas_type : String
        Type of gas used during the test.
    slope12 : Float
        Slope of the TV flow between 1 and 2 mg/s flow rate.
    slope24 : Float
        Slope of the TV flow between 2 and 4 mg/s flow rate.
    intercept12 : Float
        Intercept of the TV flow between 1 and 2 mg/s flow rate.
    intercept24 : Float
        Intercept of the TV flow between 2 and 4 mg/s flow rate.
    response_times : JSON
        Dictionary storing response times for the different pressure set points.

    Relationships
    -------------
    fms_main : relationship
        Many-to-one relationship with the FMSMain table.
    functional_results : relationship
        One-to-many relationship with the FMSFunctionalResults table.
    """

    __tablename__ = 'fms_functional' 
    id = Column(Integer, primary_key=True, autoincrement=True)
    fms_id = Column(String(50), ForeignKey('fms_main.fms_id'), nullable=False)
    test_id = Column(String(50), nullable=False)
    test_type = Column(Enum(FunctionalTestType, native_enum=False), nullable=False)
    date = Column(DateTime, nullable=False)
    remark = Column(String(255), nullable=True)
    temp_type = Column(Enum(FunctionalTestType, native_enum=False), nullable=True)
    trp_temp = Column(Float, nullable=True)
    inlet_pressure = Column(Float, nullable=True)
    outlet_pressure = Column(Float, nullable=True)
    gas_type = Column(String(50), nullable=True)
    slope12 = Column(Float, nullable=True)
    slope24 = Column(Float, nullable=True)
    intercept12 = Column(Float, nullable=True)
    intercept24 = Column(Float, nullable=True)
    response_times = Column(JSON, nullable=True)
    response_regions = Column(JSON, nullable=True)
    slope_correction = Column(Float, nullable=True)

    fms_main = relationship("FMSMain", back_populates="functional_tests")
    functional_results = relationship("FMSFunctionalResults", back_populates="main_tests")
