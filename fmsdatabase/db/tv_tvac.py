from sqlalchemy import Column, Integer, String, JSON, ForeignKey, Enum, DateTime
from sqlalchemy.orm import relationship
from .base import Base

class TVTvac(Base):
    """
    -------------------
    TV TVAC table 1.6.3
    -------------------

    Columns
    -------
    test_id : String(50)
        Primary Key. TVAC Test Identifier.
    tv_id : Integer
        Foreign Key. TV Identifier linking to TVStatus table.
    time : JSON
        List of time measurements during the TVAC test.
    outlet_elbow : JSON
        List of temperatures of the outlet elbow during the TVAC test.
    outlet_temp_1 : JSON
        List of outlet temperature 1 measurements during the TVAC test.
    outlet_temp_2 : JSON
        List of outlet temperature 2 measurements during the TVAC test.
    interface_temp : JSON
        List of interface temperature measurements during the TVAC test.
    if_plate : JSON
        List of IF plate temperature measurements during the TVAC test.
    if_plate_1 : JSON
        List of the first IF plate temperature measurements during the TVAC test.
    if_plate_2 : JSON
        List of the second IF plate temperature measurements during the TVAC test.
    tv_voltage : JSON
        List of TV voltage measurements during the TVAC test.
    tv_current : JSON
        List of TV current measurements during the TVAC test.
    vacuum : JSON
        List of vacuum measurements during the TVAC test.
    cycles : Integer
        Number of cycles completed during the TVAC test.

    Relationships
    -------------
    status : TVStatus
        Many-to-one relationship with TVStatus table.
    """
    __tablename__ = 'tv_tvac'

    test_id = Column(String(50), primary_key=True, nullable=False)
    tv_id = Column(Integer, ForeignKey('tv_status.tv_id'), nullable=False)
    time = Column(JSON, nullable=True)
    outlet_elbow = Column(JSON, nullable=True)
    outlet_temp_1 = Column(JSON, nullable=True)
    outlet_temp_2 = Column(JSON, nullable=True)
    interface_temp = Column(JSON, nullable=True)
    if_plate = Column(JSON, nullable=True)
    if_plate_1 = Column(JSON, nullable = True)
    if_plate_2 = Column(JSON, nullable = True)
    tv_voltage = Column(JSON, nullable=True)
    tv_current = Column(JSON, nullable=True)  
    vacuum = Column(JSON, nullable=True)
    cycles = Column(Integer, nullable=True)

    status = relationship("TVStatus", back_populates="tvac")
