
from __future__ import annotations
from typing import TYPE_CHECKING, Any

# Third Party Imports
from sqlalchemy import func
from sqlalchemy.orm import Session
from IPython.display import display
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
import numpy as np
from collections import defaultdict
import traceback
from datetime import datetime
import io

# Local Imports
from .general_utils import (
    display_df_in_chunks,
    find_intersections,
    get_slope, 
    FMSParts, 
    FunctionalTestType, 
    LimitStatus,
    FMSFlowTestParameters, 
    FMSMainParameters
)

from .tv_query import TVQuery
from .manifold_query import ManifoldQuery
from .hpiv_query import HPIVQuery

from ..db import (
    FMSMain,
    FMSFRTests, 
    FMSTestResults, 
    FMSFunctionalTests, 
    FMSTvac, 
    FMSFunctionalResults, 
    ManifoldStatus
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session



class FMSQuery:
    """
    Handles queries and visualizations for test/assembly data stored in the FMS database.

    Attributes
    ----------
    # Database sessions
    session : SQLAlchemy session
        Session for database interaction.
    
    # Enums
    fms_parts : FMSParts
    fms_progress_status : FMSProgressStatus
    flow_test_parameters : FMSFlowTestParameters
    fms_main_parameters : FMSMainParameters
    limit_status : LimitStatus
    functional_test_type : FunctionalTestType
    fr_status : FRStatus
    tv_parts : TVParts
    tv_test_parameters : TVTestParameters
    tv_status : TVStatus

    # Test specifications
    lpt_pressures : list[float]
    lpt_voltages : list[float]
    min_flow_rates : list[float]
    max_flow_rates : list[float]
    range12_low : list[float]
    range24_low : list[float]
    range12_high : list[float]
    range24_high : list[float]
    initial_flow_rate : float
    lpt_set_points : list[float]
    max_opening_response : float
    max_response : float

    # Query Classes
    tv_query : TVQuery
    manifold_query : ManifoldQuery
    hpiv_query : HPIVQuery

    Methods
    -------
    load_all_tests(fms_id)
        Load all test lists for the given FMS ID.
    get_open_loop_tests()
        Return functional tests corresponding to open loop tests.
    get_slope_tests()
        Return functional tests corresponding to slope tests.
    get_closed_loop_tests()
        Return functional tests corresponding to closed loop tests.
    get_fr_tests()
        Return FR characteristic tests.
    get_tvac_tests()
        Return TVAC cycle tests.
    fms_query_field()
        Create a neat form layout for FMS query selection.
    fms_status()
        Display the status of the current FMS entry.
    fms_test_remark_field(test_run, look_up_table)
        Create a clean input field for FMS test remarks.
    fr_test_query(test_id, plot)
        Query and plot FR characteristic test data.
    plot_fr_characteristics(gas_type, serial, plot)
        Plot FR characteristics.
    plot_fr_voltage(title, gas_type)
        Plot FR characteristics vs LPT voltage.
    open_loop_test_query(test_id, test_type, plot)
        Query and plot open loop test data.
    get_flow_power_slope(flows, powers, num_points)
        Calculate flow-power slopes for TV analysis.
    check_tv_slope(slope12, slope24, intercept12, intercept24)
        Check TV slope against specifications.
    plot_open_loop(serial, gas_type, plot)
        Plot open loop test data (including slope tests).
    closed_loop_test_query(test_id, plot)
        Query and plot closed loop test data.
    tvac_cycle_query(test_id, plot)
        Query and plot TVAC cycle test data.
    fms_characteristics_query()
        Query and present FMS characteristics (acceptance test results).
    fms_comparison_query(analysis_type)
        Perform FMS trend analysis based on selected type.
    free_fms_analysis(all_main_results)
        Compare FMS characteristics freely to find trends.
    plot_characteristics(parameter1, parameter2, all_main_results)
        Plot characteristics between two parameters.
    flow_rate_analysis(all_fr_tests)
        Analyze flow rate trends across all FR tests vs spec.
    tv_slope_analysis(grouped_functional_tests)
        Analyze TV slope trends across all functional tests vs spec.
    part_investigation_query(part_name)
        Investigate parts of the FMS (Manifold, TV, or HPIV).
    plot_closed_loop(serial, gas_type, plot)
        Plot closed loop test data.
    plot_tv_closed_loop(title)
        Plot TV data of the closed loop test.
    """

    def __init__(self, session: "Session", lpt_pressures = [0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.4], max_opening_response: float = 300, max_response: float = 60,
                 lpt_voltages: list[float] = [10, 15, 17, 20, 24, 25, 30, 35], min_flow_rates: list[float] = [0.61, 1.23, 1.51, 1.85, 2.40, 2.43, 3.13, 3.72], 
                 max_flow_rates: list[float] = [0.96, 1.61, 1.9, 2.34, 2.93, 3.07, 3.81, 4.54], range12_low: list[float] = [13, 41], range24_low: list[float] = [19, 54],
                 range12_high: list[float] = [25, 95], range24_high: list[float] = [35, 140], initial_flow_rate: float = 0.1, lpt_set_points: list[float] = [1, 1.625, 2.25, 1.625, 1, 0.2]):
        
        self.session = session
        self.lpt_pressures: list[float] = lpt_pressures
        self.lpt_voltages: list[float] = lpt_voltages
        self.min_flow_rates: list[float] = min_flow_rates
        self.max_flow_rates: list[float] = max_flow_rates
        self.range12_low: list[int] = range12_low
        self.range24_low: list[int] = range24_low
        self.range12_high: list[int] = range12_high
        self.range24_high: list[int] = range24_high
        self.fms_entry: None | FMSMain = None
        self.fms_id: None | int = None
        self.functional_tests_list: list = []
        self.fr_tests_list: list = []
        self.tvac_tests_list: list = []
        self.initial_flow_rate = initial_flow_rate
        self.lpt_set_points = lpt_set_points
        self.max_opening_response = max_opening_response
        self.max_response = max_response

    def load_all_tests(self, fms_id):
        """
        Load all test lists for the given FMS ID and store them as attributes.
        """
        self.fms_entry = self.session.query(FMSMain).filter_by(fms_id=fms_id).first()
        if not fms_id or not self.fms_entry:
            self.functional_tests_list = []
            self.fr_tests_list = []
            self.tvac_tests_list = []
            return
        self.functional_tests_list: list[FMSFunctionalTests] = self.fms_entry.functional_tests
        self.fr_tests_list: list[FMSFRTests] = self.fms_entry.fr_tests
        self.tvac_tests_list: list[FMSTvac] = self.fms_entry.tvac_results

    def get_open_loop_tests(self) -> list[FMSFunctionalTests]:
        """
        Return filtered functional tests corresponding to open loop tests.
        """
        return [i for i in getattr(self, "functional_tests_list", [])
                if i.test_type in (FunctionalTestType.HIGH_OPEN_LOOP,
                                FunctionalTestType.LOW_OPEN_LOOP)]
    
    def get_slope_tests(self) -> list[FMSFunctionalTests]:
        """
        Return filtered functional tests corresponding to slope tests.
        """
        return [i for i in getattr(self, "functional_tests_list", [])
                if i.test_type in (FunctionalTestType.HIGH_SLOPE,
                                FunctionalTestType.LOW_SLOPE)]

    def get_closed_loop_tests(self) -> list[FMSFunctionalTests]:
        """
        Return filtered functional tests corresponding to closed loop tests.
        """
        return [i for i in getattr(self, "functional_tests_list", [])
                if i.test_type in (FunctionalTestType.HIGH_CLOSED_LOOP,
                                FunctionalTestType.LOW_CLOSED_LOOP)]

    def get_fr_tests(self) -> list[FMSFRTests]:
        """
        Return filtered FR characteristic tests.
        """
        return getattr(self, "fr_tests_list", [])

    def get_tvac_tests(self) -> list[FMSTvac]:
        """
        Return filtered TVAC cycle tests.
        """
        return getattr(self, "tvac_tests_list", [])

    def fms_query_field(self):
        """
        Create a neat form layout for FMS query selection:
        - FMS ID dropdown
        - Test Type dropdown
        - Dynamic second dropdown that updates based on Test Type selection
        - Runs the corresponding function when a dynamic selection is made
        """
        self.test_dict = {}

        # --- Widgets ---
        fms_ids: list[str] = [f.fms_id for f in self.session.query(FMSMain).all()]
        fms_ids = sorted(fms_ids, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1].zfill(4))))

        fms_field = widgets.Dropdown(
            options=fms_ids,
            description='FMS ID:',
            layout=widgets.Layout(width='350px'),
            style={'description_width': '80px'}
        )

        query_field = widgets.Dropdown(
            options=['Status', 'Acceptance Test', 'Slope Test', 'Open Loop Test', 'Closed Loop Test', 'FR Characteristics',
                    'Tvac Cycle', 'FMS Trend Analysis', 'Part Investigation'],
            description='Query Type:',
            layout=widgets.Layout(width='350px'),
            style={'description_width': '80px'},
            value='Status'
        )

        dynamic_field = widgets.Dropdown(
            options=[],
            description='Select Test:',
            layout=widgets.Layout(width='350px'),
            style={'description_width': '80px'}
        )

        output = widgets.Output()

        # --- Helper to display remark + query together ---
        def display_with_remark(func: callable, test_run: object = None, lookup_table: object = None) -> None:
            with output:
                output.clear_output(wait=True)
                if test_run is not None:
                    self.fms_test_remark_field(test_run, lookup_table)
                func()

        # --- Map test type to callable ---
        def get_dynamic_callable(test_type: str) -> callable:
            if dynamic_field.value is None and test_type != 'Status':
                return lambda: None

            if test_type == 'Status':
                return lambda: display_with_remark(lambda: self.fms_status())
            
            elif test_type == "Acceptance Test":
                return lambda: display_with_remark(lambda: self.fms_characteristics_query())
            
            elif test_type == 'Slope Test':
                test_run = next(i for i in self.fms_entry.functional_tests if i.test_id == self.test_dict.get(dynamic_field.value))
                return lambda: display_with_remark(
                    lambda: self.open_loop_test_query(self.test_dict.get(dynamic_field.value), test_type='slope'),
                    test_run, FMSFunctionalTests
                )

            elif test_type == 'Open Loop Test':
                test_run = next(i for i in self.fms_entry.functional_tests if i.test_id == self.test_dict.get(dynamic_field.value))
                return lambda: display_with_remark(
                    lambda: self.open_loop_test_query(self.test_dict.get(dynamic_field.value), test_type='open_loop'),
                    test_run, FMSFunctionalTests
                )

            elif test_type == 'Closed Loop Test':
                test_run = next(i for i in self.fms_entry.functional_tests if i.test_id == self.test_dict.get(dynamic_field.value))
                return lambda: display_with_remark(
                    lambda: self.closed_loop_test_query(self.test_dict.get(dynamic_field.value)),
                    test_run, FMSFunctionalTests
                )

            elif test_type == 'FR Characteristics':
                test_run = next(i for i in self.fms_entry.fr_tests if i.test_id == self.test_dict.get(dynamic_field.value))
                return lambda: display_with_remark(
                    lambda: self.fr_test_query(self.test_dict.get(dynamic_field.value)),
                    test_run, FMSFRTests
                )

            elif test_type == 'Tvac Cycle':
                test_run = next(i for i in self.fms_entry.tvac_results if i.test_id == dynamic_field.value)
                return lambda: display_with_remark(
                    lambda: self.tvac_cycle_query(dynamic_field.value),
                    test_run, FMSTvac
                )

            elif test_type == 'FMS Trend Analysis':
                return lambda: display_with_remark(lambda: self.fms_comparison_query(dynamic_field.value))

            elif test_type == 'Part Investigation':
                return lambda: display_with_remark(lambda: self.part_investigation_query(dynamic_field.value))

            else:
                return lambda: None

        # --- Listeners ---
        def on_fms_change(change: dict) -> None:
            if change['type'] == 'change' and change['name'] == 'value':
                self.load_all_tests(fms_id=change['new']) if not self.fms_id else self.load_all_tests(self.fms_id)
                with output:
                    output.clear_output(wait=True)
                    self.fms_status()
                query_field.value = 'Status'
                dynamic_field.options = []
                dynamic_field.value = None

        fms_field.observe(on_fms_change, names='value')

        def on_query_change(change: dict) -> None:
            self.test_dict = {}
            dynamic_field.value = None
            if change['type'] == 'change' and change['name'] == 'value':
                test_type = change['new']

                if test_type == 'Open Loop Test':
                    filtered = self.get_open_loop_tests()
                    test_headers = [f"{i.test_id[:10]} {i.trp_temp} [degC] {i.inlet_pressure} [barA]" for i in filtered]
                    dynamic_field.options = test_headers
                    self.test_dict = {h: i.test_id for h, i in zip(test_headers, filtered)}
                    dynamic_field.description = "Select Test:" if filtered else "No Open Loop Tests Found"

                elif test_type == "Acceptance Test":
                    dynamic_field.options = []
                    dynamic_field.value = None
                    with output:
                        output.clear_output(wait=True)
                        self.fms_characteristics_query()

                elif test_type == 'Slope Test':
                    filtered = self.get_slope_tests()
                    test_headers = [f"{i.test_id[:10]} {i.trp_temp} [degC] {i.inlet_pressure} [barA]" for i in filtered]
                    dynamic_field.options = test_headers
                    self.test_dict = {h: i.test_id for h, i in zip(test_headers, filtered)}
                    dynamic_field.description = "Select Test:" if filtered else "No Slope Tests Found"

                elif test_type == 'Closed Loop Test':
                    filtered = self.get_closed_loop_tests()
                    test_headers = [f"{i.test_id[:10]} {i.trp_temp} [degC] {i.inlet_pressure} [barA]" for i in filtered]
                    dynamic_field.options = test_headers
                    self.test_dict = {h: i.test_id for h, i in zip(test_headers, filtered)}
                    dynamic_field.description = "Select Test:" if filtered else "No Closed Loop Tests Found"

                elif test_type == 'FR Characteristics':
                    filtered = self.get_fr_tests()
                    test_headers = [f"{t.test_id} {t.inlet_pressure} [barA]" for t in filtered]
                    dynamic_field.options = test_headers
                    self.test_dict = {h: t.test_id for h, t in zip(test_headers, filtered)}
                    dynamic_field.description = "Select Test:" if filtered else "No FR Tests Found"

                elif test_type == 'Tvac Cycle':
                    try:
                        dynamic_field.options = [[t.test_id for t in self.tvac_tests_list][0]]
                    except:
                        dynamic_field.options = []
                        dynamic_field.description = "No TVAC Tests"
                    dynamic_field.description = "Select Test:" if self.tvac_tests_list else "No Tvac Tests Found"

                elif test_type == 'Part Investigation':
                    dynamic_field.options = [" ".join([j.capitalize() for j in i.value.split(" ")]) if i != FMSParts.HPIV else i.value.upper() for i in FMSParts]
                    dynamic_field.description = "Select Part:"

                elif test_type == 'FMS Trend Analysis':
                    dynamic_field.options = ['TV Slope Analysis', 'Flow Rate Analysis', 'Closed Loop Analysis', 'Free Choice']
                    dynamic_field.description = "Analysis Type:"

                else:  # Status
                    dynamic_field.options = []
                    get_dynamic_callable('Status')()

        query_field.observe(on_query_change, names='value')

        def on_dynamic_change(change: dict) -> None:
            if change['type'] == 'change' and change['name'] == 'value':
                func = get_dynamic_callable(query_field.value)
                func()

        dynamic_field.observe(on_dynamic_change, names='value')

        # --- Layout ---
        form = widgets.VBox([
            widgets.HTML("<h3>FMS Query Form</h3>"),
            fms_field,
            query_field,
            dynamic_field,
            output
        ], layout=widgets.Layout(border='1px solid #ccc', padding='12px', width='100%'))

        display(form)

        # --- Run default status ---
        if fms_field.value:
            self.load_all_tests(fms_id=fms_field.value)
            with output:
                output.clear_output(wait=True)
                self.fms_status()

    def fms_status(self) -> None:
        """
        Display the status of the selected FMS in a structured DataFrame.
        """
        if self.fms_entry:
            columns = [c.name for c in self.fms_entry.__table__.columns]
            values = [getattr(self.fms_entry, c) for c in columns]
            df = pd.DataFrame({"Field": columns, "Value": values})
            display_df_in_chunks(df)
        else:
            print("FMS ID not found.")
            
    def fms_test_remark_field(self, test_run: object = None, look_up_table: object = None) -> None:
        """
        Create a clean input field for adding/modifying remarks with properly styled widgets.
        """
        label_width = '150px'
        field_width = '600px'
        
        title = widgets.HTML("<h3>Add a remark if necessary</h3>")

        def field(description):
            return {
                'description': description,
                'style': {'description_width': label_width},
                'layout': widgets.Layout(width=field_width, height='40px')
            }

        # Remark input
        remark_widget = widgets.Textarea(**field("Remark:"), value = test_run.remark)

        # Submit button
        submit_button = widgets.Button(
            description="Submit Remark",
            button_style="success",
            layout=widgets.Layout(width='150px', margin='10px 0px 0px 160px')  # align under field
        )

        submitted = {'done': False}
        output = widgets.Output()
        # Form layout
        form = widgets.VBox([
            title,
            widgets.HTML('<p>Examine the plots and add a remark if necessary.</p>'
            ),
            widgets.HBox([remark_widget]),
            submit_button,
            output
        ], layout=widgets.Layout(
            border='1px solid #ccc',
            padding='20px',
            width='fit-content',
            gap='15px',
            background_color="#f9f9f9"
        ))

        display(form)

        def on_submit_clicked(b):
            with output:
                output.clear_output()
                remark = remark_widget.value.strip()
                if not remark:
                    print("No remark submitted.")
                    return
                prev_remark = test_run.remark or ""
                if remark == prev_remark:
                    print("Already submitted!")
                else:
                    test_run.remark = remark
                    self.session.commit()
                    print("Remark Submitted!")

        submit_button.on_click(on_submit_clicked)

    def fr_test_query(self, test_id: str, plot: bool = True) -> io.BytesIO | None:
        """
        Query and plot FR characteristic test data for the given test ID.
        Args:
            test_id (str): The ID of the FR test to query.
            plot (bool): Whether to display the plot or return it as a BytesIO object.
        Returns:
            BytesIO: The plot image in a BytesIO object if plot is False.
        """
        test_runs: list[FMSFRTests] = self.fms_entry.fr_tests
        test_run: FMSFRTests = next((tr for tr in test_runs if tr.test_id == test_id), None)
        self.intersections = {}
        if not test_run:
            print("Test ID not found for this FMS.")
            return
        else:
            self.gas_type = self.fms_entry.gas_type if self.fms_entry else 'Xe'
            manifold: list[ManifoldStatus] = self.fms_entry.manifold
            if manifold:
                self.ratio = manifold[0].ac_ratio_specified
            else:
                self.ratio = 13
            if not self.ratio:
                self.ratio = 13
            self.temperature = test_run.trp_temp
            self.inlet_pressure = test_run.inlet_pressure
            self.inlet_pressure = 10 if self.inlet_pressure < 100 else 190
            self.outlet_pressure = test_run.outlet_pressure
            self.units = {
                FMSFlowTestParameters.TOTAL_FLOW.value: f'mg/s',
                FMSFlowTestParameters.LPT_PRESSURE.value: 'bar',
                FMSFlowTestParameters.ANODE_FLOW.value: f'mg/s',
                FMSFlowTestParameters.CATHODE_FLOW.value: f'mg/s',

            }
            self.logtime = test_run.logtime
            self.anode_flow = test_run.anode_flow
            self.cathode_flow = test_run.cathode_flow
            self.total_flow = test_run.total_flow
            self.lpt_pressure = test_run.lpt_pressure
            self.lpt_voltage = test_run.lpt_voltage
            self.ac_ratio = test_run.ac_ratio
    
            self.intersections = find_intersections(self.lpt_voltage, self.total_flow, self.lpt_voltages, self.min_flow_rates, self.max_flow_rates)

            image = self.plot_fr_characteristics(serial = self.fms_entry.fms_id, gas_type = self.gas_type, plot=plot)
            return image

    def plot_fr_characteristics(self, gas_type: str = 'Xe', serial: str = '25-050', plot: bool = True) -> io.BytesIO | None:
        """
        Plot FR characteristics for the given gas type and serial number.

        Args:
            gas_type (str): The type of gas used in the test (default is 'Xe').
            serial (str): The serial number of the FMS (default is '25-050').
            plot (bool): Whether to display the plot or return it as a BytesIO object.
        Returns:
            BytesIO: The plot image in a BytesIO object if plot is False.
        """
        
        fig, ax1 = plt.subplots(figsize=(9, 5))

        df_data = {
            "Inlet Pressure [barA]": self.lpt_pressure,
            f"Anode Flow \n Restrictor Flow Rate [mg/s {self.gas_type}]": [f"{i:.3f}" for i in self.anode_flow],
            f"Cathode Flow \n Restrictor Flow Rate [mg/s {self.gas_type}]": [f"{i:.3f}" for i in self.cathode_flow],
            "Anode-to-Cathode Ratio [13 +/- 0.5]": [f"{i:.2f}" for i in self.ac_ratio],
            f"Total Flow Rate [mg/s {self.gas_type}]": [f"{i:.3f}" for i in self.total_flow],
            "LPT Pressure [mV]": [f"{i:.2f}" for i in self.lpt_voltage]
        }

        df = pd.DataFrame(df_data)
        df = df.to_html(index=False)
        df_widget = widgets.HTML(value=df, layout=widgets.Layout(width='50%'))

        l1, = ax1.plot(self.lpt_pressure, self.anode_flow, label=f'Anode Flow [{self.units[FMSFlowTestParameters.ANODE_FLOW.value]}]')
        l2, = ax1.plot(self.lpt_pressure, self.cathode_flow, label=f'Cathode Flow [{self.units[FMSFlowTestParameters.CATHODE_FLOW.value]}]')
        l3, = ax1.plot(self.lpt_pressure, self.total_flow, label=f'Total Flow [{self.units[FMSFlowTestParameters.TOTAL_FLOW.value]}]')
        ax1.set_xlabel(f'LPT Pressure [{self.units[FMSFlowTestParameters.LPT_PRESSURE.value]}]')
        ax1.set_ylabel(f'{gas_type} Mass Flow Rate [{self.units[FMSFlowTestParameters.ANODE_FLOW.value]} {gas_type}]')
        ax1.grid(True)

        title = (
            f'{gas_type} LP FMS - SN {serial} - {self.inlet_pressure} [barA] Inlet Pressure - {self.outlet_pressure} [mbar] Outlet Pressure'
            f' - TRP at {self.temperature} [degC] - Pvac <1E-1 [mbar]'
        )

        ax2 = ax1.twinx()
        l4, = ax2.plot(self.lpt_pressure, self.ac_ratio, color='tab:red', label='Anode/Cathode Ratio')
        l5_upper = ax2.axhline(self.ratio + 0.5, color='tab:orange', linestyle='--', label=f'Ratio Tolerance: {self.ratio}')
        l5_lower = ax2.axhline(self.ratio - 0.5, color='tab:orange', linestyle='--', label='Ratio Tolerance')
        ax2.set_ylabel('Anode-to-Cathode Ratio')
        ax2.set_ylim(bottom=0, top=20)
        ax2.set_yticks(np.arange(0, 21, 1))

        # Combine legends from both axes
        lines = [l1, l2, l3, l4, l5_upper]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

        plt.title(title, wrap=True)
        plot_output = widgets.Output()
        data_widget = widgets.HBox([plot_output, df_widget], layout=widgets.Layout(align_items='center', gap='20px'))
        display(data_widget)
        if plot:
            with plot_output:
                plt.show()
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf
        self.plot_fr_voltage(title=title, gas_type=gas_type)

    def plot_fr_voltage(self, title: str = None, gas_type: str = 'Xe') -> None:
        """
        Plot FR characteristics vs LPT voltage with interactive polynomial order selection.
        """
        description = widgets.HTML("<h3>FR Characteristics Extrapolation Analysis</h3>")
        poly_order_widget = widgets.IntSlider(  
            value=3,
            min=1,
            max=20,
            step=1,
            description='Choose Polynomial Order for Extrapolation:',
            continuous_update=True,
            layout=widgets.Layout(width='450px'),
            style={'description_width': '250px'}
        )

        form = widgets.VBox([
            description,
            poly_order_widget
        ], layout=widgets.Layout(padding='12px', width='fit-content'))
        display(form)

        # Output widgets
        plot_output = widgets.Output()
        df_widget = widgets.HTML(layout=widgets.Layout(width='50%'))

        # Single HBox to hold plot + table
        data_widget = widgets.HBox([plot_output, df_widget],
                                layout=widgets.Layout(align_items='center', gap='20px'))
        display(data_widget)

        def on_poly_order_change(change: dict) -> None:
            order = change['new']
            polyfit = np.polyfit(self.lpt_voltage, self.total_flow, order)
            total_flow_check = np.polyval(polyfit, self.lpt_voltage).flatten().tolist()
            calculated_total_flows = np.polyval(polyfit, self.lpt_voltages).flatten().tolist()

            # Compute R²
            r2 = r2_score(self.total_flow, total_flow_check)

            df = pd.DataFrame({
                'LPT Voltage [mV]': self.lpt_voltages,
                f'Min Flow Rate [mg/s {gas_type}]': self.min_flow_rates,
                f'Max Flow Rate [mg/s {gas_type}]': self.max_flow_rates,
                f'Calculated Total Flow [mg/s {gas_type}]': calculated_total_flows,
                'Compliance': [
                    'C' if min_f <= calc_f <= max_f else 'F'
                    for min_f, max_f, calc_f in zip(self.min_flow_rates, self.max_flow_rates, calculated_total_flows)
                ]
            })

            # Define the styling function
            def style_total_flow(row):
                min_val = row[f'Min Flow Rate [mg/s {gas_type}]']
                max_val = row[f'Max Flow Rate [mg/s {gas_type}]']
                val = row[f'Calculated Total Flow [mg/s {gas_type}]']
                color = 'green' if min_val <= val <= max_val else 'red'
                return [
                    f'background-color: {color}' if col == f'Calculated Total Flow [mg/s {gas_type}]' else ''
                    for col in row.index
                ]

            # Column formatting
            fmt_dict = {
                'LPT Voltage [mV]': "{:.0f}",
                f'Min Flow Rate [mg/s {gas_type}]': "{:.2f}",
                f'Max Flow Rate [mg/s {gas_type}]': "{:.2f}",
                f'Calculated Total Flow [mg/s {gas_type}]': "{:.3f}"
            }

            # Apply styling and formatting
            styled_df = df.style.apply(style_total_flow, axis=1).format(fmt_dict).hide(axis='index')
            df_widget.value = styled_df.to_html(index = False)

            with plot_output:
                plot_output.clear_output()
                fig, ax1 = plt.subplots(figsize=(9,7))
                l3, = ax1.plot(self.lpt_voltage, self.total_flow, label=f'Total Flow [{self.units[FMSFlowTestParameters.TOTAL_FLOW.value]}]')
                ax1.set_xlabel('LPT Voltage [mV]')
                ax1.set_ylabel(f'Mass Flow Rate [{self.units[FMSFlowTestParameters.ANODE_FLOW.value]} {gas_type}]')
                ax1.grid(True)
                if title:
                    plt.title(title, wrap=True)

                ax2 = ax1.twinx()
                l4, = ax2.plot(self.lpt_voltage, self.ac_ratio, color='tab:red', label='Anode/Cathode Ratio')
                l7, = ax1.plot(self.lpt_voltages, calculated_total_flows, linestyle='--', color='tab:blue', label=f'Calculated Total Flow (R²={r2:.2f})')
                l5, = ax1.plot(self.lpt_voltages, self.min_flow_rates, linestyle='--', color='tab:grey', label='Flow Rate Limits')
                l6, = ax1.plot(self.lpt_voltages, self.max_flow_rates, linestyle='--', color='tab:grey', label='Flow Rate Limits')
                l5_upper = ax2.axhline(self.ratio + 0.5, color='tab:orange', linestyle='--', label=f'Ratio Tolerance: {self.ratio}')
                l5_lower = ax2.axhline(self.ratio - 0.5, color='tab:orange', linestyle='--', label='Ratio Tolerance')

                ax2.set_ylabel('Anode-to-Cathode Ratio')
                ax2.set_ylim(bottom=0, top=20)
                ax2.set_yticks(np.arange(0, 21, 1))

                if self.intersections.get('intersections', None):
                    intersections = self.intersections["intersections"]
                    for voltage, flow in intersections:
                        ax1.plot(voltage, flow, 'ro')

                # Legend
                lines = [l3, l4, l7, l5, l5_upper]
                labels = [line.get_label() for line in lines]
                ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
                plt.tight_layout()
                plt.show()

        poly_order_widget.observe(on_poly_order_change, names='value')
        on_poly_order_change({'new': poly_order_widget.value})

    def open_loop_test_query(self, test_id: str, test_type: str = 'slope', plot: bool = True) -> io.BytesIO | None:
        """
        Query and plot open loop test data for the given test ID and test type.
        Args:
            test_id (str): The ID of the open loop test to query.
            test_type (str): The type of open loop test ('slope' or 'open_loop').
            plot (bool): Whether to display the plot or return it as a BytesIO object.
        Returns:
            BytesIO: The plot image in a BytesIO object if plot is False.
        """
        test_run: FMSFunctionalTests | None = next((tr for tr in self.fms_entry.functional_tests if tr.test_id == test_id), None)
        self.tv_slope = None
        if not test_run:
            print("Test ID not found for this FMS.")
            return
        else:
            self.gas_type = self.fms_entry.gas_type if self.fms_entry else 'Xe'
            self.temperature = test_run.trp_temp
            self.inlet_pressure = test_run.inlet_pressure
            self.inlet_pressure = 10 if self.inlet_pressure < 100 else 190
            correction_factor = test_run.slope_correction
            self.outlet_pressure = test_run.outlet_pressure
            plot_output = widgets.Output()
            correction_checkbox = widgets.Checkbox(
                value = False,
                description = 'Show Inlet Pressure Correction:',
                indent = False,
                label_width = '150px'
            )
            test_results: list[FMSFunctionalResults] = test_run.functional_results
            if test_results:
                self.logtime = [res.logtime for res in test_results if res.parameter_name == FMSFlowTestParameters.ANODE_FLOW.value] or None

                params = [
                    FMSFlowTestParameters.AVG_TV_POWER.value,
                    FMSFlowTestParameters.TOTAL_FLOW.value,
                    FMSFlowTestParameters.LPT_PRESSURE.value,
                    FMSFlowTestParameters.TV_PT1000.value,
                ]

                df_filtered = pd.DataFrame([{
                    'parameter_name': res.parameter_name,
                    'parameter_value': res.parameter_value,
                    'parameter_unit': getattr(res, 'parameter_unit', None),
                    'logtime': res.logtime
                } for res in test_results if res.parameter_name in params])


                # df_filtered = df[df['parameter_name'].isin(params)]

                def get_values(param_name: str) -> list | None:
                    vals = df_filtered.loc[df_filtered['parameter_name'] == param_name, 'parameter_value'].tolist()
                    return vals or None

                def get_unit(param_name: str) -> str | None:
                    unit = df_filtered.loc[df_filtered['parameter_name'] == param_name, 'parameter_unit']
                    return unit.iloc[0] if not unit.empty else None

                if 'slope' in test_type.lower():
                    tv_df = df_filtered[df_filtered['parameter_name'] == FMSFlowTestParameters.AVG_TV_POWER.value]
                    p = tv_df['parameter_value'].to_numpy(dtype=float)
                    t = tv_df['logtime'].to_numpy(dtype=float)
                    mask = p[1:] > p[:-1]
                    tv_powers_masked = p[:-1][mask][50:]
                    tv_times_masked = t[:-1][mask][50:]
                    self.tv_slope = np.mean(np.diff(tv_powers_masked) / np.diff(tv_times_masked)) * 60

                self.tv_powers = get_values(FMSFlowTestParameters.AVG_TV_POWER.value)
                self.total_flow = get_values(FMSFlowTestParameters.TOTAL_FLOW.value)
                self.lpt_pressure = get_values(FMSFlowTestParameters.LPT_PRESSURE.value)
                self.pt1000 = get_values(FMSFlowTestParameters.TV_PT1000.value)
                self.units = {param: get_unit(param) for param in params}

                image = self.plot_open_loop(serial=self.fms_entry.fms_id, gas_type=self.gas_type, plot=plot)

                def on_correction_change(change: dict):
                    correction = correction_checkbox.value
                    total_flow = np.array(self.total_flow) / correction_factor if correction else np.array(self.total_flow)
                    if plot and 'slope' in test_type.lower():
                        flows = total_flow
                        powers = np.array(self.tv_powers)
                        flow_power_slope = self.get_flow_power_slope(flows, powers)
                        with plot_output:
                            plot_output.clear_output()
                            self.check_tv_slope(**flow_power_slope, correction = correction, serial = self.fms_entry.fms_id)

                correction_checkbox.observe(on_correction_change, names='value')
                display(correction_checkbox)
                display(plot_output)
                on_correction_change({})
                return image

    def get_flow_power_slope(self, flows: list[float], powers: list[float], num_points: int = 300) -> dict:
        """
        Calculate the flow-power slope for specified ranges of flow rates.
        Args:
            flows (list[float]): List of flow rate values.
            powers (list[float]): List of power values.
            num_points (int): Number of points for smoothing.
        Returns:
            dict: A dictionary containing smoothed power and flow values, slopes, and intercepts for 1-2 mg/s and 2-4 mg/s ranges.
        """
        mask = powers > 0.2
        flows = flows[mask]
        powers = powers[mask]

        def get_region(flow_vals: np.ndarray, power_vals: np.ndarray, lower_bound: float, upper_bound: float) -> tuple[np.ndarray, np.ndarray]:
            below_idx = np.where(flow_vals < lower_bound)[0]
            above_idx = np.where(flow_vals > upper_bound)[0]

            if len(below_idx) == 0:
                start_idx = 0
            else:
                start_idx = below_idx[-1]

            if len(above_idx) == 0:
                end_idx = len(flow_vals) - 1
            else:
                end_idx = above_idx[0]

            return power_vals[start_idx:end_idx + 1], flow_vals[start_idx:end_idx + 1]

        def smooth_and_slope(power_segment: np.ndarray, flow_segment: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
            if len(power_segment) < 2 or len(flow_segment) < 2:
                return np.array([]), np.array([]), 0, 0

            interp_func = interp1d(power_segment, flow_segment, kind='linear', fill_value="extrapolate")
            power_smooth = np.linspace(power_segment.min(), power_segment.max(), num_points)
            flow_smooth = interp_func(power_smooth)

            model = LinearRegression()
            model.fit(power_smooth.reshape(-1, 1), flow_smooth)
            slope = model.coef_[0]
            intercept = model.intercept_

            return power_smooth, flow_smooth, slope, intercept

        # 1–2 mg/s
        tv_power_12, total_flows_12 = get_region(flows, powers, 1, 2)
        tv_power_12_smooth, total_flows_12_smooth, slope12, intercept12 = smooth_and_slope(tv_power_12, total_flows_12)

        # 2–4 mg/s
        tv_power_24, total_flows_24 = get_region(flows, powers, 2, 4)
        tv_power_24_smooth, total_flows_24_smooth, slope24, intercept24 = smooth_and_slope(tv_power_24, total_flows_24)

        array_dict = {
            'tv_power_12': tv_power_12_smooth,
            'total_flows_12': total_flows_12_smooth,
            'slope12': slope12,
            'intercept12': intercept12,
            'tv_power_24': tv_power_24_smooth,
            'total_flows_24': total_flows_24_smooth,
            'slope24': slope24,
            'intercept24': intercept24
        }

        return array_dict

    def check_tv_slope(self, tv_power_12: np.ndarray, tv_power_24: np.ndarray, total_flows_12: np.ndarray, \
                       total_flows_24: np.ndarray, slope12: float, slope24: float, intercept12: float, intercept24: float, correction: bool = False,\
                        serial: str = "") -> None:
        """
        Plot the flow-power slopes and check against specifications.
        Args:
            tv_power_12 (np.ndarray): Smoothed TV power values for 1-2 mg/s range.
            tv_power_24 (np.ndarray): Smoothed TV power values for 2-4 mg/s range.
            total_flows_12 (np.ndarray): Smoothed flow values for 1-2 mg/s range.
            total_flows_24 (np.ndarray): Smoothed flow values for 2-4 mg/s range.
            slope12 (float): Slope for 1-2 mg/s range.
            slope24 (float): Slope for 2-4 mg/s range.
            intercept12 (float): Intercept for 1-2 mg/s range.
            intercept24 (float): Intercept for 2-4 mg/s range.
        """
        try:
            slope_line_12 = slope12 * tv_power_12 + intercept12
            slope_line_24 = slope24 * tv_power_24 + intercept24

            min_slope12 = self.range12_low[0] if self.inlet_pressure < 100 else self.range12_high[0]
            max_slope12 = self.range12_low[1] if self.inlet_pressure < 100 else self.range12_high[1]
            min_slope24 = self.range24_low[0] if self.inlet_pressure < 100 else self.range24_high[0]
            max_slope24 = self.range24_low[1] if self.inlet_pressure < 100 else self.range24_high[1]

            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            plt.plot(tv_power_12, total_flows_12, 'b-', label='1-2 mg/s')
            plt.plot(tv_power_12, slope_line_12, 'g--', label=(f"Slope: {slope12:.2f} mg/s W^-1 [✓]" if min_slope12 <= slope12 <= max_slope12 
                                                               else f"Slope: {round(slope12)} mg/s W^-1 [✗]"))
            plt.xlabel(f'TV Power [{self.units[FMSFlowTestParameters.AVG_TV_POWER.value]}]')
            plt.ylabel(f'Total Flow [{self.units[FMSFlowTestParameters.TOTAL_FLOW.value]} {self.gas_type}]')
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            textstr = '\n'.join([
                'Slopes from Spec:',
                f'Min Slope: {min_slope12} [mg/s W^-1]',
                f'Max Slope: {max_slope12} [mg/s W^-1]',
            ])
            props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='red', alpha=0.9)
            plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='left',
                    bbox=props)
            plt.title(f'Total Flow {serial} vs TV Power (1-2 mg/s)\nTRP temp: {self.temperature} [degC], Inlet Pressure: {self.inlet_pressure} [barA]' if not correction \
                else f'Total Flow {serial} vs TV Power (1-2 mg/s)\nTRP temp: {self.temperature} [degC], Inlet Pressure: {self.inlet_pressure} [barA]' + ' (Corrected)')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(tv_power_24, total_flows_24, 'r-', label='2-4 mg/s')
            plt.plot(tv_power_24, slope_line_24, 'g--', label=(f"Slope: {slope24:.2f} mg/s W^-1 [✓]" if min_slope24 <= slope24 <= max_slope24 
                                                               else f"Slope: {round(slope24)} mg/s W^-1 [✗]"))
            plt.xlabel(f'TV Power [{self.units[FMSFlowTestParameters.AVG_TV_POWER.value]}]')
            plt.ylabel(f'Total Flow [{self.units[FMSFlowTestParameters.TOTAL_FLOW.value]} {self.gas_type}]')
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            textstr = '\n'.join([
                'Slopes from Spec:',
                f'Min Slope: {min_slope24} [mg/s W^-1]',
                f'Max Slope: {max_slope24} [mg/s W^-1]',
            ])
            props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='red', alpha=0.9)
            plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='left',
                    bbox=props)    
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
            plt.title(f'Total Flow {serial} vs TV Power (2-4 mg/s)\nTRP temp: {self.temperature} [degC], Inlet Pressure: {self.inlet_pressure} [barA]' if not correction \
                    else f'Total Flow {serial} vs TV Power (2-4 mg/s)\nTRP temp: {self.temperature} [degC], Inlet Pressure: {self.inlet_pressure} [barA]' + ' (Corrected)')
            plt.grid(True)   
            plt.tight_layout()
            plt.show()
        except:
            traceback.print_exc()

    def plot_open_loop(self, serial: str ='25-050', gas_type: str ='Xe', plot: bool =True) -> io.BytesIO | None:
        """
        Plot open loop or slope test data for the given gas type and serial number.
        Args:
            gas_type (str): The type of gas used in the test (default is 'Xe').
            serial (str): The serial number of the FMS (default is '25-050').
            plot (bool): Whether to display the plot or return it as a BytesIO object.
        Returns:
            BytesIO: The plot image in a BytesIO object if plot is False.
        """
        fms = FMSFlowTestParameters
        fig, ax1 = plt.subplots(figsize=(9, 5))
        color1 = 'tab:blue'
        color2 = 'tab:orange'
        color3 = 'tab:green'

        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel(f'TV Temperature [{self.units[fms.TV_PT1000.value]}]')
        l1, = ax1.plot(self.logtime, self.pt1000, label='TV Temperature')

        ax2 = ax1.twinx()
        ax2.set_ylabel(f'Total Flow [{self.units[fms.TOTAL_FLOW.value]} {gas_type}] / LPT Pressure [{self.units[fms.LPT_PRESSURE.value]}]')
        l2, = ax2.plot(self.logtime, self.total_flow, label='Total Flow', color=color2)
        l3, = ax2.plot(self.logtime, self.lpt_pressure, label='LPT Pressure', color=color3)

        title = (
            f'LP FMS - SN {serial}, TRP at {self.temperature} [degC], MLI, '
            f'{self.inlet_pressure} [barA] Inlet Pressure, '
        )

        if self.tv_slope:
            title += f'{self.tv_slope:.2f} [W/min], '
        else:
            title += f'{max(self.tv_powers):.1f}W, '

        title += f'\nPvac <1E-1 [mbarA], {self.outlet_pressure} [mbar] Outlet Pressure'
        ax1.set_title(title, wrap=True)
        fig.tight_layout()

        lines = [l1, l2, l3]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        ax1.grid(True)

        if plot:
            plt.show()
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf
        
    def closed_loop_test_query(self, test_id: str, plot: bool =True) -> io.BytesIO | None:
        """
        Query and plot closed loop test data for the given test ID.
        Args:
            test_id (str): The ID of the closed loop test to query.
            plot (bool): Whether to display the plot or return it as a BytesIO object.
        Returns:
            BytesIO: The plot image in a BytesIO object if plot is False.
        """
        test_run: FMSFunctionalTests | None = next((tr for tr in self.fms_entry.functional_tests if tr.test_id == test_id), None)
        if not test_run:
            print("Test ID not found for this FMS.")
            return
        else:
            plot_output = widgets.Output()
            tv_plot_output = widgets.Output()
            df_widget = widgets.VBox()
            self.gas_type = self.fms_entry.gas_type if self.fms_entry else 'Xe'
            test_results: list[FMSFunctionalResults] = test_run.functional_results
            self.temperature = test_run.trp_temp
            self.inlet_pressure = test_run.inlet_pressure
            self.inlet_pressure = 10 if self.inlet_pressure < 100 else 190
            self.outlet_pressure = test_run.outlet_pressure
            self.test_type: str = test_run.test_type.value
            self.response_times = test_run.response_times
            self.response_regions = test_run.response_regions
            test_info = []

            show_response_times_checkbox = widgets.Checkbox(
                value = False,
                description = 'Show Response Times:',
                indent = False,
                label_width = '150px'
            )

            if self.response_times:
                row = {"Temperature [degC]": self.temperature, 
                "Inlet Pressure [barA]": self.inlet_pressure,
                "Description": "Opening Time"}
                
                opening_times = self.response_times.get("opening_time", [])
                for i, tau in enumerate(opening_times[:3]):
                    row[f"{i+1}_tau [s]"] = tau
                row["Actual Time [s]"] = opening_times[-1] if opening_times else "N/A"
                test_info.append(row)
                
                for set_idx, set_point in enumerate(self.lpt_set_points):
                    row = {"Temperature [degC]": self.temperature,
                            "Inlet Pressure [barA]": self.inlet_pressure}

                    if set_idx == 0:
                        key = f"response_time_to_{set_point}_barA"
                    elif set_idx == len(self.lpt_set_points) - 1:
                        key = f"closing_time_to_{set_point}_barA"
                    else:
                        key = f"response_{self.lpt_set_points[set_idx-1]}_to_{set_point}_barA"

                    description = key.replace("_", " ").title().replace('Bara', 'barA')
                    row["Description"] = description
                    tau_list = self.response_times.get(key, [])
                    for i, tau in enumerate(tau_list[:3]):
                        row[f"{i+1}_tau [s]"] = tau
                    row["Actual Time [s]"] = tau_list[-1] if tau_list else "N/A"
                    if not all(np.isnan(t) for t in tau_list):
                        test_info.append(row)
                    else:
                        tau_list = ["N/A" for _ in range(len(tau_list))]
                        for i, tau in enumerate(tau_list[:3]):
                            row[f"{i+1}_tau [s]"] = tau
                        row["Actual Time [s]"] = tau_list[-1] if tau_list else "N/A"
                        test_info.append(row)
                df = pd.DataFrame(test_info)
                df = df.fillna("N/A")
                tau_cols = [col for col in df.columns if "_tau" in col]

                def color_cells(val, desc):
                    if val == "N/A" or pd.isna(val):
                        return ''
                    if "Opening Time" in desc:
                        if val < 300:
                            return 'background-color: green'
                        elif val == 300:
                            return 'background-color: orange'
                        else:
                            return 'background-color: red'
                    else:
                        if val < 60:
                            return 'background-color: green'
                        elif val == 60:
                            return 'background-color: orange'
                        else:
                            return 'background-color: red'

                def format_numeric(val):
                    return f"{val:.1f}" if isinstance(val, (int, float)) else val
                
                format_dict = {
                    **{col: format_numeric for col in tau_cols},
                    "Temperature [degC]": "{:.0f}",
                    "Inlet Pressure [barA]": "{:.0f}",
                    "Actual Time [s]": format_numeric
                }
                styled_df = df.style.apply(
                    lambda row: [color_cells(v, df.loc[row.name, "Description"]) for v in row],
                    axis=1,
                    subset=tau_cols
                ).format(format_dict).hide(axis='index')

                df_widget = widgets.HTML(value=styled_df.to_html(index=False), layout=widgets.Layout(width='50%'))

            if test_results:
                # Convert test_results into a DataFrame
                # df = pd.DataFrame([{
                #     "parameter_name": res.parameter_name,
                #     "parameter_value": res.parameter_value,
                #     "parameter_unit": res.parameter_unit,
                #     "logtime": res.logtime
                # } for res in test_results])

                # Helper to get values or None
                def get_values(param_name):
                    return [res.parameter_value for res in test_results if res.parameter_name == param_name]

                # Helper to get first unit
                def get_unit(param_name):
                    return [res.parameter_unit for res in test_results if res.parameter_name == param_name][0]

                # Populate class attributes
                self.logtime = [res.logtime for res in test_results if res.parameter_name == FMSFlowTestParameters.ANODE_FLOW.value] or None
                self.anode_flow = get_values(FMSFlowTestParameters.ANODE_FLOW.value)
                self.total_flow = get_values(FMSFlowTestParameters.TOTAL_FLOW.value)
                self.cathode_flow = get_values(FMSFlowTestParameters.CATHODE_FLOW.value)
                self.closed_loop_pressure = get_values(FMSFlowTestParameters.CLOSED_LOOP_PRESSURE.value)
                self.lpt_pressure = get_values(FMSFlowTestParameters.LPT_PRESSURE.value)
                self.tv_power = get_values(FMSFlowTestParameters.AVG_TV_POWER.value)
                self.pt1000 = get_values(FMSFlowTestParameters.TV_PT1000.value)
                # self.logtime = get_values(FMSFlowTestParameters.ANODE_FLOW.value)  # or whichever logtime you need

                # Units
                self.units = {
                    FMSFlowTestParameters.ANODE_FLOW.value: get_unit(FMSFlowTestParameters.ANODE_FLOW.value),
                    FMSFlowTestParameters.CATHODE_FLOW.value: get_unit(FMSFlowTestParameters.CATHODE_FLOW.value),
                    FMSFlowTestParameters.CLOSED_LOOP_PRESSURE.value: get_unit(FMSFlowTestParameters.CLOSED_LOOP_PRESSURE.value),
                    FMSFlowTestParameters.LPT_PRESSURE.value: get_unit(FMSFlowTestParameters.LPT_PRESSURE.value),
                    FMSFlowTestParameters.AVG_TV_POWER.value: get_unit(FMSFlowTestParameters.AVG_TV_POWER.value),
                    FMSFlowTestParameters.TV_PT1000.value: get_unit(FMSFlowTestParameters.TV_PT1000.value),
                    FMSFlowTestParameters.TOTAL_FLOW.value: get_unit(FMSFlowTestParameters.TOTAL_FLOW.value),
                }

                form = widgets.VBox([show_response_times_checkbox, widgets.HBox([plot_output, widgets.HBox(layout = widgets.Layout(width = "50px")), df_widget], 
                                layout=widgets.Layout(align_items='center', spacing='20px')), tv_plot_output], layout=widgets.Layout(padding='12px', width='fit-content'))
                display(form) 
                image = self.plot_closed_loop(serial=self.fms_entry.fms_id, gas_type=self.gas_type, plot=plot, plot_output = plot_output, tv_plot_output = tv_plot_output)

                def on_checkbox_clicked(change):
                    show_response_times = show_response_times_checkbox.value
                    image = self.plot_closed_loop(serial=self.fms_entry.fms_id, gas_type=self.gas_type, plot=plot, plot_output = plot_output, tv_plot_output=tv_plot_output, show_response_times=show_response_times)

                show_response_times_checkbox.observe(on_checkbox_clicked, names='value')

                return image

                # flows = np.array(self.total_flow)
                # mask = flows <= np.max(flows)
                # flows = flows[mask]
                # powers = np.array(self.tv_power)[mask]
                # logtime = np.array(self.logtime)[mask]
                # plt.plot(logtime, flows)
                # plt.show()
                # plt.plot(flows, powers)
                # plt.show()
                # flow_power_slope = self.get_flow_power_slope(flows, powers)

            else:
                print("No test results found for this test run.")
                return

    def plot_closed_loop(self, serial: str = '25-050', gas_type: str = 'Xe', plot: bool = True, plot_output: widgets.Output = None, tv_plot_output: widgets.Output = None,\
                         show_response_times: bool = False) -> io.BytesIO | None:
        """
        Plot closed loop test data including anode flow, cathode flow, closed loop pressure, and LPT pressure.
        Args:
            serial (str): The serial number of the FMS.
            gas_type (str): The type of gas used in the test.
            plot (bool): Whether to display the plot or return it as a BytesIO object.
            plot_output (widgets.Output, optional): The output widget to display the plot. Defaults to None.
            show_response_times (bool, optional): Whether to show response times on the plot. Defaults to False.    
        Returns:
            BytesIO: The plot image in a BytesIO object if plot is False.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(self.logtime, self.anode_flow, label=f'Anode Flow [{self.units[FMSFlowTestParameters.ANODE_FLOW.value]}]')
        ax.plot(self.logtime, self.cathode_flow, label=f'Cathode Flow [{self.units[FMSFlowTestParameters.CATHODE_FLOW.value]}]')
        ax.plot(self.logtime, self.closed_loop_pressure, label=f'Closed Loop Setpoint [{self.units[FMSFlowTestParameters.CLOSED_LOOP_PRESSURE.value]}]')
        ax.plot(self.logtime, self.lpt_pressure, label=f'LPT Pressure [{self.units[FMSFlowTestParameters.LPT_PRESSURE.value]}]')

        title = f'LP FMS - SN {serial}, TRP at {self.temperature} [degC], MLI, {self.inlet_pressure} [barA] Inlet Pressure, {self.test_type.replace("_", " ").title()}, \nPvac <1E-1 [mbarA], {self.outlet_pressure} [mbar] Outlet Pressure'
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Mass Flow Rate [{self.units[FMSFlowTestParameters.ANODE_FLOW.value]} {gas_type}]/LPT & Setpoint Pressure [barA]')
        ax.set_title(title, wrap=True)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        ax.grid()

        if show_response_times:
            count = 0
            for region_key, (cl_start_time, lpt_start_time) in self.response_regions.items():
                y_fill = max(max(self.anode_flow), max(self.cathode_flow))
                y_level = self.lpt_set_points[count]
                count += 1

                x0 = cl_start_time
                x1 = lpt_start_time
                xm = (x0 + x1) / 2
                dt = x1 - x0

                ax.fill_between([x0, x1], y1=0, y2=y_fill, alpha=0.4, color='tab:blue')

                ax.plot([x0, x1], [y_level, y_level], color = 'black', linewidth=1)
                ax.plot([x0, x0], [y_level - 0.1, y_level + 0.1], color = 'black', linewidth=1)
                ax.plot([x1, x1], [y_level - 0.1, y_level + 0.1], color = 'black', linewidth=1)
                ax.text(xm, y_level + 0.3, f"{dt:.1f} s", ha='center', va='bottom')

        if plot:
            if plot_output:
                with plot_output:
                    plot_output.clear_output()
                    plt.show()
            else:
                plt.show()
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf

        if tv_plot_output:
            with tv_plot_output:
                tv_plot_output.clear_output()
                self.plot_tv_closed_loop(title=title)

    def plot_tv_closed_loop(self, title: str = None) -> None:
        """
        Plot thermal valve closed loop test data including TV power and TV PT1000 temperature.
        Args:
            title (str): The title for the plot.
        """
        fig, ax1 = plt.subplots(figsize=(9, 7))
        color1 = 'tab:blue'
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel(f'TV Power [{self.units[FMSFlowTestParameters.AVG_TV_POWER.value]}]', color=color1)
        ax1.plot(self.logtime, self.tv_power, color=color1, label='TV Power')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(f'TV PT1000 Temperature [{self.units[FMSFlowTestParameters.TV_PT1000.value]}]', color=color2)
        ax2.plot(self.logtime, self.pt1000, color=color2, label='TV PT1000 Temperature')
        ax2.tick_params(axis='y', labelcolor=color2)

        if title:
            plt.title(title, wrap=True)
        fig.tight_layout()
        plt.grid(True)
        plt.show()
            
    def tvac_cycle_query(self, test_id: str, plot: bool = True) -> io.BytesIO | None:
        """
        Query and plot TVAC cycle test data for the given test ID.
        Allows interactivity by adjusting the time range slider.
        Args:
            test_id (str): The ID of the TVAC cycle test to query.
            plot (bool): Whether to display the plot or return it as a BytesIO object.
        Returns:
            BytesIO: The plot image in a BytesIO object if plot is False.
        """
        test_results: list[FMSTvac] = [tr for tr in self.fms_entry.tvac_results]
        if not test_results:
            print("Test ID not found for this FMS.")
            return
        if len(test_results) > 1:
            test_results = sorted(test_results, key=lambda x: x.date)
            time = []
            trp1 = []
            trp2 = []
            for tr in test_results:
                time.extend(tr.logtime)
                trp1.extend(tr.trp1)
                trp2.extend(tr.trp2)
        else:
            tr = test_results[0]
            time = tr.logtime
            trp1 = tr.trp1
            trp2 = tr.trp2

        time_hours = np.array(time) / 3600.0 
        trp1 = np.array(trp1)
        trp2 = np.array(trp2)

        # Sort time and associated arrays
        sort_idx = np.argsort(time_hours)
        time_hours = time_hours[sort_idx]
        trp1 = trp1[sort_idx]
        trp2 = trp2[sort_idx]

        slider = widgets.IntRangeSlider(
            value=[int(min(time_hours)), int(max(time_hours))],
            min=int(min(time_hours)),
            max=int(max(time_hours)),
            step=1,
            description='Logtime Range [hrs]:',
            continuous_update=True,
            style={'description_width': '200px'}, 
            layout={'width': '600px'}             
        )
        output = widgets.Output()

        def update_plot(change: dict) -> None:
            with output:
                output.clear_output(wait=True)
                plt.figure(figsize=(9, 7))
                mask = (time_hours >= change['new'][0]) & (time_hours <= change['new'][1])
                plt.plot(time_hours[mask], trp1[mask], label='TRP1', color='blue')
                plt.plot(time_hours[mask], trp2[mask], label='TRP2', color='orange')
                plt.xlabel('Time [hrs]')
                plt.ylabel('Temperature [degC]')
                plt.title(f'TVAC Acceptance Cycles LP FMS, SN: {self.fms_entry.fms_id}, Pvac < 1E-1 mbar, MLI')
                plt.legend()
                plt.grid()
                plt.show()

        if plot:
            slider.observe(update_plot, names='value')
            ui = widgets.VBox([slider, output])
            display(ui)
            update_plot({'new': slider.value})
        else:
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.plot(time_hours, trp1, label='TRP1', color='blue')
            ax.plot(time_hours, trp2, label='TRP2', color='orange')
            ax.set_xlabel('Time [hrs]')
            ax.set_ylabel('Temperature [degC]')
            ax.set_title(f'TVAC Acceptance Cycles LP FMS, SN: {self.fms_entry.fms_id}, Pvac < 1E-1 mbar, MLI')
            ax.legend()
            ax.grid()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf

    def fms_characteristics_query(self) -> None:
        """
        Query and display the FMS characteristics including power budgets, envelope, and other parameters.
        """
        test_results: list[FMSTestResults] = self.fms_entry.test_results
        if not test_results:
            print("No test results found for this FMS.")
            return None

        df = pd.DataFrame([{
            'parameter_name': res.parameter_name,
            'parameter_value': res.parameter_value,
            'parameter_json': res.parameter_json,
            'parameter_unit': res.parameter_unit,
            'larger': res.larger,
            'lower': res.lower,
            'equal': res.equal,
            'within_limits': res.within_limits
        } for res in test_results])

        def get_entry(parameter_name: str) -> dict:
            entry = df[df['parameter_name'] == parameter_name]
            if entry.empty:
                return {}
            row = entry.iloc[0]
            return {
                'parameter_name': row['parameter_name'],
                'parameter_value': row['parameter_value'],
                'parameter_json': row['parameter_json'],
                'parameter_unit': row['parameter_unit'],
                'larger': row['larger'],
                'lower': row['lower'],
                'equal': row['equal'],
                'within_limits': row['within_limits']
            }

        # --- Power Budgets ---
        hot_power_budget = get_entry(FMSMainParameters.POWER_BUDGET_HOT.value)
        cold_power_budget = get_entry(FMSMainParameters.POWER_BUDGET_COLD.value)
        room_power_budget = get_entry(FMSMainParameters.POWER_BUDGET_ROOM.value)

        power_budget_data = [
            {"Temperature": t, **({key:f"{value:.3f}" for key, value in pb.get("parameter_json", {}).items()})} 
            for pb, t in zip([hot_power_budget, cold_power_budget, room_power_budget],
                            ['Hot', 'Cold', 'Room']) if pb.get("parameter_json", None) is not None 
        ]
        if not power_budget_data:
            power_budget_data = [{}]

        power_budget_df = pd.DataFrame(power_budget_data)
        display(widgets.HTML("<h3>FMS Power Budgets [W]</h3>"))
        display(power_budget_df.style.set_table_attributes("style='display:inline'"))

        display(widgets.HTML("<h3>FMS Envelope [mm]</h3>"))

        inlet_location = get_entry(FMSMainParameters.INLET_LOCATION.value)
        outlet_anode = get_entry(FMSMainParameters.OUTLET_ANODE.value)
        outlet_cathode = get_entry(FMSMainParameters.OUTLET_CATHODE.value)
        fms_envelope = get_entry(FMSMainParameters.FMS_ENVELOPE.value)
        envelope_data = [
            {"Parameter": j, "x": f"{entry.get("parameter_json", [])[0]:.2f}", "y": f"{entry.get("parameter_json", [])[1]:.2f}", "z": f"{entry.get("parameter_json", [])[2]:.2f}"} 
            for entry, j in zip([inlet_location, outlet_anode, outlet_cathode, fms_envelope],   
                        [FMSMainParameters.INLET_LOCATION.value,
                      FMSMainParameters.OUTLET_ANODE.value,
                      FMSMainParameters.OUTLET_CATHODE.value,
                      FMSMainParameters.FMS_ENVELOPE.value])
        ]
        envelope_df = pd.DataFrame(envelope_data)
        display(envelope_df.style.set_table_attributes("style='display:inline'"))

        display(widgets.HTML("<h3>FMS Characteristics</h3>"))
        other_params = []
        for param in FMSMainParameters:
            if param in [FMSMainParameters.POWER_BUDGET_HOT,
                         FMSMainParameters.POWER_BUDGET_COLD,
                         FMSMainParameters.POWER_BUDGET_ROOM,
                         FMSMainParameters.SERIAL_NUMBER,
                         FMSMainParameters.OUTLET_ANODE,
                         FMSMainParameters.OUTLET_CATHODE,
                         FMSMainParameters.FMS_ENVELOPE,
                         FMSMainParameters.INLET_LOCATION
                         ]:
                continue
            entry = get_entry(param.value)
            if entry:
                value = entry["parameter_value"]
                equal = entry["equal"]
                if not equal:
                    larger = entry.get("larger", False)
                    lower = entry.get("lower", False)
                    if larger:
                        value = f"> {value}"
                    elif lower:
                        value = f"< {value}"
                other_params.append({
                    "Parameter": param.value.title(),
                    "Value": value,
                    "Unit": entry["parameter_unit"],
                    "Within Limits": entry["within_limits"]
                })

        other_params_df = pd.DataFrame(other_params)
        n = len(other_params_df)
        split_idx1 = n // 3
        split_idx2 = 2 * n // 3

        df_left = other_params_df.iloc[:split_idx1].reset_index(drop=True)
        df_middle = other_params_df.iloc[split_idx1:split_idx2].reset_index(drop=True)
        df_right = other_params_df.iloc[split_idx2:].reset_index(drop=True)

        def style_cell(val: Any, within: LimitStatus) -> str:
            if within == LimitStatus.TRUE:
                return ''
            elif within == LimitStatus.FALSE:
                return 'color: red; font-weight: bold;'
            elif within == LimitStatus.ON_LIMIT:
                return 'color: orange; font-weight: bold;'
            return ''

        def format_value(val: Any) -> Any:
            if isinstance(val, (int, float)):
                if abs(val) < 1e-4 or abs(val) > 1e5:
                    return f"{val:.3e}"
                return f"{val:.3f}"
            return val

        for df_ in [df_left, df_middle, df_right]:
            df_['Value'] = df_['Value'].apply(format_value)

        def style_df(df: pd.DataFrame) -> pd.io.formats.style.Styler:
            df_styling = df.drop(columns=['Within Limits'])
            return df_styling.style.apply(
                lambda col: [style_cell(v, w) if col.name == 'Value' else '' 
                            for v, w in zip(col, df['Within Limits'])],
                axis=0
            ).set_table_attributes("style='display:inline; margin-right:20px;'")

        styled_left = style_df(df_left)
        styled_middle = style_df(df_middle)
        styled_right = style_df(df_right)
        styled_right.set_table_attributes("style='display:inline;'")

        display(widgets.HTML(f"<div style='display:flex'>{styled_left.to_html()}{styled_middle.to_html()}{styled_right.to_html()}</div>"))

    def fms_comparison_query(self, comparison_type: str) -> None:
        """
        Perform FMS comparison analysis based on the selected comparison type.
        Args:
            comparison_type (str): The type of comparison analysis to perform.
        """
        all_main_results = self.session.query(FMSMain).all()
        all_fr_tests = self.session.query(FMSFRTests).all()
        all_functional_tests = self.session.query(FMSFunctionalTests).all()
        grouped_functional_tests = defaultdict(lambda: defaultdict(list))
        for test in all_functional_tests:
            grouped_functional_tests[test.test_type][test.temp_type].append(test)
        if comparison_type == "TV Slope Analysis":
            self.tv_slope_analysis(grouped_functional_tests)
        elif comparison_type == "Flow Rate Analysis":
            self.flow_rate_analysis(all_fr_tests)
        elif comparison_type == "Closed Loop Analysis":
            self.closed_loop_analysis(grouped_functional_tests)
        else:
            self.free_fms_analysis(all_main_results)

    def free_fms_analysis(self, all_main_results: list[FMSTestResults]) -> None:
        """
        Perform free FMS characteristic analysis by allowing user to select any two characteristics.
        Displays interactive dropdowns for parameter selection and plots the results.
        Args:
            all_main_results (list[FMSTestResults]): List of all FMS main entries.
        """
        characteristic_1 = widgets.Dropdown(
            options=[i.value for i in FMSMainParameters if not i == FMSMainParameters.POWER_BUDGET_COLD and not \
                     i == FMSMainParameters.SERIAL_NUMBER and not i == FMSMainParameters.POWER_BUDGET_HOT and not i == FMSMainParameters.POWER_BUDGET_ROOM],
            description='Choose Parameter:',
            layout=widgets.Layout(width='350px'),
            style={'description_width': '150px'},
            value=FMSMainParameters.TV_HIGH_LEAK.value      
        )

        characteristic_2 = widgets.Dropdown(
            options=[i.value for i in FMSMainParameters if not i == characteristic_1.value and not i == FMSMainParameters.SERIAL_NUMBER and not\
                      i == FMSMainParameters.POWER_BUDGET_COLD and not i == FMSMainParameters.POWER_BUDGET_HOT and not i == FMSMainParameters.POWER_BUDGET_ROOM],
            description='Choose Parameter:',
            layout=widgets.Layout(width='350px'),
            style={'description_width': '150px'},
            value=FMSMainParameters.TV_RESISTANCE.value     
        )

        output = widgets.Output()

        def on_char2_change(change):
            characteristic_1.options = [i.value for i in FMSMainParameters if not i == characteristic_2.value and not i == FMSMainParameters.SERIAL_NUMBER and not\
                                         i == FMSMainParameters.POWER_BUDGET_COLD and not i == FMSMainParameters.POWER_BUDGET_HOT and not i == FMSMainParameters.POWER_BUDGET_ROOM]
            if change['new'] != change['old']:
                parameter1 = characteristic_1.value
                parameter2 = characteristic_2.value
                with output:
                    output.clear_output()
                    self.plot_characteristics(parameter1, parameter2, all_main_results)

        def on_char1_change(change):
            if change['new'] != change['old']:
                parameter1 = characteristic_1.value
                parameter2 = characteristic_2.value
                with output:
                    output.clear_output()
                    self.plot_characteristics(parameter1, parameter2, all_main_results)

        characteristic_2.observe(on_char2_change, names='value')
        characteristic_1.observe(on_char1_change, names='value')

        form = widgets.VBox([
            widgets.HTML("<h3>Select Two Characteristics to Perform the Trend Analysis</h3>"),
            characteristic_1,
            characteristic_2,
            output
        ])

        display(form)

        with output:
            output.clear_output()
            parameter1 = characteristic_1.value
            parameter2 = characteristic_2.value
            self.plot_characteristics(parameter1, parameter2, all_main_results)

    def plot_characteristics(self, parameter1: str, parameter2: str, all_main_results: list[FMSTestResults]) -> None:
        """
        Plot the selected characteristics against each other for FMS comparison analysis.
        Args:
            parameter1 (str): The first characteristic parameter name.
            parameter2 (str): The second characteristic parameter name.
            all_main_results (list[object]): List of all FMSTestResults objects, holding acceptance test results.
        """
        
        current_results: list[FMSTestResults] = self.fms_entry.test_results
        if current_results:
            parameter_check1 = next((p for p in current_results if p.parameter_name == parameter1), None)
            parameter_check2 = next((p for p in current_results if p.parameter_name == parameter2), None)
            if not parameter_check1 or not parameter_check2:
                print("Selected parameters not found in current entry.")
                return
        
        if not parameter_check1 or not parameter_check2:
            print("Selected parameters not found in current entry.")
            return

        parameter1_value = parameter_check1.parameter_value
        unit1 = parameter_check1.parameter_unit
        within_limits1 = parameter_check1.within_limits

        parameter2_value = parameter_check2.parameter_value
        unit2 = parameter_check2.parameter_unit
        within_limits2 = parameter_check2.within_limits

        # Align all_param1 and all_param2 by self.fms_entry.id
        aligned_param1 = []
        aligned_param2 = []
        missing_param1_ids = []
        missing_param2_ids = []

        for entry in all_main_results:
            res1: FMSTestResults | None = next((r for r in entry.test_results if r.parameter_name == parameter1), None)
            res2: FMSTestResults | None = next((r for r in entry.test_results if r.parameter_name == parameter2), None)

            if res1:
                aligned_param1.append(res1.parameter_value)
            else:
                aligned_param1.append(None)
                missing_param1_ids.append(entry.fms_id)

            if res2:
                aligned_param2.append(res2.parameter_value)
            else:
                aligned_param2.append(None)
                missing_param2_ids.append(entry.fms_id)

        print(f"Aligned lengths: {len(aligned_param1)}, {len(aligned_param2)}")
        if missing_param1_ids:
            print(f"Entries missing {parameter1}: {missing_param1_ids}")
        if missing_param2_ids:
            print(f"Entries missing {parameter2}: {missing_param2_ids}")

        plt.figure(figsize=(10, 7))
        # Use aligned lists for plotting; matplotlib ignores None values
        plt.scatter(aligned_param1, aligned_param2, alpha=0.6)
        plt.scatter([parameter1_value], [parameter2_value],
                    color='red',
                    label=f'FMS ID: {self.fms_entry.fms_id}: ({parameter1_value}, {parameter2_value})',
                    edgecolors='black')
        plt.title(f'FMS Characteristic Analysis - {parameter1} vs {parameter2}\n'
                f'FMS SN: {self.fms_entry.fms_id}, {parameter1} Limit: {within_limits1}, {parameter2} Limit: {within_limits2}',
                wrap=True)
        plt.xlabel(f'{parameter1} ({unit1})')
        plt.ylabel(f'{parameter2} ({unit2})')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        plt.grid()
        plt.show()

    def flow_rate_analysis(self, all_fr_tests: list[FMSFRTests]) -> None:
        """
        Perform flow rate analysis by plotting flow rate slopes against anode/cathode ratios.
        Args:
            all_fr_tests (list[FMSFRTests]): List of all FMS flow rate test entries.
        """
        slopes = []
        ratios = []
        current_slope = None
        current_ratio = None
        min_slope = get_slope(self.lpt_voltages, self.min_flow_rates)
        max_slope = get_slope(self.lpt_voltages, self.max_flow_rates)
        current_test: FMSFRTests = self.fms_entry.fr_tests[0] if self.fms_entry.fr_tests else None
        temperature = current_test.trp_temp if current_test else None
        inlet_pressure = current_test.inlet_pressure if current_test else None
        inlet_pressure = 10 if inlet_pressure < 100 else 190
        outlet_pressure = current_test.outlet_pressure if current_test else None
        if len(all_fr_tests) == 0:
            print("No FR tests found in the database.")
            return
        for test in all_fr_tests:
            fms_main: FMSMain = test.fms_main
            fms_id = test.fms_id
            if fms_main:
                manifold: ManifoldStatus = fms_main.manifold[0] if fms_main.manifold else None
                if manifold:
                    ratio = manifold.ac_ratio 
                else:
                    ratio = 13
            else:
                ratio = 13
            lpt_voltage = test.lpt_voltage
            total_flow = test.total_flow
            intersections = find_intersections(lpt_voltage, total_flow, self.lpt_voltages, self.min_flow_rates, self.max_flow_rates)
            if fms_id == self.fms_entry.fms_id:
                current_slope = intersections['flow_slope']
                current_ratio = ratio
                continue
            slopes.append(intersections['flow_slope'])
            ratios.append(ratio)

        plt.figure(figsize=(10, 7))
        plt.scatter(ratios, slopes, alpha=0.6)
        plt.scatter([current_ratio], [current_slope], color='red', label=f'FMS ID: {self.fms_entry.fms_id}: ({current_ratio:.1f}, {current_slope:.3f})', edgecolors='black')
        plt.axhline(y=min_slope, color='red', linestyle='--', label='Slope Specification')
        plt.axhline(y=max_slope, color='red', linestyle='--')
        plt.title(f'FMS Flow Rate Analysis - Slope vs Anode/Cathode Ratio\nFMS SN: {self.fms_entry.fms_id}, Inlet Pressure: {inlet_pressure} [barA], \n Outlet Pressure: {outlet_pressure} [mbar], TRP Temp at {temperature} [degC]', wrap=True)
        plt.xlabel('Anode/Cathode Ratio')
        plt.ylabel('Flow Rate Slope [mg/s mV^-1]')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.grid(True)
        plt.show()
        
    def closed_loop_analysis(self, grouped_functional_tests: dict[str, dict[str, list[FMSFunctionalTests]]]) -> None:
        fms_id = self.fms_entry.fms_id
        if not grouped_functional_tests:
            print("No functional tests found in the database.")
            return

        output = widgets.Output()
        test_info = []
        first_hot_date = next((t.date for type, temp_groups in grouped_functional_tests.items() for temp_type, tests in temp_groups.items() for t in tests if "hot" in temp_type.value and t.fms_id == fms_id), None)
        for test_type, temp_groups in grouped_functional_tests.items():
            if "closed_loop" not in test_type.value:
                continue

            for temp_type, tests in temp_groups.items():
                # if len(test_info) > 0:
                #     break
                for test in tests:
                    if test.fms_id != fms_id:
                        continue
                    temperature = test.trp_temp
                    inlet_pressure = 10 if test.inlet_pressure < 100 else 190
                    response_times = test.response_times
                    date = test.date
                    if first_hot_date and date < first_hot_date and "room" in temp_type.value:
                        label_temp = f"{temperature:.0f} (Pre Vibration)"
                    else:
                        label_temp = f"{temperature:.0f}"

                    row = {"Temperature [degC]": label_temp, 
                            "Inlet Pressure [barA]": inlet_pressure,
                            "Description": "Opening Time"}
                    
                    opening_times = response_times.get("opening_time", [])
                    for i, tau in enumerate(opening_times[:3]):
                        row[f"{i+1}_tau [s]"] = tau
                    row["Actual Time [s]"] = opening_times[-1] if opening_times else "N/A"
                    test_info.append(row)
                    
                    for set_idx, set_point in enumerate(self.lpt_set_points):
                        row = {"Temperature [degC]": label_temp,
                                "Inlet Pressure [barA]": inlet_pressure}

                        if set_idx == 0:
                            key = f"response_time_to_{set_point}_barA"
                        elif set_idx == len(self.lpt_set_points) - 1:
                            key = f"closing_time_to_{set_point}_barA"
                        else:
                            key = f"response_{self.lpt_set_points[set_idx-1]}_to_{set_point}_barA"

                        description = key.replace("_", " ").title().replace('Bara', 'barA')
                        row["Description"] = description
                        tau_list = response_times.get(key, [])
                        for i, tau in enumerate(tau_list[:3]):
                            row[f"{i+1}_tau [s]"] = tau
                        row["Actual Time [s]"] = tau_list[-1] if tau_list else "N/A"
                        if not all(t == np.nan for t in tau_list):
                            test_info.append(row)
                        else:
                            tau_list = ["N/A" for _ in range(len(tau_list))]
                            for i, tau in enumerate(tau_list[:3]):
                                row[f"{i+1}_tau [s]"] = tau
                            row["Actual Time [s]"] = "N/A"
                            test_info.append(row)

        display(output)
        df = pd.DataFrame(test_info)
        df = df.fillna("N/A")
        tau_cols = [col for col in df.columns if "_tau" in col]

        def color_cells(val, desc):
            if val == "N/A" or pd.isna(val):
                return ''
            if "Opening Time" in desc:
                if val < 300:
                    return 'background-color: green'
                elif val == 300:
                    return 'background-color: orange'
                else:
                    return 'background-color: red'
            else:
                if val < 60:
                    return 'background-color: green'
                elif val == 60:
                    return 'background-color: orange'
                else:
                    return 'background-color: red'

        def format_numeric(val):
            return f"{val:.1f}" if isinstance(val, (int, float)) else val
        
        format_dict = {
            **{col: format_numeric for col in tau_cols},
            "Inlet Pressure [barA]": "{:.0f}",
            "Actual Time [s]": format_numeric
        }
        styled_df = df.style.apply(
            lambda row: [color_cells(v, df.loc[row.name, "Description"]) for v in row],
            axis=1,
            subset=tau_cols
        ).format(format_dict).hide(axis='index')

        display(styled_df)

    def tv_slope_analysis(self, grouped_functional_tests: dict[str, dict[str, list[FMSFunctionalTests]]]) -> None:
        """
        Perform TV slope analysis by plotting slope 1-2 vs slope 2-4 for different test types and temperatures.
        Compares against specification ranges and highlights the current FMS entry.
        Args:
            grouped_functional_tests (dict[str, dict[str, list[FMSFunctionalTests]]]): 
                Grouped functional test entries by test type, pressure and temperature.
        """
        fms_id = self.fms_entry.fms_id

        if len(grouped_functional_tests) == 0:
            print("No functional tests found in the database.")
            return

        output = widgets.Output()
        correction_checkbox = widgets.Checkbox(
            value = False,
            description = 'Show Inlet Pressure Correction:',
            indent = False,
            label_width = '150px'
        )
        first_hot_date = next((t.date for type, temp_groups in grouped_functional_tests.items() for temp_type, tests in temp_groups.items() for t in tests if "hot" in temp_type.value and t.fms_id == fms_id), None)
        
        def on_correction_change(change: dict):
            correction = correction_checkbox.value
            correction_suffix = " (Corrected)" if correction else ""
            plot_dict = {}
            table_rows = []
            with output:
                output.clear_output()
                for test_type, temp_groups in grouped_functional_tests.items():
                    if "slope" not in test_type.value:
                        continue
                    
                    for temp_type, tests in temp_groups.items():
                        slope12s = []
                        slope24s = []
                        min_range_12 = self.range12_low[0] if "low" in test_type.value else self.range12_high[0]
                        min_range_24 = self.range24_low[0] if "low" in test_type.value else self.range24_high[0]
                        max_range_12 = self.range12_low[1] if "low" in test_type.value else self.range12_high[1]
                        max_range_24 = self.range24_low[1] if "low" in test_type.value else self.range24_high[1]
                        current_slope12 = []
                        current_slope24 = []


                        for test in tests:
                            gas_type = self.fms_entry.gas_type if self.fms_entry else "Xe"
                            temperature = test.trp_temp
                            inlet_pressure = test.inlet_pressure
                            slope12 = test.slope12
                            slope24 = test.slope24
                            correction_factor = test.slope_correction
                            date = test.date

                            if slope12 is None or slope24 is None:
                                continue

                            if test.fms_id == fms_id:
                                label_temp = f"{temperature:.0f}"
                                if "room" in temp_type.value and first_hot_date and date < first_hot_date:
                                    label_temp = f"{temperature:.0f} (Pre Vibration)"
                                    first_room = False
                                
                                table_rows.append({
                                "Temperature [degC]": label_temp,
                                "Inlet Pressure [barA]": 10 if inlet_pressure < 100 else 190,
                                f"Slope 1-2 [mg/s W⁻¹ {gas_type}]": slope12/(correction_factor if correction else 1),
                                f"Slope 2-4 [mg/s W⁻¹ {gas_type}]": slope24/(correction_factor if correction else 1),
                                })

                                current_slope12.append(slope12/(correction_factor if correction else 1))
                                current_slope24.append(slope24/(correction_factor if correction else 1))
                            else:
                                slope12s.append(slope12)
                                slope24s.append(slope24)

                        if test_type not in plot_dict:
                            plot_dict[test_type] = {}

                        inlet_pressure_adj = 190 if inlet_pressure >= 100 else 10

                        plot_dict[test_type][temp_type] = {
                            "slope12s": slope12s,
                            "slope24s": slope24s,
                            "min_range_12": min_range_12,
                            "max_range_12": max_range_12,
                            "min_range_24": min_range_24,
                            "max_range_24": max_range_24,
                            "current_slope12": current_slope12,
                            "current_slope24": current_slope24,
                            "title": (
                                f"FMS SN: {self.fms_entry.fms_id} {gas_type}, "
                                f"Inlet Pressure: {inlet_pressure_adj} [barA],\n "
                                f"TRP Temp: {temperature} [degC]{correction_suffix}"
                            ),
                        }

                df = pd.DataFrame(table_rows)
                slope_cols = [f"Slope 1-2 [mg/s W⁻¹ {gas_type}]", f"Slope 2-4 [mg/s W⁻¹ {gas_type}]"]

                title = widgets.HTML(f"<h3>Slope Overview for FMS {self.fms_entry.fms_id}{correction_suffix}</h3>")
                display(title)
                # Round slope columns for display
                df[slope_cols] = df[slope_cols].round(1)

                def style_slopes(val12, val24, pressure):
                    styles = []
                    if pressure == 10:
                        min_val_12, max_val_12 = self.range12_low
                        min_val_24, max_val_24 = self.range24_low
                    else:
                        min_val_12, max_val_12 = self.range12_high
                        min_val_24, max_val_24 = self.range24_high
                    for val, min_val, max_val in [(val12, min_val_12, max_val_12), (val24, min_val_24, max_val_24)]:
                        styles.append("background-color: green" if min_val <= val <= max_val else "background-color: red")
                    
                    return styles

                # Apply only to slope columns
                styled_df = df.style.apply(
                    lambda row: [style_slopes(df.loc[row.name, f"Slope 1-2 [mg/s W⁻¹ {gas_type}]"], df.loc[row.name, f"Slope 2-4 [mg/s W⁻¹ {gas_type}]"], df.loc[row.name, "Inlet Pressure [barA]"])[i] for i in range(2)],
                    axis=1,
                    subset=slope_cols
                ).format({col: "{:.1f}" for col in slope_cols})

                display(styled_df)

                n_cols = 2
                n_rows = (sum(len(v) for v in plot_dict.values()) + 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), constrained_layout=True)
                axes = axes.flatten()

                items = [(tt, tp, d) for tt, sub in plot_dict.items() for tp, d in sub.items()]
                handles, labels = [], []

                for ax, (test_type, temp_type, data) in zip(axes, items):
                    slope12s = data["slope12s"]
                    slope24s = data["slope24s"]
                    current_slope12 = data["current_slope12"]
                    current_slope24 = data["current_slope24"]
                    min_range_12 = data["min_range_12"]
                    max_range_12 = data["max_range_12"]
                    min_range_24 = data["min_range_24"]
                    max_range_24 = data["max_range_24"]

                    ax.scatter(slope12s, slope24s, alpha=0.7)

                    ax.scatter(
                        current_slope12,
                        current_slope24,
                        color="red",
                        edgecolors="black",
                        label=f"FMS SN: {fms_id}",
                    )

                    ax.axvline(min_range_12, color="red", linestyle="-", label="Min 1-2 mg/s Spec")
                    ax.axvline(max_range_12, color="red", linestyle="--", label="Max 1-2 mg/s Spec")
                    ax.axhline(min_range_24, color="orange", linestyle="-", label="Min 2-4 mg/s Spec")
                    ax.axhline(max_range_24, color="orange", linestyle="--", label="Max 2-4 mg/s Spec")

                    ax.set_title(data["title"], wrap=True)
                    ax.set_xlabel(f"TV Slope 1-2 [mg/s W⁻¹ {gas_type}]")
                    ax.set_ylabel(f"TV Slope 2-4 [mg/s W⁻¹ {gas_type}]")
                    ax.grid(True)

                    for h, l in zip(*ax.get_legend_handles_labels()):
                        if l not in labels:
                            handles.append(h)
                            labels.append(l)

                fig.legend(
                    handles, labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.04),
                    ncol=3,
                    frameon=False
                )

                plt.show()
        correction_checkbox.observe(on_correction_change, names='value')
        display(correction_checkbox)
        display(output)
        on_correction_change({})

    def part_investigation_query(self, part_type: str) -> None:
        """
        Query and display part investigation data based on the selected part type.
        """
        if part_type == "Thermal Valve":
            tv_query = TVQuery(session=self.session, fms_entry=self.fms_entry)
            tv_query.tv_query_field()

        elif part_type == "Manifold":
            manifold_query = ManifoldQuery(session=self.session, fms_entry=self.fms_entry)
            manifold_query.manifold_query_field()
        
        elif part_type == "HPIV":
            hpiv_query = HPIVQuery(session=self.session, fms_entry=self.fms_entry)
            hpiv_query.hpiv_query_field()
