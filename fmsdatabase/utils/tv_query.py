# Future imports
from __future__ import annotations

# Standard library
from typing import TYPE_CHECKING
from collections import defaultdict

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from scipy.stats import norm
from sqlalchemy import func
from tqdm.notebook import tqdm

# Local imports
from .general_utils import (
    display_df_in_chunks,
    plot_distribution,
    plot_simulated_distribution,
    TVTestParameters, 
    TVParts
)
from ..db import (
    TVTestResults,
    TVTestRuns,
    TVStatus,
    FMSMain,
    TVCertification,
    TVTvac,
)

# Type-checking imports
if TYPE_CHECKING:
    from sqlalchemy.orm import Session

#TODO weld gap analysis, ask for measurements of SK tech

class TVQuery:
    """
    Base class for querying thermal valve data from the database and generating interactive
    visualizations using ipywidgets. Provides a UI for selecting TVs, querying different types of data,
    and plotting results such as flow tests, trend analyses, part investigations, and TVAC analyses.

    Attributes
    ----------
    session : Session
        SQLAlchemy session for database interaction.
    fms_entry : FMSMain
        FMS entry associated with the thermal valve.
    allocated_certifications : list
        List of certifications of TVs that have been allocated to an FMS.
    tv_test_dict : dict
        Dictionary mapping test descriptions to test references.
    tvac_runs : list[TVTvac]
        List of TVAC test runs for the selected thermal valve.
    tv_part_dict : dict
        Dictionary mapping part descriptions to part names.
    tv_id : int
        ID of the selected thermal valve.
    tv : TVStatus
        ORM instance representing the selected thermal valve.
    test_runs : list[TVTestRuns]
        List of test runs for the selected thermal valve.
    test_reference : str
        Reference of the selected test run.
    actions : list[str]
        List of available query actions for the selected thermal valve.
    value : str
        Currently selected query action.
    welded_map : dict
        Mapping of welded boolean values to descriptive strings.
    all_tvs : list[TVStatus]
        List of all thermal valves in the database.
    all_allocated : list[TVCertification]
        List of all allocated thermal valve certifications in the database.

    Methods
    -------
    tv_query_field():
        Create interactive widgets for querying thermal valve data.
    plot_flow_test():
        Plot flow test results for the selected test reference.
    plot_all_flow_tests():
        Plot all flow test results for the selected thermal valve.
    part_investigation(part_name):
        Perform part investigation for the specified part name.
    measurement_analysis():
        Perform measurement trend analysis for thermal valve parts.
    get_trend(certification, part_name, current_cert):
        Retrieve and plot trend data for the specified certification and part name.
    plot_dimension_trend(dimension_dict, opening_temps, part_name, current_dims, certification, current_certification):
        Plot dimension trend analysis for thermal valve parts.
    tv_remark_field():
        Create an input field for adding or changing thermal valve remarks.
    tv_status():
        Display the status and certifications of the selected thermal valve.
    weld_analysis():
        Perform weld analysis comparing pre-welded and welded thermal valves.
    statistical_analysis(pre_welded_runs, welded_runs, opening_temp, ranges=None):
        Perform statistical analysis of weld data and plot results.
    weld_analysis_per_tv(pre_welded_runs, welded_runs, ranges):
        Perform weld analysis per thermal valve and plot results.
    revisional_analysis(pre_welded_runs, welded_runs):
        Perform revisional analysis of weld data and plot results.
    count_cycles(data, threshold):
        Count the number of cycles of a TVAC test on a certain TV based on a threshold.
    tvac_analysis():
        Perform TVAC analysis for the selected test reference.
    plot_all_tvac_tests():
        Plot all TVAC test results for the selected thermal valve.
    plot_tvac_analysis():
        Plot TVAC analysis results for the selected test reference.
    get_certifications():
        Display all certifications and corresponding data of parts that are relevant to the TV.
    """

    def __init__(self, session: "Session", fms_entry: FMSMain = None):
        
        self.session = session
        self.fms_entry: type[FMSMain] = fms_entry
        self.allocated_certifications = []
        self.tv_test_dict = {}
        self.tvac_runs: list[TVTvac] = []
        self.tv_part_dict = {}
        self.tv_id = None
        self.tv = None
        self.test_runs = None
        self.test_reference = None
        if self.fms_entry:
            tv: TVStatus = self.fms_entry.thermal_valve[0] if self.fms_entry.thermal_valve else None
            self.tv = tv if tv else None
            self.tv_from_fms = self.fms_entry.tv_id
            if not self.tv and not self.tv_from_fms:
                print("No TV ID associated with this FMS entry.")
                return
            if not self.tv:
                self.tv_id = self.tv_from_fms
                print(f"No TV status available for thermal valve {self.tv_from_fms}")
                self.test_runs = self.session.query(TVTestRuns).filter_by(tv_id=self.tv_from_fms).all()
                self.tvac_runs = self.session.query(TVTvac).filter_by(tv_id=self.tv_from_fms).all()
                if not self.test_runs:
                    print(f"No test runs found for thermal valve {self.tv_from_fms}")
                else:
                    self.value = 'Flow Test'
                    self.actions = ['Flow Test', 'Certifications']
            else:
                self.tv_id = self.tv.tv_id
                self.test_runs = self.tv.test_runs
                self.tvac_runs = self.tv.tvac
                self.value = 'Status'
                self.actions = ['Status', 'Flow Test', 'Trend Analysis', 'Part Investigation', 'Certifications', 'TVAC Analysis']
            self.certifications: list[TVCertification] = self.tv.certifications if self.tv else []

        else:
            self.value = 'Status'
            self.actions = ['Status', 'Flow Test', 'Trend Analysis', 'Part Investigation', 'Certifications', 'TVAC Analysis']

        self.welded_map = {True: 'Welded', False: 'Pre-weld'}
        self.all_tvs = self.session.query(TVStatus).all()

        self.all_allocated = (
            self.session.query(TVCertification)
            .filter(TVCertification.tv_id != None)
            .all()
        )
        
    def tv_query_field(self) -> None:
        """
        Create interactive widgets for querying thermal valve data.
        """
        if not self.tv_id and self.fms_entry:
            return

        tv_field = widgets.Dropdown(
            description='TV ID:',
            layout=widgets.Layout(width='350px'),
            style={'description_width': '80px'},
            value=self.tv_id,
            disabled=True if self.tv_id else False,
            options=[self.tv_id] if self.tv_id else [tv.tv_id for tv in self.all_tvs]
        )

        query_field = widgets.Dropdown(
            options=self.actions if self.tv_id else ["Trend Analysis", "Part Investigation", "Certifications"],
            description='Select Query:',
            layout=widgets.Layout(width='350px'),
            style={'description_width': '80px'},
            value=self.value if self.tv_id else None
        )

        dynamic_field = widgets.Dropdown(
            options=[],
            description='Select Action:',
            layout=widgets.Layout(width='350px'),
            style={'description_width': '80px'}
        )

        output = widgets.Output()

        def get_dynamic_callable(test_type: str) -> callable | None:
            if dynamic_field.value is None and test_type != 'Status':
                return None
            elif test_type == 'Flow Test':
                self.test_reference = self.tv_test_dict.get(dynamic_field.value, None) if not dynamic_field.value == 'all' else 'all'
                return self.plot_flow_test
            elif test_type == 'Part Investigation':
                return lambda: self.part_investigation(dynamic_field.value)
            elif test_type == 'Trend Analysis':
                if dynamic_field.value == 'Measurement Analysis':
                    return lambda: self.measurement_analysis()
                elif dynamic_field.value == 'Weld Analysis':
                    return lambda: self.weld_analysis()

            elif test_type == 'TVAC Analysis':
                self.test_reference = self.tv_test_dict.get(dynamic_field.value, None) if not dynamic_field.value == 'all' else 'all'
                return self.tvac_analysis
            else:
                return None

        def on_tv_field_change(change: dict) -> None:
            if change['new'] != change['old']:
                self.tv_id = change['new']
                self.tv = next((tv for tv in self.all_tvs if tv.tv_id == self.tv_id), None)
                if not self.tv:
                    print(f"No TV status available for thermal valve {self.tv_id}")
                self.test_runs = self.session.query(TVTestRuns).filter_by(tv_id=self.tv_id).all()
                if not self.test_runs:
                    print(f"No test runs found for thermal valve {self.tv_id}")
                self.certifications = self.tv.certifications if self.tv else []
                self.all_allocated = (
                    self.session.query(TVCertification)
                    .filter(TVCertification.tv_id != None)
                    .all()
                )
                self.tvac_runs = self.session.query(TVTvac).filter_by(tv_id=self.tv_id).all()
                dynamic_field.options = []
                dynamic_field.description = "Select Action:"
                if self.tv:
                    self.actions = ['Status', 'Flow Test', 'Trend Analysis', 'Part Investigation', 'Certifications', 'TVAC Analysis']
                    query_field.options = self.actions
                    query_field.value = 'Status'
                    with output:
                        output.clear_output()
                        self.tv_status()
                else:
                    self.actions = []
                    query_field.options = self.actions
                    query_field.value = None

        def on_query_change(change: dict) -> None:
            choice = change['new']
            output.clear_output()
            if choice == 'Flow Test':
                headers = [f"{tr.test_reference} {self.welded_map.get(tr.welded, 'Unknown')}" for tr in self.test_runs]
                dynamic_field.description = "Select Test:" if self.test_runs else "No Flow Tests Found"
                dynamic_field.value = None
                dynamic_field.options = ['all'] + headers
                self.tv_test_dict = {header: tr.test_reference for header, tr in zip(headers, self.test_runs)}

            elif choice == 'Trend Analysis':
                dynamic_field.description = "Select Action:"
                dynamic_field.options = ["Weld Analysis", "Measurement Analysis"]

            elif choice == 'Certifications':
                with output:
                    output.clear_output()
                    self.get_certifications()
                dynamic_field.options = []

            elif choice == 'Part Investigation':
                headers = [" ".join([j.capitalize() for j in i.value.replace("thermal valve ", "").split(" ")])
                           for i in TVParts
                           if i != TVParts.WELD and i != TVParts.HOLDER_1 and i != TVParts.HOLDER_2]
                dynamic_field.options = headers
                self.tv_part_dict = {header: part for header, part in zip(headers,
                                                                          [i.value for i in TVParts if
                                                                           i != TVParts.WELD and i != TVParts.HOLDER_1 and i != TVParts.HOLDER_2])}
                dynamic_field.description = "Select Part:"

            elif choice == 'TVAC Analysis':
                dynamic_field.description = "Select Test:"
                headers = [f"{tr.test_id} - {tr.cycles} Cycles" for tr in self.tvac_runs]
                dynamic_field.options = ['all'] + headers
                self.tv_test_dict = {header: tr.test_id for header, tr in zip(headers, self.tvac_runs)}
            else:
                with output:
                    output.clear_output()
                    dynamic_field.options = []
                    self.tv_status()

        query_field.observe(on_query_change, names='value')

        def on_dynamic_change(change: dict) -> None:
            if query_field.value == 'Certifications':
                return

            func = get_dynamic_callable(query_field.value)
            if func:
                with output:
                    output.clear_output()
                    func()

        dynamic_field.observe(on_dynamic_change, names='value')
        tv_field.observe(on_tv_field_change, names='value')

        form = widgets.VBox([
            widgets.HTML('<h3>Thermal Valve Investigation</h3>'),
            tv_field,
            query_field,
            dynamic_field,
            output,
        ])

        display(form)

        if len(self.actions) > 2:
            if tv_field.value:
                with output:
                    output.clear_output()
                    self.tv_status()
        else:
            if self.test_runs:
                dynamic_field.description = "Select Test:"
                headers = [f"{tr.test_reference} {self.welded_map.get(tr.welded, 'Unknown')}" for tr in self.test_runs]
                dynamic_field.options = headers
                self.tv_test_dict = {header: tr.test_reference for header, tr in zip(headers, self.test_runs)}
                dynamic_field.value = headers[0]
                with output:
                    output.clear_output()
                    self.test_reference = self.tv_test_dict.get(dynamic_field.value, None)
                    self.plot_flow_test()

    def plot_flow_test(self) -> None:
        """
        Plot flow test results for the selected test reference.
        """
        if self.test_reference == 'all':
            self.plot_all_flow_tests()
            return
        test_run = next((tr for tr in self.test_runs if tr.test_reference == self.test_reference), None)
        opening_temp = test_run.opening_temp
        used_temp = test_run.used_temp.value
        hysteresis = test_run.hysteresis
        if not test_run:
            print("Test Reference not found for this TV.")
            return
        test_results: list[TVTestResults] = test_run.test_results
        welded = test_run.welded
        if not test_results:
            print("No test results found for this test run.")
            return
        flow_rates = [res.parameter_value for res in test_results if res.parameter_name == TVTestParameters.ANODE_FLOW.value] or None
        temperatures = [res.parameter_value for res in test_results if res.parameter_name == used_temp] or None
        plt.figure()
        plt.plot(temperatures, flow_rates)
        if opening_temp:
            plt.axvline(opening_temp, color='r', linestyle='--', label=f'Opening Temp: {opening_temp:.2f} [degC]')
            plt.legend()
        welded_state = "Welded" if welded else "Pre-weld"
        plt.ylabel('Anode Flow Rate [mg/s]')
        temp = " ".join([j.capitalize() for j in used_temp.split("_")])
        plt.xlabel(temp + ' [degC]')
        base_title = f'{temp} vs Anode Flow Rate\n For {welded_state} TV ID {self.tv_id}, Test Ref: {self.test_reference}'
        if hysteresis is not None:
            base_title += f'\nHysteresis @ 0.25 [mg/s]: {hysteresis:.2f} [degC]'
        plt.title(base_title)
        plt.grid(True)
        plt.show()

    def plot_all_flow_tests(self) -> None:
        """
        Plot all flow test results for the selected thermal valve.
        """
        if not self.test_runs:
            print("No test runs found for this TV.")
            return

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.test_runs)))

        progress = tqdm(self.test_runs, desc="Plotting Flow Tests", unit="test")

        for idx, test_run in enumerate(progress):
            test_results: list[TVTestResults] = test_run.test_results
            if not test_results:
                continue

            flow_rates = [res.parameter_value for res in test_results if res.parameter_name == TVTestParameters.ANODE_FLOW.value] or None
            used_temp = test_run.used_temp.value
            temperatures = [res.parameter_value for res in test_results if res.parameter_name == used_temp] or None
            opening_temp = test_run.opening_temp
            welded = test_run.welded

            if not flow_rates or not temperatures:
                continue

            label = f'Test Ref: {idx} ({ "Welded" if welded else "Pre-weld"}) ({opening_temp:.2f} [degC])'
            plt.plot(temperatures, flow_rates, label=label, color=colors[idx])

        plt.ylabel('Anode Flow Rate [mg/s]')
        plt.xlabel('Temperature [degC]')
        plt.title(f'All Flow Tests for TV ID {self.tv_id}')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=len(self.test_runs)//3 + 1)
        plt.show()

    def part_investigation(self, part_name: str) -> None:
        """
        Generic part investigation for any part. Handles distribution and simulated plots
        for all recorded dimensions of the part.
        """
        part_name = self.tv_part_dict.get(part_name, None)
        if self.tv:
            self.allocated_certifications = self.tv.certifications
        else:
            self.allocated_certifications = []
        part_cert = next((i for i in self.allocated_certifications if i.part_name == part_name and i.tv_id == self.tv_id), None)
        dummy = False
        if not part_cert:
            part_cert = self.session.query(TVCertification).filter(TVCertification.part_name==part_name, TVCertification.tv_id!=None).first()
            dummy = True
        if not part_cert:
            print(f"No relevant certifications found for {part_name}.")
            return
        all_parts = self.session.query(TVCertification).filter_by(part_name=part_name).all()
        if not all_parts:
            print(f"No {part_name}s found in the database.")
            return

        dimensions = part_cert.dimensions
        if not dimensions:
            print(f"No dimensions recorded for this {part_name}.")
            return

        nominal_dimensions = part_cert.nominal_dimensions if part_cert.nominal_dimensions else [None] * len(dimensions)
        min_dimensions_part = part_cert.min_dimensions if part_cert.min_dimensions else [None] * len(dimensions)
        max_dimensions_part = part_cert.max_dimensions if part_cert.max_dimensions else [None] * len(dimensions)

        for idx, dim in enumerate(dimensions):
            nominal_dim = nominal_dimensions[idx] if idx < len(nominal_dimensions) else None
            min_dim = min_dimensions_part[idx] if idx < len(min_dimensions_part) else None
            max_dim = max_dimensions_part[idx] if idx < len(max_dimensions_part) else None

            # Prepare dimension lists from all parts
            dimension_list = [p.dimensions[idx] for p in all_parts if p.dimensions and len(p.dimensions) > idx]
            min_dimensions_all = [p.min_dimensions[idx] for p in all_parts if p.min_dimensions and len(p.min_dimensions) > idx]
            max_dimensions_all = [p.max_dimensions[idx] for p in all_parts if p.max_dimensions and len(p.max_dimensions) > idx]

            if not dimension_list:
                print(f"No dimensions recorded for any {part_name} at index {idx}.")
                continue
            
            name = " ".join([j.capitalize() for j in part_name.replace("thermal valve ", "").split(" ")])
            # Plot distribution
            plot_distribution(
                array=dimension_list,
                part_name=name,
                tv_id=self.tv_id,
                nominal=nominal_dim,
                value=dim if not dummy else None,
                title=f"{name} Dimension Distribution (Dim {idx+1})",
                xlabel="Dimension [mm]",
                ylabel="Frequency",
                bins=25
            )

            # Plot simulated distribution
            if min_dim is not None and max_dim is not None:
                plot_simulated_distribution(
                    nominal=nominal_dim,
                    min_val=min_dim,
                    max_val=max_dim,
                    part_name=name,
                    tv_id=self.tv_id,
                    value=dim if not dummy else None,
                    title=f"{name} Dimension Simulation (Dim {idx+1})"
                )

    def measurement_analysis(self) -> None:
        """
        Perform measurement trend analysis for thermal valve parts.
        """
        if self.tv:
            current_cert = next((cert for cert in self.certifications if cert.part_name == TVParts.GASKET.value), None)
        else:
            current_cert = None
        if current_cert:
            options = list(set([cert.certification if cert.certification != current_cert.certification else f"{cert.certification} \
                                (Current)" for cert in self.all_allocated if cert.part_name == TVParts.GASKET.value]))
        else:
            options = list(set([cert.certification for cert in self.all_allocated if cert.part_name == TVParts.GASKET.value]))

        certification_field = widgets.Dropdown(
            options=['all'] + options if len(options) > 1 else ['all'],
            description='Select Certification:',
            layout=widgets.Layout(width='350px'),
            style={'description_width': '150px'},
            value='all'
        )
        headers = [" ".join([j.capitalize() for j in i.value.replace("thermal valve ", "").split(" ")]) for i in TVParts\
                    if i!=TVParts.WELD and i !=TVParts.HOLDER_1 and i !=TVParts.HOLDER_2]
        self.tv_part_dict = {header: part for header, part in zip(headers, [i.value for i in TVParts\
                                                                             if i!=TVParts.WELD and i !=TVParts.HOLDER_1 and i !=TVParts.HOLDER_2])}

        part_field = widgets.Dropdown(
            options=headers,
            description='Select Part:',
            layout=widgets.Layout(width='350px'),
            style={'description_width': '150px'},
            value="Gasket"
        )

        output = widgets.Output()
        part_change = False
        def on_part_change(change: dict) -> None:
            nonlocal part_change
            if change['new'] != change['old']:
                part_change = True
            with output:
                output.clear_output()
                part_name = self.tv_part_dict.get(part_field.value, None)
                if self.tv:
                    current_cert = next((cert for cert in self.certifications if cert.part_name == part_name), None)
                else:
                    current_cert = None
                if current_cert:
                    options = list(set([cert.certification if cert.certification != current_cert.certification else f"{cert.certification} (Current)" for cert in self.all_allocated if cert.part_name == part_name]))
                else:
                    options = list(set([cert.certification for cert in self.all_allocated if cert.part_name == part_name]))
                certification_field.options = ['all'] + options if len(options) > 1 else ['all']
                certification_field.value = 'all'
                self.get_trend('all', part_name, current_cert)
                part_change = False

        def on_cert_change(change: dict) -> None:
            if part_change:
                return
            part_name = self.tv_part_dict.get(part_field.value, None)
            certification = certification_field.value.replace(" (Current)", "")
            current_cert = next((cert for cert in self.certifications if cert.part_name == part_name), None)
            with output:
                output.clear_output()
                self.get_trend(certification, part_name, current_cert)

        part_field.observe(on_part_change, names='value')
        certification_field.observe(on_cert_change, names='value')

        form = widgets.VBox([
            widgets.HTML("<b>Select dimensions to include in the analysis:</b>"),
            part_field,
            certification_field,
            output
        ])

        display(form)

        with output:
            output.clear_output()
            part_name = self.tv_part_dict.get(part_field.value, None)
            self.get_trend('all', part_name, current_cert)

    def get_trend(self, certification: str, part_name: str, current_cert: TVCertification | None) -> None:   
        """
        Retrieve and plot trend data for the specified certification and part name.
        """  
        dimension_dict = {} 
        opening_temps = {}  
        current_dims = current_cert.dimensions if current_cert and current_cert.dimensions else []
        current_certification = current_cert.certification if current_cert else None
        all_allocated = [i for i in self.all_allocated if i.part_name == part_name]
        if certification != 'all':
            all_allocated = [
                cert for cert in self.all_allocated 
                if cert.certification == certification and cert.part_name == part_name
            ]

        for cert in all_allocated:
            dims = cert.dimensions
            if not dims:
                continue

            tv_main = cert.status if cert.status else None
            if not tv_main:
                tv_main = self.session.query(TVStatus).filter_by(tv_id=cert.tv_id).first()
            if not tv_main:
                tv_main = (
                    self.session.query(TVTestRuns)
                    .filter_by(tv_id=cert.tv_id)
                    .order_by(TVTestRuns.id.desc())
                    .first()
                )

            if tv_main and tv_main.opening_temp is not None:
                for idx, dim_val in enumerate(dims):
                    dimension_dict.setdefault(idx, []).append(dim_val)
                    opening_temps.setdefault(idx, []).append(tv_main.opening_temp)

        self.plot_dimension_trend(dimension_dict, opening_temps, part_name, current_dims, certification, current_certification)


    def plot_dimension_trend(self, dimension_dict: dict[int, list[float]], opening_temps: dict[int, list[float]],\
                              part_name: str, current_dims: list[float], certification: str, current_certification: str) -> None:
        """
        Plot dimension trend analysis for thermal valve parts.
        """
        if not dimension_dict:
            print("No data to plot.")
            return

        num_dims = len(dimension_dict)
        fig, axes = plt.subplots(1, num_dims, figsize=(8*num_dims, 6))
        if isinstance(axes, plt.Axes):
            axes = [axes]  # single subplot case
        else:
            axes = axes.flatten()

        current_opening_temp = self.tv.opening_temp if self.tv else None


        for idx in range(num_dims):
            x_array = dimension_dict[idx]
            y_array = opening_temps[idx]
            current_x = current_dims[idx] if idx < len(current_dims) else None
            current_y = current_opening_temp if current_opening_temp is not None else None

            if certification == 'all':
                title = f"{part_name} Dimension {idx+1} Trend Analysis"
            else:
                title = f"{part_name} Dimension {idx+1} Trend Analysis of {current_certification} @ TV ID: {self.tv_id},\n Compared to {certification}"
            axes[idx].scatter(x_array, y_array)
            if current_x is not None and current_y is not None:
                axes[idx].scatter([current_x], [current_y], color='red', label=f'TV ID: {self.tv_id} {part_name} ({current_certification})' if self.tv else 'Current TV')
                axes[idx].legend(loc = 'lower center', bbox_to_anchor=(0.5, -0.3))
            axes[idx].set_xlabel(f"{part_name} Dimension {idx+1}")
            axes[idx].set_ylabel("Opening Temperature [degC]")
            axes[idx].set_title(title)
            axes[idx].grid(True)

        plt.tight_layout()
        plt.show()

    def tv_remark_field(self) -> None:
        """
        Create a clean input field for TV test remarks with properly styled widgets.
        """
        label_width = '150px'
        field_width = '600px'
        
        title = widgets.HTML("<h3>Add or change remark if necessary</h3>")
        remark = self.tv.remark

        def field(description):
            return {
                'description': description,
                'style': {'description_width': label_width},
                'layout': widgets.Layout(width=field_width, height='50px')
            }

        # Remark input
        remark_widget = widgets.Textarea(**field("Remark:"), value = remark if remark else "")

        # Submit button
        submit_button = widgets.Button(
            description="Submit",
            button_style="success",
            layout=widgets.Layout(width='150px', margin='10px 0px 0px 160px')  # align under field
        )

        submitted = {'done': False}
        output = widgets.Output()

        # Form layout
        form = widgets.VBox([
            title,
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

        # Submission handler
        def on_submit_clicked(b):
            with output:
                output.clear_output()
                new_remark = remark_widget.value.strip()
                if not new_remark:
                    print("Please enter a remark before submitting.")
                    return

                if new_remark == remark:
                    print("Already submitted!")
                else:
                    self.tv.remark = new_remark
                    self.session.commit()
                    print("Remark Submitted!")
        submit_button.on_click(on_submit_clicked)

    def tv_status(self) -> None:
        """
        Display the status and certifications of the selected thermal valve.
        """
        if self.tv:
            self.tv_remark_field()
            columns = [c.name for c in TVStatus.__table__.columns]
            values = [getattr(self.tv, c) for c in columns]
            df = pd.DataFrame({"Field": columns, "Value": values})
            display(widgets.HTML(f"<h4>Thermal Valve ID: {self.tv_id} Status</h4>"))
            display_df_in_chunks(df)

            columns = [cert.part_name for cert in self.certifications]
            values = [cert.certification for cert in self.certifications]
            df = pd.DataFrame({"Part": columns, "Certification": values})
            display(widgets.HTML(f"<h4>Thermal Valve ID: {self.tv_id} Part Certifications</h4>"))
            display_df_in_chunks(df)   

    def weld_analysis(self) -> None:
        """
        Perform weld analysis comparing pre-welded and welded thermal valves.
        """
        all_tv_test_runs: list[TVTestRuns] = []
        for tv in self.all_tvs:
            all_tv_test_runs.extend(tv.test_runs)
        pre_welded_runs = defaultdict(list)
        welded_runs = defaultdict(list)
        ranges = list(set((tv.min_opening_temp, tv.max_opening_temp) for tv in self.all_tvs if tv.min_opening_temp is not None and tv.max_opening_temp is not None))
        all_tv_ids = [tv.tv_id for tv in self.all_tvs]

        for tr in all_tv_test_runs:
            if tr.opening_temp is None:
                continue
            if tr.welded:
                welded_runs[tr.tv_id].append(tr.opening_temp)
            else:
                pre_welded_runs[tr.tv_id].append(tr.opening_temp)

        revisions = list(set(tv.revision for tv in self.all_tvs if tv.revision))
        revisions = "/".join(revisions) if revisions else ""
        if not pre_welded_runs or not welded_runs:
            print("Not enough data for weld analysis.")
            return  
        
        slider = widgets.IntRangeSlider(
            value=[min(all_tv_ids), max(all_tv_ids)],
            min=int(min(all_tv_ids)),
            max=int(max(all_tv_ids)),
            step=1,
            description=f'TVs ({revisions}):',
            continuous_update=True,
            style={'description_width': '200px'}, 
            layout={'width': '600px'}             
        )

        output = widgets.Output()
        def update_plot(change: dict) -> None:
            with output:
                output.clear_output()
                selected_range = change['new']
                selected_tvs = [tv_id for tv_id in all_tv_ids if selected_range[0] <= tv_id <= selected_range[1]]
                filtered_pre_welded = {tv_id: temps for tv_id, temps in pre_welded_runs.items() if tv_id in selected_tvs}
                filtered_welded = {tv_id: temps for tv_id, temps in welded_runs.items() if tv_id in selected_tvs}
                revisions = list(set(tv.revision for tv in self.all_tvs if tv.revision and tv.tv_id in selected_tvs))
                revisions = "/".join(revisions) if revisions else ""
                opening_temp = self.tv.opening_temp if self.tv and self.tv_id in selected_tvs else None
                slider.description = f'TVs ({revisions}):'
                if not filtered_pre_welded or not filtered_welded:
                    print("Not enough data in the selected range for weld analysis.")
                    return
                self.statistical_analysis(filtered_pre_welded, filtered_welded, opening_temp, ranges)

        slider.observe(update_plot, names='value')
        display(slider)

        opening_temp = self.tv.opening_temp if self.tv else None
        if not opening_temp:
            opening_temp = self.test_runs[-1].opening_temp if self.test_runs else None
        with output:
            output.clear_output()
            self.statistical_analysis(pre_welded_runs, welded_runs, opening_temp, ranges)
        display(output)

        #self.revisional_analysis(pre_welded_runs, welded_runs)
        # self.weld_analysis_per_tv(pre_welded_runs, welded_runs, ranges)

    def statistical_analysis(self, pre_welded_runs: dict[int, list[float]], welded_runs: dict[int, list[float]], opening_temp: float, ranges=None) -> None:
        """
        Perform statistical analysis and plot distributions for pre-welded and welded thermal valves.
        """
        pre_welded_avg = {tv_id: temps[-1] for tv_id, temps in pre_welded_runs.items() if temps}
        welded_last = {tv_id: temps[-1] for tv_id, temps in welded_runs.items() if temps}
        differences = {tv_id: welded_last[tv_id] - pre_welded_avg[tv_id] for tv_id in pre_welded_avg if tv_id in welded_last}

        welded_tv_ids = set(welded_last.keys())
        pre_welded_avg = {tv_id: temp for tv_id, temp in pre_welded_avg.items() if tv_id in welded_tv_ids}
        pre_welded_vals = np.array(list(pre_welded_avg.values()))
        welded_vals = np.array(list(welded_last.values()))
        difference_vals = welded_vals - pre_welded_vals
        absolute_differences = np.abs(difference_vals)
        absolute_mean = np.mean(absolute_differences)
        pre_mean, pre_std = np.mean(pre_welded_vals), np.std(pre_welded_vals)
        welded_mean, welded_std = np.mean(welded_vals), np.std(welded_vals)
        diff_mean, diff_std = np.mean(difference_vals), np.std(difference_vals)

        x_diff = np.linspace(diff_mean - 4*diff_std, diff_mean + 4*diff_std, 500)
        data_min = min(pre_welded_vals.min(), welded_vals.min())
        data_max = max(pre_welded_vals.max(), welded_vals.max())
        data_range = data_max - data_min

        x_min = data_min - 0.35 * data_range 
        x_max = data_max + 0.35 * data_range
        x = np.linspace(x_min, x_max, 500)

        pre_pdf = norm.pdf(x, loc=pre_mean, scale=pre_std)
        welded_pdf = norm.pdf(x, loc=welded_mean, scale=welded_std)
        diff_pdf = norm.pdf(x_diff, loc=diff_mean, scale=diff_std)

        plt.figure(figsize=(8,5))
        plt.plot(x, pre_pdf, label='Pre-Welded', color='blue')
        plt.plot(x, welded_pdf, label='Welded', color='red')
        if opening_temp is not None:
            plt.axvline(opening_temp, color='green', linestyle='--', label=f'TV {self.tv_id}: {opening_temp:.2f} [degC]')
            plt.legend()
        if ranges:
            colors = ['black', 'grey']
            for i, r in enumerate(ranges):
                plt.axvline(r[0], color=colors[i], linestyle=':', label=f'Min Spec: {r[0]:.2f} [degC]')
                plt.axvline(r[1], color=colors[i], linestyle='-', label=f'Max Spec: {r[1]:.2f} [degC]')
        plt.fill_between(x, pre_pdf, alpha=0.2, color='blue')
        plt.fill_between(x, welded_pdf, alpha=0.2, color='red')
        plt.title(f'Pre-Welded vs Welded Opening Temperature Distributions\nMean Pre-Welded: {pre_mean:.2f} [degC],\
                   Mean Welded: {welded_mean:.2f} [degC]\nStd Pre-Welded: {pre_std:.2f} [degC], Std Welded: {welded_std:.2f} [degC]')
        plt.xlabel(f'Temperature [degC]')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)
        plt.show()   

        plt.figure(figsize=(8,6))
        plt.hist(list(differences.values()), bins=15, label='Pre-Welded', color='tab:blue', density=True)     
        plt.title(f'Histogram of Opening Temperature Differences (Welded - Pre-Welded)\nMean: {diff_mean:.2f} [degC],\
                   Std: {diff_std:.2f} [degC]\nAbsolute Mean: {absolute_mean:.2f} [degC]', wrap=True)
        plt.xlabel('Temperature Difference [degC]')
        plt.ylabel('Density')
        plt.show()

        # plt.subplot(1,2,2)
        # plt.plot(x_diff, diff_pdf, label='Difference', color='purple')
        # plt.fill_between(x_diff, diff_pdf, alpha=0.2, color='purple')
        # plt.title(f'Difference Distribution\nMean Difference: {diff_mean:.2f} [degC], Std Difference: {diff_std:.2f} [degC]')
        # plt.xlabel('Temperature [degC]')
        # plt.ylabel('Probability Density')
        # plt.legend()
        # plt.grid(True)
        # plt.show()


    def revisional_analysis(self, pre_welded_runs: dict[int, list[float]], welded_runs: dict[int, list[float]]) -> None:
        """
        Perform revisional analysis comparing pre-welded and welded thermal valves by revision.
        """
        revisions = sorted(set(tv.revision for tv in self.all_tvs if tv.revision))
        if not revisions:
            print("No revision data available.")
            return

        pre_means, welded_means = [], []
        pre_stds, welded_stds = [], []

        for rev in revisions:
            tv_ids = [tv.tv_id for tv in self.all_tvs if tv.revision == rev]

            pre_vals = [np.mean(pre_welded_runs[tv_id]) for tv_id in tv_ids if tv_id in pre_welded_runs and pre_welded_runs[tv_id]]
            welded_vals = [welded_runs[tv_id][-1] for tv_id in tv_ids if tv_id in welded_runs and welded_runs[tv_id]]

            if pre_vals:
                pre_means.append(np.mean(pre_vals))
                pre_stds.append(np.std(pre_vals))
            else:
                pre_means.append(np.nan)
                pre_stds.append(np.nan)

            if welded_vals:
                welded_means.append(np.mean(welded_vals))
                welded_stds.append(np.std(welded_vals))
            else:
                welded_means.append(np.nan)
                welded_stds.append(np.nan)

        x = np.arange(len(revisions))
        width = 0.2

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Left y-axis: means
        ax1.bar(x - width, pre_means, width, label='Pre-Weld Mean', color='tab:blue')
        ax1.bar(x, welded_means, width, label='Welded Mean', color='tab:red')
        ax1.set_ylabel('Mean Opening Temp [°C]')
        ax1.set_xticks(x)
        ax1.set_xticklabels(revisions)
        ax1.set_xlabel('Revision')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # Right y-axis: std deviations
        ax2 = ax1.twinx()
        ax2.bar(x + width, pre_stds, width, label='Pre-Weld Std', color='tab:blue', alpha=0.4)
        ax2.bar(x + 2*width, welded_stds, width, label='Welded Std', color='tab:red', alpha=0.4)
        ax2.set_ylabel('Standard Deviation [°C]')
        ax2.legend(loc='upper right')

        plt.title('Opening Temperature Analysis by Revision')
        plt.tight_layout()
        plt.show()

    def weld_analysis_per_tv(self, pre_welded_runs: dict[int, list[float]], welded_runs: dict[int, list[float]], ranges=None) -> None:
        """
        Perform weld analysis per thermal valve comparing pre-welded and welded states.
        """
        pre_mean = {tv_id: temps[-1] for tv_id, temps in pre_welded_runs.items() if temps}
        welded_last = {tv_id: temps[-1] for tv_id, temps in welded_runs.items() if temps}
        tvs_per_plot = 10

        # Only include TVs that exist in both sets
        tv_ids = [tv for tv in pre_mean.keys() if tv in welded_last]
        num_tvs = len(tv_ids)

        # Number of full groups (no separate subplot for leftovers)
        num_subplots = max(1, num_tvs // tvs_per_plot)

        fig, axes = plt.subplots(num_subplots, 1, figsize=(12, num_subplots * 4))
        if num_subplots == 1:
            axes = [axes]

        for i in range(num_subplots):
            start = i * tvs_per_plot
            end = start + tvs_per_plot

            # If last subplot, include all remaining TVs (even if incomplete)
            if i == num_subplots - 1:
                end = num_tvs

            tv_subset = tv_ids[start:end]

            pre_vals = [pre_mean[tv] for tv in tv_subset]
            post_vals = [welded_last[tv] for tv in tv_subset]

            x = np.arange(len(tv_subset))
            width = 0.35

            ax = axes[i]
            ax.bar(x - width / 2, pre_vals, width, label='Pre Weld')
            ax.bar(x + width / 2, post_vals, width, label='Welded')

            ax.set_xticks(x)
            ax.set_xticklabels(tv_subset, rotation=45, ha='right')
            ax.set_xlim(-0.5, len(tv_subset) - 0.5)
            ax.set_ylabel('Temperature [°C]')
            ax.set_title('TV Weld Analysis per TV')
            ax.set_yticks(np.arange(0, 130, 10))
            ax.grid(True)

            if ranges:
                colors = ['black', 'grey']
                for j, r in enumerate(ranges):
                    ax.axhline(r[0], color=colors[j], linestyle=':', label=f'Min Spec: {r[0]:.2f} [°C]')
                    ax.axhline(r[1], color=colors[j], linestyle='-', label=f'Max Spec: {r[1]:.2f} [°C]')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()


    def count_cycles(self, data: list[float], threshold: float = 0.05) -> int:
        """
        Counts the number of cycles in the given data.
        A cycle is defined as a segment that rises above `threshold`
        and then falls back below it.
        
        Parameters:
            data (array-like): Input signal.
            threshold (float): Fraction of the max amplitude used as zero threshold.
            
        Returns:
            int: Number of cycles detected.
        """
        data = np.asarray(data)
        
        thresh_value = np.max(np.abs(data)) * threshold

        active = data > thresh_value

        transitions = np.diff(active.astype(int))
        
        starts = np.sum(transitions == 1)
        
        return starts

    def tvac_analysis(self) -> None:
        """
        Perform TVAC analysis with interactive time range selection.
        """
        if self.test_reference == 'all':
            if not self.tvac_runs:
                print("No TVAC test runs found for this TV.")
                return

            max_time = max(max(tr.time) for tr in self.tvac_runs)
            time_slider = widgets.FloatRangeSlider(
                value=[4.5, 15],
                min=0,
                max=max_time,
                step=0.1,
                description='Time Range [h]:',
                continuous_update=True,
                style={'description_width': '200px'},
                layout={'width': '600px'}
            )
            output = widgets.Output()

            def update_plot(change):
                selected_range = change['new']
                with output:
                    output.clear_output()
                    self.plot_all_tvac_tests(time_range=selected_range)

            time_slider.observe(update_plot, names='value')
            display(time_slider, output)
            with output:
                output.clear_output()
                self.plot_all_tvac_tests(time_range=time_slider.value)
            return

        tvac_run = next((tr for tr in self.tvac_runs if tr.test_id == self.test_reference), None)
        if not tvac_run:
            print("Test ID not found for this TV.")
            return

        # Single test slider
        time_slider = widgets.FloatRangeSlider(
            value=[4.5, 15],
            min=0,
            max=max(tvac_run.time),
            step=0.1,
            description='Time Range [h]:',
            continuous_update=True,
            style={'description_width': '200px'},
            layout={'width': '600px'}
        )
        output = widgets.Output()

        def update_plot(change):
            selected_range = change['new']
            with output:
                output.clear_output()
                self.plot_tvac_analysis(time_range=selected_range)

        time_slider.observe(update_plot, names='value')
        display(time_slider, output)
        with output:
            output.clear_output()
            self.plot_tvac_analysis(time_range=time_slider.value)

    def plot_all_tvac_tests(self, time_range: tuple[float, float] = None) -> None:
        """
        Plot all TVAC test runs for the thermal valve with optional time range filtering.
        """
        if not self.tvac_runs:
            print("No TVAC test runs found for this TV.")
            return

        plt.figure(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.tvac_runs)))
        legend_labels = []

        # create subplots
        ax1 = plt.subplot(2, 2, 1)  # Outlet Temp 2
        ax2 = plt.subplot(2, 2, 2)  # IF Plate
        ax3 = plt.subplot(2, 2, 3)  # Vacuum
        ax4 = plt.subplot(2, 2, 4)  # Power

        for idx, tvac_run in enumerate(self.tvac_runs):
            time = tvac_run.time
            outlet_temp_2 = tvac_run.outlet_temp_2
            if_plate = tvac_run.if_plate if tvac_run.if_plate else tvac_run.if_plate_1 if tvac_run.if_plate_1 else tvac_run.if_plate_2
            vacuum_in_mbar = [10**(v - 5.5) for v in tvac_run.vacuum] if tvac_run.vacuum else []
            current = tvac_run.tv_current
            voltage = tvac_run.tv_voltage
            power = np.array(current) * np.array(voltage)
            cycles = tvac_run.cycles
            max_tv_outlet = max(outlet_temp_2) if outlet_temp_2 else None
            min_tv_outlet = min(outlet_temp_2) if outlet_temp_2 else None
            actual_cycles = self.count_cycles(power) if power.any() else 0

            # Apply time range if given
            if time_range:
                indices = [i for i, t in enumerate(time) if time_range[0] <= t <= time_range[1]]
                if not indices:
                    continue
                time = [time[i] for i in indices]
                outlet_temp_2 = [outlet_temp_2[i] for i in indices]
                if_plate = [if_plate[i] for i in indices] if if_plate else []
                vacuum_in_mbar = [vacuum_in_mbar[i] for i in indices] if vacuum_in_mbar else []
                power = [power[i] for i in indices]

            # Cycle range for label
            cycle_start = 0 if cycles <= 1000 else cycles - 1000
            cycle_end = cycles
            label = f'{tvac_run.test_id} | Cycles: {cycle_start}-{cycle_end}\n Actual Cycles: {actual_cycles} | Max TV Outlet: {max_tv_outlet:.2f} °C | Min TV Outlet: {min_tv_outlet:.2f} °C'
            legend_labels.append((colors[idx], label))

            # Plot curves in subplots
            ax1.plot(time, outlet_temp_2, color=colors[idx])
            if if_plate:
                ax2.plot(time, if_plate, color=colors[idx])
            if vacuum_in_mbar:
                ax3.plot(time, vacuum_in_mbar, color=colors[idx])
            ax4.plot(time, power, color=colors[idx])

        # Set titles, labels, grids
        ax1.set_title(f"TVAC Outlet Temp 2 for cycles 0-{cycle_end}")
        ax1.set_xlabel("Time [h]")
        ax1.set_ylabel("Temperature [°C]")
        ax1.grid(True)

        ax2.set_title(f"IF Plate Temperature for cycles 0-{cycle_end}")
        ax2.set_xlabel("Time [h]")
        ax2.set_ylabel("Temperature [°C]")
        ax2.grid(True)

        ax3.set_title(f"Vacuum Level for cycles 0-{cycle_end}")
        ax3.set_xlabel("Time [h]")
        ax3.set_ylabel("Vacuum [mbar]")
        ax3.grid(True)

        ax4.set_title(f"Power Consumption for cycles 0-{cycle_end}")
        ax4.set_xlabel("Time [h]")
        ax4.set_ylabel("Power [W]")
        ax4.grid(True)

        # Single legend at bottom
        handles = [plt.Line2D([0], [0], color=c, lw=2) for c, _ in legend_labels]
        labels = [l for _, l in legend_labels]
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.figlegend(handles, labels, loc="lower center", ncol=2, fontsize=8)

        plt.show()

    def plot_tvac_analysis(self, time_range: tuple[float, float] = None) -> None:
        """
        Plot TVAC analysis for the selected test run with optional time range filtering.
        """
        tvac_run = next((tr for tr in self.tvac_runs if tr.test_id == self.test_reference), None)
        if not tvac_run:
            print("Test ID not found for this TV.")
            return
        
        time = tvac_run.time
        outlet_temp_1 = tvac_run.outlet_temp_1
        outlet_temp_2 = tvac_run.outlet_temp_2
        cycles = tvac_run.cycles
        vacuum = tvac_run.vacuum
        vacuum_in_mbar = [10**(v-5.5) for v in vacuum] if vacuum else []
        if_plate = tvac_run.if_plate if tvac_run.if_plate else tvac_run.if_plate_1 if tvac_run.if_plate_1 else tvac_run.if_plate_2
        current = tvac_run.tv_current
        voltage = tvac_run.tv_voltage
        power = np.array(current)*np.array(voltage)
        max_tv_outlet = max(max(outlet_temp_1), max(outlet_temp_2)) 
        min_tv_outlet = min(min(outlet_temp_1), min(outlet_temp_2))
        actual_cycles = self.count_cycles(power) if power.any() else 0
        if time_range:
            indices = [i for i, t in enumerate(time) if time_range[0] <= t <= time_range[1]]
            if not indices:
                print("No data in the selected time range.")
                return
            time = [time[i] for i in indices]
            outlet_temp_1 = [outlet_temp_1[i] for i in indices]
            outlet_temp_2 = [outlet_temp_2[i] for i in indices]
            if_plate = [if_plate[i] for i in indices] if if_plate else []
            vacuum_in_mbar = [vacuum_in_mbar[i] for i in indices] if vacuum_in_mbar else []
            power = [power[i] for i in indices] 

        plt.figure(figsize=(14, 8))
        start_cycle = 0 if cycles <= 1000 else cycles - 1000
        end_cycle = cycles
        plt.subplot(2, 2, 1)
        plt.plot(time, outlet_temp_1, color="tab:blue", label='Outlet Temp 1')
        plt.plot(time, outlet_temp_2, color="tab:orange", label='Outlet Temp 2')
        plt.title(f'TVAC Outlet Temperatures\nTest ID: {self.test_reference} | Cycles: {start_cycle}-{end_cycle}\n \
                  Actual Cycles: {actual_cycles} | Max TV Outlet: {max_tv_outlet:.2f} °C | Min TV Outlet: {min_tv_outlet:.2f} °C')
        plt.xlabel('Time [h]')
        plt.ylabel('Temperature [°C]')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(time, if_plate, label='IF Plate Temp', color='tab:orange')
        plt.title(f'TV IF Plate Temperature\nTest ID: {self.test_reference} | Cycles: {start_cycle}-{end_cycle}\n\
                   Actual Cycles: {actual_cycles}')
        plt.xlabel('Time [h]')
        plt.ylabel('Temperature [°C]')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(time, vacuum_in_mbar, label='Vacuum', color='tab:green')
        plt.title(f'TVAC Vacuum Level\nTest ID: {self.test_reference} | Cycles: {start_cycle}-{end_cycle}\n Actual Cycles: {actual_cycles}')
        plt.xlabel('Time [h]')
        plt.ylabel('Vacuum [mbar]')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(time, power, label='Power', color='tab:red')
        plt.title(f'TVAC Power Consumption\nTest ID: {self.test_reference} | Cycles: {start_cycle}-{end_cycle}\n Actual Cycles: {actual_cycles}')
        plt.xlabel('Time [h]')
        plt.ylabel('Power [W]')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def get_certifications(self) -> None:
        """
        Retrieve and display certifications and corresponding data for the
        relevant thermal valve parts.
        """
        grouped_part_certs = self.session.query(
            TVCertification.certification,
            TVCertification.part_name,
            func.count(TVCertification.part_id)
        ).filter(TVCertification.part_name != TVParts.WELD.value).group_by(
            TVCertification.certification,
            TVCertification.part_name
        ).all()
        df_parts = pd.DataFrame(grouped_part_certs, columns=['Certification', 'Part Name', 'Amount/IDs'])

        grouped_weld_certs = self.session.query(
            TVCertification.certification,
            TVCertification.part_name,
            func.group_concat(TVCertification.tv_id)
        ).filter(
            TVCertification.part_name == TVParts.WELD.value,
            TVCertification.tv_id != None
        ).group_by(
            TVCertification.certification
        ).all()
        df_weld = pd.DataFrame(grouped_weld_certs, columns=['Certification', 'Part Name', 'Amount/IDs'])

        main_df = pd.concat([df_parts, df_weld], ignore_index=True, sort=False)

        main_df["Cert_Group"] = main_df["Certification"].str.extract(r"C(\d+)-\d+").astype(int)
        main_df["Cert_Num"] = main_df["Certification"].str.extract(r"C\d+-(\d+)").astype(int)

        main_df = main_df.sort_values(by=["Cert_Group", "Cert_Num"], ascending=[True, True]).reset_index(drop=True)
        main_df = main_df.drop(columns=["Cert_Group", "Cert_Num"])

        display(main_df)

        