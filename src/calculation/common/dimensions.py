import pandas as pd

from typing import Any, Dict


class ModelDimensions:
    """Contains the categories for each model dimension."""

    combustion_type: Dict[int, Dict[str, Any]]
    emission_type: Dict[int, Dict[str, Any]]
    employment_sector: Dict[int, Dict[str, Any]]
    firm_size: Dict[int, Dict[str, Any]]
    flow_type: Dict[int, Dict[str, Any]]
    logistic_node: Dict[int, Dict[str, Any]]
    logistic_segment: Dict[int, Dict[str, Any]]
    municipality: Dict[int, Dict[str, Any]]
    nstr: Dict[int, Dict[str, Any]]
    road_type: Dict[int, Dict[str, Any]]
    shipment_size: Dict[int, Dict[str, Any]]
    van_segment: Dict[int, Dict[str, Any]]
    vehicle_type: Dict[int, Dict[str, Any]]

    def __init__(self, dim_folder: str):
        """Constructor of a ModelDimensions object, fills its attributes."""
        for attr_name in (
            'combustion_type',
            'emission_type',
            'employment_sector',
            'firm_size',
            'flow_type',
            'logistic_node',
            'logistic_segment',
            'municipality',
            'nstr',
            'road_type',
            'shipment_size',
            'van_segment',
            'vehicle_type',
        ):
            setattr(
                self,
                attr_name,
                dict(
                    (row["ID"], row)
                    for row in pd.read_csv(f"{dim_folder}{attr_name}.txt", sep='\t').to_dict('records')
                )
            )

    def get_id_from_label(self, attr_name: str, label: str) -> int:
        """Returns the ID belonging to a label of a dimension."""
        try:
            attr: Dict[int, Dict[str, Any]] = getattr(self, attr_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Unknown attribute name passed into ModelDimensions class: '{attr_name}'"
            ) from exc

        for row in attr.values():
            if row["Comment"] == label:
                return int(row["ID"])

        raise AttributeError(
            f"Label '{label}' was not found in ModelDimensions attribute '{attr_name}'"
        )