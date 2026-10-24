"""Bench2Drive entry point; legacy controllers use only this snapshot/model."""
from fmsrpdm import bootstrap
bootstrap.configure_imports()
from sensor_agent import SensorAgent


def get_entry_point():
    return 'FMSRPDMAgent'


class FMSRPDMAgent(SensorAgent):
    pass
