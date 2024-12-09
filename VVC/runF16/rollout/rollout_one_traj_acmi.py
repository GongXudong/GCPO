from pathlib import Path
import pandas as pd
import itertools
import logging
import sys

PROJECT_ROOT_DIR: Path = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))
    
from runF16.f16Classes import Guide, ControlLaw, PlaneModel
from runF16.utils_my.csv2acmi import csv2acmi

def rollout(target_v, target_mu, target_chi, rollout_length):

    h0 = 5000
    v0 = 200
    gtmp = Guide()
    f16cl = ControlLaw(stepTime=0.1)
    f16model = PlaneModel(h0, v0, stepTime=0.1)
    # f16model.setLog('f16trace4.csv')  # 设置日志名
    stsDict = f16model.getPlaneState()

    traj_logs = {
        "time": [],
        "lon": [],
        "lat": [],
        "alt": [],
        "roll": [],
        "pitch": [],
        "yaw": []
    }

    wCmds = {}
    # wCmds['v'] = 180
    # wCmds['mu'] = 20
    # wCmds['chi'] = 30
    wCmds['v'] = target_v
    wCmds['mu'] = target_mu
    wCmds['chi'] = target_chi
    for i in range(rollout_length):

        traj_logs["time"].append(i * 0.1)
        traj_logs["lon"].append(stsDict["lon"])
        traj_logs["lat"].append(stsDict["lat"])
        traj_logs["alt"].append(stsDict["h"])
        traj_logs["roll"].append(stsDict["phi"])
        traj_logs["pitch"].append(stsDict["theta"])
        traj_logs["yaw"].append(stsDict["psi"])

        gtmp.step(wCmds, stsDict)
        gout = gtmp.getOutputDict()
        f16cl.step(gout, stsDict)
        clout = f16cl.getOutputDict()
        f16model.step(clout)
        stsDict = f16model.getPlaneState()

    traj_df = pd.DataFrame(data=traj_logs, columns=["time", "lon", "lat", "alt", "roll", "pitch", "yaw"])

    csv2acmi(
        traj_df=traj_df,
        time_column_name="time",
        lon_column_name="lon", 
        lat_column_name="lat",
        alt_column_name="alt",
        roll_column_name="roll",
        pitch_column_name="pitch",
        yaw_column_name="yaw",
        acmi_file_save_dir=Path(__file__).absolute().parent,
        acmi_file_name=f"expert_{str(target_v)}_{str(target_mu)}_{str(target_chi)}.acmi"
    )

if __name__ == "__main__":
    rollout(100, 0, 0, 779)