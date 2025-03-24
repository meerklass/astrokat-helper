import os
import katpoint
import astrokat
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse


def ra_hms2deg(ra: str):
    data = ra.split(':')
    deg = (float(data[0]) + float(data[1])/60 + float(data[2])/60/60)*360/24
    return deg

def dec_dms2deg(dec: str):
    data = dec.split(':')
    temp=float(data[0])
    if temp < 0:
        deg = temp - float(data[1])/60 - float(data[2])/60/60
    else: 
        deg = temp + float(data[1])/60 + float(data[2])/60/60
    return deg
    
cli = argparse.ArgumentParser()
cli.add_argument(
    "--files",
    nargs='+', 
    type=str,
    help='same name of .yaml file and corresponding .out file produced by astrokat-observe script. Can combine files',
    required=True
)
cli.add_argument(
    "--plot",
    nargs=1,
    type=str,
    help='File name for .png fig',
    required=True
)
args = cli.parse_args()
      
#MEERKAT_REFERENCE_LOCATION = "ref, -30:42:39.8, 21:26:38.0, 1035.0, 0.0, , , 1.15"
#ref_antenna = katpoint.Antenna(MEERKAT_REFERENCE_LOCATION)
location = astrokat.Observatory().location
ref_antenna = katpoint.Antenna(location)

plt.figure(figsize=(20, 5))

for file_name in args.files:
    radec_p = []
    file = open(file_name + '.yaml', 'r')
    for line in file.readlines():
        if "radec_p" in line:
            data = line.split()
            ra=data[1].split(',')[0]
            dec=data[2]
            print(ra,dec)
            radec_p.append([ra_hms2deg(ra), dec_dms2deg(dec)])    
    pointx, pointy = [], []
    file = open(file_name + '.out', 'r')
    for line in file.readlines():
        if "Scan duration is" in line:
            duration = float(re.split("Scan duration is|and scan speed is", line)[1])
            print(duration)
        if "Azimuth scan extent" in line:
            extenta, extentb = float(re.split("\[|\]", line)[-2].split(',')[0]), float(
                re.split("\[|\]", line)[-2].split(',')[1])
            print(extenta, extentb)
        if "Scan target: scan_azel_with_nd_trigger" in line:
            data = line.split()
            print(data)
            datatime = "%s %s" % (data[0], data[1])
            datatime = datatime.replace('Z', ' ')  # remove 'Z' from data
            targetbase = katpoint.Target("point, azel , %s ,%s" % (data[7], data[8].split(',')[0]))
            print(katpoint.Timestamp(datatime))
            print(datatime,extenta,extentb,data[7],data[8].split(',')[0])
            for offset, offtime in zip(np.linspace(extenta, extentb), np.linspace(0, duration)):
                az, el = np.degrees(targetbase.azel())
                target = katpoint.Target("point, azel , %f ,%f" % (az + offset, el))
                radec = np.degrees(target.radec(timestamp=katpoint.Timestamp(datatime) + offtime, antenna=ref_antenna))
                pointx.append(radec[0])
                pointy.append(radec[1])

    plt.plot(pointx, pointy, '.', alpha=0.3)
    #plt.plot(x[0],x[1], label=f'rise {i}')
    #plt.xlabel('time of day UTC [hours]')
    #plt.legend()

#    for corner in radec_p:
#        plt.plot(corner[0], corner[1], "*")
    ra_corners=[x[0] for x in radec_p]
    dec_corners=[x[1] for x in radec_p]
    ramin=min(ra_corners)
    ramax=max(ra_corners)
    decmin=min(dec_corners)
    decmax=max(dec_corners)
    plt.plot([ramin,ramin,ramax,ramax,ramin], [decmin,decmax,decmax,decmin,decmin])
    #plt.plot([ramin,ramin],[decmin,decmax])
    #plt.plot([ramin,ramax],[decmin,decmin])
    #plt.plot([ramin,ramax],[decmax,decmax])
    #plt.plot([ramax,ramax],[decmin,decmax])

plt.savefig(args.plot[0]+'.png')
plt.close() 
