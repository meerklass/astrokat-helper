import katpoint
import matplotlib.pyplot as plt
import numpy as np
import re

MEERKAT_REFERENCE_LOCATION = "ref, -30:42:39.8, 21:26:38.0, 1035.0, 0.0, , , 1.15"
ref_antenna = katpoint.Antenna(MEERKAT_REFERENCE_LOCATION)
file1 = open('../output/observe-output.txt', 'r')
pointx, pointy = [], []
for line in file1.readlines():
    if "Scan duration is" in line:
        duration = float(re.split("Scan duration is|and scan speed is", line)[1])
        # print(duration)
    if "Azimuth scan extent" in line:
        extenta, extentb = float(re.split("\[|\]", line)[-2].split(',')[0]), float(
            re.split("\[|\]", line)[-2].split(',')[1])
    if "Scan target: scan_azel_with_nd_trigger" in line:
        data = line.split()
        datatime = "%s %s" % (data[0], data[1])
        datatime = datatime.replace('Z', ' ')  # added by Amadeus
        targetbase = katpoint.Target("point, azel , %s ,%s" % (data[7], data[8].split(',')[0]))
        # print(katpoint.Timestamp(datatime))
        # print(datatime,extenta,extentb,data[7],data[8].split(',')[0])
        for offset, offtime in zip(np.linspace(extenta, extentb), np.linspace(0, duration)):
            az, el = np.degrees(targetbase.azel())
            target = katpoint.Target("point, azel , %f ,%f" % (az + offset, el))
            radec = np.degrees(target.radec(timestamp=katpoint.Timestamp(datatime) + offtime, antenna=ref_antenna))
            pointx.append(radec[0])
            pointy.append(radec[1])

plt.plot(pointx, pointy, 'k.', alpha=0.3)

radec_p = []
radec_p.append(
    katpoint.Target("point, radec , %s ,%s " % ("10:16:00.00", "-08:51:00.00")))  # Note the comma separation.
radec_p.append(
    katpoint.Target("point, radec , %s ,%s " % ("09:20:55.20", "-08:51:00.00")))  # Note the comma separation.
radec_p.append(katpoint.Target("point, radec , %s ,%s " % ("09:20:55.20", "05:26:24.00")))  # Note the comma separation.
radec_p.append(katpoint.Target("point, radec , %s ,%s " % ("10:16:00.00", "05:26:24.00")))  # Note the comma separation.








for corner in radec_p:
    x, y = np.degrees(corner.radec(antenna=ref_antenna))
    plt.plot(x, y, "r*")
plt.show()