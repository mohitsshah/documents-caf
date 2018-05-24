# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:44:50 2018

@author: genesis
"""
import cv2
import numpy as np
import parseXMLForSegmentation as pxfs


def visSegs(new_img, parts):
    segs = []
    height, width = new_img.shape
    cnt = 1
    for p in parts:
        partimg = new_img[p[0]:p[1], :]
        pH, pW = partimg.shape
        cnt += 1
        if partimg.any():
            xc0, yc0, xc1, yc1 = cropSeg(partimg)
            partimg = partimg[xc0:xc1, yc0:yc1]
            h, w = partimg.shape
            percs = []
            classes = []
            for i in range(w):
                hSliver = partimg[:, i]
                percs.append(float(np.sum(hSliver)) / (255 * h))
            heatmap = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(w):
                if percs[i] > 0.6:
                    heatmap[:, i] = (255, 0, 0)
                    classes.append(0)
                elif percs[i] > 0.05:
                    heatmap[:, i] = (0, 255, 0)
                    classes.append(0)
                else:
                    heatmap[:, i] = (0, 0, 255)
                    classes.append(1)
            classes = np.array(classes)
            partitions = np.where(classes[1:] != classes[:-1])[0] + 1
            runs = []
            start = 0
            if len(partitions) > 1:
                for pa in partitions:
                    runs.append((start, pa - 1, classes[start]))
                    start = pa
            runs.append((start, len(classes) - 1, classes[start]))
            percs = np.array(percs)
            pointzero5 = float(len(percs[percs > 0.05]) - len(percs[percs > 0.6])) / len(percs)
            if len(runs) == 1:
                # print("SINGLE COLUMN PAGE")
                segs.append((p[0] + xc0, yc0, p[1] - (pH - xc1), yc1))
            elif len(runs) == 3 and runs[1][2] == 1 and pointzero5 < 0.85:
                wsrun = runs[1]
                wsloc = yc0 + (wsrun[0] + (float(wsrun[1] - wsrun[0]) / 2))

                if abs(wsloc - (width / 2)) < 0.04 * width and (float(h) / height) > 0.15:
                    # print("2-COLUMN PAGE")
                    semipart1 = partimg[:, :wsrun[0]]
                    sp1_segs = pxfs.segmentPage4(semipart1)
                    for sp in sp1_segs:
                        semi = semipart1[sp[0]:sp[1]]
                        if semi.any():
                            segs.append((p[0] + xc0 + sp[0], yc0, p[0] + xc0 + sp[1], yc0 + wsrun[0]))

                    semipart2 = partimg[:, wsrun[1]:]

                    sp2_segs = pxfs.segmentPage4(semipart2)
                    for sp in sp2_segs:
                        semi = semipart2[sp[0]:sp[1]]
                        if semi.any():
                            segs.append((p[0] + xc0 + sp[0], yc0 + wsrun[1], p[0] + xc0 + sp[1], yc1))
                else:
                    # print("SINGLE COLUMN PAGE 1")
                    segs.append((p[0] + xc0, yc0, p[1] - (pH - xc1), yc1))
            else:
                # print("UNK")
                segs.append((p[0] + xc0, yc0, p[1] - (pH - xc1), yc1))
    return segs


def cropSeg(segimg):
    xc0, yc0 = 0, 0
    xc1, yc1 = segimg.shape
    for i in range(xc1):
        line = segimg[i, :]
        if np.any(line):
            xc0 = i
            break

    for i in range(yc1):
        line = segimg[:, i]
        if np.any(line):
            yc0 = i
            break

    for i in range(xc1 - 1, xc0, -1):
        line = segimg[i, :]
        if np.any(line):
            xc1 = i + 1
            break

    for i in range(yc1 - 1, yc0, -1):
        line = segimg[:, i]
        if np.any(line):
            yc1 = i + 1
            break

    return xc0, yc0, xc1, yc1
