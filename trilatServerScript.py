import matplotlib.pyplot as plt

import MySQLdb
from datetime import datetime
import numpy as np
import pandas as pd
import re
import math as math
import trilatWrapper as tw
import trilatDistWrapper as tdw

def haversinedist1(p1, p2):
    lat1 = p1[0]
    lon1 = p1[1]
    lat2 = p2[0]
    lon2 = p2[1]
    # distance in km
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2)) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * c * 1.46843 / 100

def haversinedist(lat1, lon1, lat2, lon2):
    # distance in km
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2)) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * c * 1.46843 / 100


class TDOA(object):
    def __init__(self, dba, experiment_id):

        self.__dbFile = dba

        #initial data processing
        self.get_server_data_from_file()
        self.get_experiment_data(experiment_id=experiment_id)
        self.get_ts_data(experiment_id=experiment_id)
        self.get_cluster_ts_data(experiment_id=experiment_id)

    def get_server_data_from_file(self):
        data = pd.read_csv(self.__dbFile.filename, names=self.__dbFile.SCHEMA_DTOA)
        self.serverdata = data

    def get_experiment_data(self, experiment_id):
        '''
        #1. takes only the relevant rows
        #2. remove msgIDs that have at least one missing etime
        #since we won't be able to use the msg
        :return:
        '''
        data = []
        if experiment_id == 'four_sub_ids_per_measurement_dec_13':
            data = self.serverdata.iloc[0:660,:]
            data = data[np.invert(data['etime'].str.contains('None'))]
        elif experiment_id == 'one_measurement_per_second_dec_13':
            data = self.serverdata.iloc[683:,:]
        elif experiment_id == 'tdoa_dec_10':
            data = self.serverdata[np.invert(self.serverdata['MsgNo'].str.contains('83L'))]
        else:
            pass
        self.experiment_data = data

    def get_ts_data(self, experiment_id):
        #3. transform each column and produce a time series object data
        tsObjData = pd.DataFrame([])
        non_decimal = re.compile(r'[^\d.]+')
        for index, row in self.experiment_data.iterrows():
            tsObjData.loc[row['RID'], 'rid'] = row['RID']
            tsObjData.loc[row['RID'], 'gwip'] = row['gwip']
            tsObjData.loc[row['RID'], 'msgno'] = non_decimal.sub('', row['MsgNo'])
            if (len(row['ts']) == 27):
                tsObjData.loc[row['RID'], 'ts'] = row['ts'][19:26]
                tsObjData.loc[row['RID'], 'date'] = row['ts'][2:12]
            else:
                tsObjData.loc[row['RID'], 'ts'] = row['ts'][21:28]
                tsObjData.loc[row['RID'], 'date'] = row['ts'][2:12]
            tsObjData.loc[row['RID'], 'gpslat'] = non_decimal.sub('', str(row['gpslat']))
            tsObjData.loc[row['RID'], 'netgpslon'] = non_decimal.sub('', str(row['gpslon']))
            tsObjData.loc[row['RID'], 'gwlat'] = non_decimal.sub('', str(row['gwlat']))
            tsObjData.loc[row['RID'], 'netgwlon'] = non_decimal.sub('', str(row['gwlon']))

            tsObjData.loc[row['RID'], 'rssic'] = row['rssic']
            tsObjData.loc[row['RID'], 'lsnr'] = float(row['lsnr'])
            tsObjData.loc[row['RID'], 'ftime'] = long(row['ftime']) / 1e9

        if experiment_id == 'four_sub_ids_per_measurement_dec_13':
            self.ts_data = tsObjData[tsObjData['date'].str.contains('2017-12-13')]
        elif experiment_id == 'tdoa_dec_10':
            self.ts_data = tsObjData[tsObjData['date'].str.contains('2017-12-10')]
        else:
            pass

    def get_cluster_ts_data(self, experiment_id):
        clusterTsData = pd.DataFrame([])
        currMsgCount = 0
        currMsgNo = 2  # based on first value in the data
        countSuccess = 0
        prevRow = None
        prevPrevRow = None
        prev3Row = None
        non_decimal = re.compile(r'[^\d.]+')

        for index, row in self.ts_data.iterrows():
            prevMsgNo = currMsgNo
            currMsgNo = row["msgno"]
            if prevMsgNo == currMsgNo:
                # print "Found one more"
                currMsgCount += 1
            else:
                # print "New Message"
                currMsgCount = 1
            if currMsgCount == 4:  # adjust to number of gateways
                # how to add previous row on as well?
                addrows = [prev3Row, prevPrevRow, prevRow, row]
                for ar in addrows:
                    clusterTsData.loc[ar['rid'], 'msgno'] = ar['msgno']
                    clusterTsData.loc[ar['rid'], 'gwip'] = ar['gwip']
                    clusterTsData.loc[ar['rid'], 'ts'] = ar['ts']
                    clusterTsData.loc[ar['rid'], 'ftime'] = ar['ftime']
                    clusterTsData.loc[ar['rid'], 'gpslat'] = ar['gpslat']
                    clusterTsData.loc[ar['rid'], 'netgpslon'] = ar['netgpslon']
                    clusterTsData.loc[ar['rid'], 'gwlat'] = ar['gwlat']
                    clusterTsData.loc[ar['rid'], 'netgwlon'] = ar['netgwlon']
                    clusterTsData.loc[ar['rid'], 'rssic'] = ar['rssic']
                    clusterTsData.loc[ar['rid'], 'lsnr'] = ar['lsnr']
                    clusterTsData.loc[ar['rid'], 'geodist'] = haversinedist(float(non_decimal.sub('', ar['gpslat'])),
                                                                            float(non_decimal.sub('', ar['netgpslon'])),
                                                                            float(non_decimal.sub('', ar['gwlat'])),
                                                                            float(non_decimal.sub('', ar['netgwlon'])))
                countSuccess += 1
            prev3Row = prevPrevRow
            prevPrevRow = prevRow
            prevRow = row
        self.cluster_ts_data = clusterTsData

class DatabaseAccess(object):
    def __init__(self, db_name='lgdb'):
        self.db = MySQLdb.connect(host="hgltdb.ckapwzvj8jd5.us-west-1.rds.amazonaws.com",  # your host, usually localhost
                             user="logisticadmin",  # your username
                             passwd="ppnanaapr29",  # your password
                             db=db_name)  # name of the data base
        #schema for the table dtoa
        if db_name == 'lg3':
            self.SCHEMA_DTOA = ["RID", "ExpId", "SubId", "MsgNo", "TID", "tmst", "ts", "gwip", "gpslat", "gpslon", "temp", "pressure",
                                "extraCol1", "lsnr",
                                 "gwlat", "gwlon",  "etime", "ftime","extraCol2","extraCol3",
                               "extraCol4","rssic","extraCol6"]
        if db_name == 'lgdb':
            self.SCHEMA_DTOA = ["RID", "MsgNo", "TID", "tmst", "ts", "gwip", "gpslat", "gpslon", "temp", "pressure",
                                 "gwlat", "gwlon", "lsnr", "etime", "ftime","extraCol2","extraCol3",
                               "extraCol4","rssic","extraCol6"]
        #self.cursor = self.db.cursor()

    def table_names(self):
        cursor = self.db.cursor()
        cursor.execute("show tables")
        tables = cursor.fetchall()
        return tables
    def fetch_dtoa_first(self):
        cur = self.db.cursor()
        cur.execute("SELECT * FROM dtoa")

        row1 = cur.fetchone()
        print row1
    def fetch_dtoa_five(self):
        cur = self.db.cursor()
        cur.execute("SELECT * FROM dtoa")

        row5 = cur.fetchmany(5)
        print row5
    def dtoa_raw_from_db_to_file(self):
        filename = 'ServerData' + str(datetime.now().date()).replace(" ", "").replace(":", "-") + '.csv'
        f = open(filename, 'w')
        cur = self.db.cursor()
        cur.execute("SELECT * FROM dtoa")

        i_gpslat = self.SCHEMA_DTOA.index("gpslat")
        i_gpslon = self.SCHEMA_DTOA.index('gpslon')
        i_gwlat = self.SCHEMA_DTOA.index('gwlat')
        i_gwlon = self.SCHEMA_DTOA.index('gwlon')
        i_gwip = self.SCHEMA_DTOA.index('gwip')

        prevLatLonDict = {}
        for row in cur.fetchall():
            if (row[i_gpslat] and row[i_gpslon]):
                if (row[i_gwlat] and row[i_gwlon]):
                    prevLat = row[i_gwlat]
                    prevLon = row[i_gwlon]
                    prevLatLonDict[row[i_gwip]] = [prevLat, prevLon]
                    f.write(str(row))
                    f.write('\n')
                else:
                    try:
                        extrapLat, extrapLon = prevLatLonDict.get(row[5])
                        newRow = str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "," + str(
                            row[4]) + "," + str(row[5]) + "," + str(row[6]) + "," + str(row[7]) + "," + str(
                            row[8]) + "," + str(
                            row[9]) + "," + str(row[10]) + "," + extrapLat + "," + extrapLon + "," + str(
                            row[13]) + "," + str(
                            row[14])
                        f.write(newRow)
                        f.write('\n')
                    except:
                        pass

        f.close()
        self.filename = filename
        return filename



def main():

    db = DatabaseAccess()
    filename = db.dtoa_raw_from_db_to_file()

    # Use all the SQL you like
    # cur.execute("SELECT t.name AS table_name, SCHEMA_NAME(schema_id) AS schema_name, c.name AS column_name FROM sys.tables AS t 	INNER JOIN sys.columns c ON t.OBJECT_ID = c.OBJECT_ID where t.name = 'ProductItem'  AND C.name like '%retail%' ORDER BY schema_name, table_name")

if __name__ == '__main__':
    main()
