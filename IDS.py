# --------------------------------------------------------------------------------------------------------------
#
# Filename: IDS.py
# Author: Shahid Alam (shalam3@gmail.com)
# Dated: September, 3, 2025
# IDS => Intrusion Detection System
#
# --------------------------------------------------------------------------------------------------------------

from __future__ import print_function
import os, sys, argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from Classify import *

__DEBUG__  = True
CLASS_LABEL = "Class"
# Classes
classes = ['Normal', 'Blackhole', 'TCP-SYN', 'PortScan', 'Diversion', 'Overflow']

if __name__=="__main__":
	print("--- IDS START ---")

	parser = argparse.ArgumentParser()

	parser.add_argument("-csv", type=str)
	args = parser.parse_args()

	if args.csv == None:
		print("ERROR:IDS: Missing arguments\nUsage:\n")
		print("   python IDS.py -csv <csv_filename>")
		sys.exit(1)
	csv_filename = args.csv.strip()
	if os.path.exists(csv_filename) == False:
		print("File '%s' does not exist"%csv_filename)
		print("   python IDS.py -csv <csv_filename>")
		sys.exit(1)

	CL = Classifier(CLASS_LABEL, classes)
	dataset = pd.read_csv(csv_filename)
	dataset.rename(columns={"Label": CLASS_LABEL}, inplace=True)
	print(dataset)
    ## ----------------------------------------------- Z
if len(dataset) >= 29:
    if dataset.isna().any(axis=None) == True:
        dataset.interpolate(method='linear', inplace=True)
    if dataset.isna().any(axis=None) == True:
        print("Missing values still present in the dataset")
        sys.exit(1)

    # 20% test size (original)
    CL.classify(dataset, csv_filename, testSize=20)

    # 50% test size (Task 2)
    CL.classify(dataset, csv_filename, testSize=50)
else:
    print("ERROR:Nothing to classify in file %s"%csv_filename)
    
## -------------------------------------------
	print("--- IDS END ---")

#-------------------------------------------------------------------------------------------------
# This dataset contains 5000 records of features extracted from Network Port Statistic to protect
# modern-day computer networks from cyber attacks and are thereby classified into 5 classes
# 
# --- 5 CLASSES ---
# 0 - Normal
# 1 - Blackhole
# 2 - TCP-SYN
# 3 - PortScan
# 4 - Diversion
# 5 - Overflow
# 
# --- FEATURES ---
# Switch ID - The switch through which the network flow passed
# Port Number - The switch port through which the flow passed
# Received Packets - Number of packets received by the port
# Received Bytes - Number of bytes received by the port
# Sent Bytes - Number of bytes sent by the port
# Sent Packets - Number of packets sent by the port
# Port alive Duration (S) - The time port has been alive in seconds
# Packets Rx Dropped - Number of packets dropped by the receiver
# Packets Tx Dropped - Number of packets dropped by the sender
# Packets Rx Errors - Number of transmit errors
# Delta Received Packets - Number of packets received by the port
# Delta Received Bytes - Number of bytes received by the port
# Delta Sent Bytes - Number of bytes sent by the port
# Delta Sent Packets - Number of packets sent by the port
# Delta Port alive Duration (S) - The time port has been alive in seconds
# Delta Packets Rx Dropped - Number of packets dropped by the receiver
# Delta Packets Tx Dropped - Number of packets dropped by the sender
# Delta Packets Rx Errors - Number of receive errors
# Delta Packets Tx Errors - Number of transmit errors
# Connection Point - Network connection point expressed as a pair of the network element identifier and port number.
# Total Load/Rate - Obtain the current observed total load/rate (in bytes/s) on a link.
# Total Load/Latest - Obtain the latest total load bytes counter viewed on that link.
# Load/Rate - Obtain the current observed unknown-sized load/rate (in bytes/s) on a link.
# Unknown Load/Latest - Obtain the latest unknown-sized load bytes counter viewed on that link.
# Latest bytes counter - Latest bytes counted in the switch port
# Checkis_valit - Indicates whether this load was built on valid values.
# vpn_keyTable ID - Returns the Table ID values.
# Active Flow Entries - Returns the number of active flow entries in this table.
# Packets Looked Up - Returns the number of packets looked up in the table.
# Packets Matched - Returns the number of packets that successfully matched in the table.
# Max Size - Returns the maximum size of this table.
# Label - Label types (CLASSES) for intrusions
#-------------------------------------------------------------------------------------------------

