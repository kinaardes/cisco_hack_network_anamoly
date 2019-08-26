from lxml import objectify
import pandas as pd
import pickle


#
# xml = objectify.parse('/Users/kindesai/PycharmProjects/CISCOHACK/TestbedThuJun17-3Flows.xml',  low_memory=False)
#
#
# root = xml.getroot()
#
# data=[]
#
# for i in range(len(root.getchildren())):
#     data.append([child.text for child in root.getchildren()[i].getchildren()])
#
# df = pd.DataFrame(data)
# df.columns = ['appName', 'totalSourceBytes', 'totalDestinationBytes',
#               'totalDestinationPackets',  'totalSourcePackets',
#               'sourcePayloadAsBase64', 'sourcePayloadAsUTF', 'destinationPayloadAsBase64',
#               'destinationPayloadAsUTF', 'direction', 'sourceTCPFlagsDescription',
#               'destinationTCPFlagsDescription', 'source', 'protocolName', 'sourcePort', 'destination',
#               'destinationPort', 'startDateTime', 'stopDateTime', 'Tag']
#
# with open('TestBed_06_17', 'wb') as o:
#      pickle.dump(df, o)

names=[

    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted' ,'num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate']

print(len(names))