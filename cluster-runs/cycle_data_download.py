import os
from google_drive_downloader import GoogleDriveDownloader as gdd

maxar2duke_id = '1IIK2zEEK20iOPYnR_QmVromlhKSMEQVI'
duke2australia_id =  '1RlsIoNBAoGVR22wSqQbz8pwK8t2FOsyL'
duke2bangladesh_id = '1Q_nJrQ6-6rmnVfUfhYeaFNunbsX72poE'
duke2drc_id = '1-0hVIvhPHbqJTMxUI5OYtAQuBMXW1P2q'
duke2ghana_id = '1-2gAgmi29sZ-9jNV-byK3I2VlI5bnEeH'

# dest_path = os.path.join(os.getcwd(), 'datasets', 'hold')
# gdd.download_file_from_google_drive(file_id=maxar2duke_id, 
#                                     dest_path=dest_path, unzip=True)
# os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

# dest_path = os.path.join(os.getcwd(), 'datasets', 'hold')
# gdd.download_file_from_google_drive(file_id=duke2australia_id, 
#                                     dest_path=dest_path, unzip=True)
# os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

dest_path = os.path.join(os.getcwd(), 'datasets', 'hold')
gdd.download_file_from_google_drive(file_id=duke2bangladesh_id, 
                                    dest_path=dest_path, unzip=True)
os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

dest_path = os.path.join(os.getcwd(), 'datasets', 'hold')
gdd.download_file_from_google_drive(file_id=duke2drc_id, 
                                    dest_path=dest_path, unzip=True)
os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

dest_path = os.path.join(os.getcwd(), 'datasets', 'hold')
gdd.download_file_from_google_drive(file_id=duke2ghana_id, 
                                    dest_path=dest_path, unzip=True)
os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))