# mini script to download files from google drive using the 'googledrivedownloader' package

from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='FILE ID PROVIDED IN TEAMS',
                                    dest_path='./../data/sample.zip',
                                    unzip=True)