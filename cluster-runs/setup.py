import os
from google_drive_downloader import GoogleDriveDownloader as gdd


def setup_detectron(root=''):

    path = os.path.join(os.getcwd(), 'cluster-runs')
    os.chdir(path)

    # make datasets dir
    if os.path.isdir('datasets') is not True:
        os.mkdir('datasets')

    # make dir for models
    if os.path.isdir('models') is not True:
        os.mkdir('models')

    # downloads
    duke_train_id = '1-FZdNPoId9uU5FS_b4doZAV-fEC27LGy'
    duke_val_id = '1-17Mll4llFVy5w5o0EFaMrVejVWbPM2j'
    maxar_train_id = '1bW87dtWLwcvkLk-Wq0Z3BAk5deVKTJPw'
    maxar_val_id = '1-5eScmX0bzcW2wUIEyU8o8hjtAKJNXZN'

    model_id = '1-8VQMy0lI4QKW8hxOKzPv10TPoslC1kc'

    # download datasets
    dest_path = os.path.join(os.getcwd(), 'datasets', 'hold')
    gdd.download_file_from_google_drive(file_id=maxar_val_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))
                                    
    gdd.download_file_from_google_drive(file_id=maxar_train_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

    gdd.download_file_from_google_drive(file_id=duke_val_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

    gdd.download_file_from_google_drive(file_id=duke_train_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))


    # download model
    gdd.download_file_from_google_drive(file_id=model_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'models', 'hold'))



if __name__ == '__main__':
    setup_detectron()