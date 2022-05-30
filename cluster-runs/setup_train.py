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
    fake_maxar_train_id = '1-1NjCRjZPQOg3-cHGLEkj4jQhUuhfgMU'
    fake_maxar_val_id = '1WtQvQDm_KUCJ1c1QgU7gJA3bjnFqTztf'
    manual_maxar_val_id = '1-52lUr-T5AA2iMMakumgijaZ6i7Sf5Ns'

    duke_512_train_id = '1W3JptxEo9AK3YiY8mX47PmYMeEmNzn6Z'
    duke_512_val_id = '1QtaShvrrY9LM3YJa8K2wt0ZAUkNRR9U0'
    
    fake_australia_512_train_id = '1-0BJUVvkrjGzOkTdhRnygyp1G54AtCFU'
    fake_australia_512_val_id = '1-3GifZ1wE98PSpDg9ft_yo9ffKyY-hCY'
    australia_val_id = '1AQWAhwmgntPk9P046RD6Y6V66uXInTFY'

    transmission_04_train_id = '14Ytg2HuvyvDGQgr1mIWNojPrLOKFEjXK'

    model_id = '1-8VQMy0lI4QKW8hxOKzPv10TPoslC1kc'

    # download datasets
    dest_path = os.path.join(os.getcwd(), 'datasets', 'hold')
    '''
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

    gdd.download_file_from_google_drive(file_id=fake_maxar_val_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

    gdd.download_file_from_google_drive(file_id=fake_maxar_train_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

    gdd.download_file_from_google_drive(file_id=manual_maxar_val_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

    gdd.download_file_from_google_drive(file_id=duke_512_train_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

    gdd.download_file_from_google_drive(file_id=duke_512_val_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))
    gdd.download_file_from_google_drive(file_id=fake_australia_512_train_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

    gdd.download_file_from_google_drive(file_id=fake_australia_512_val_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))

    gdd.download_file_from_google_drive(file_id=australia_val_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))
    '''
    gdd.download_file_from_google_drive(file_id=transmission_04_train_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'datasets', 'hold'))
    # download model
    '''
    gdd.download_file_from_google_drive(file_id=model_id, 
                                        dest_path=dest_path, unzip=True)
    os.remove(os.path.join(os.getcwd(), 'models', 'hold'))
    '''



if __name__ == '__main__':
    setup_detectron()