from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build

drive_service = build('drive', 'v3')
def save_fig_to_drive(fig_fname):
  file_metadata = {
    'name': fig_fname,
    'mimeType': 'application/pdf'
  }


  media = MediaFileUpload(fig_fname, 
                        mimetype='application/pdf',
                        resumable=True)
  created = drive_service.files().create(body=file_metadata,
                                       media_body=media,
                                       fields='id').execute()
  print('File ID: {}'.format(created.get('id')))